import pickle
import datetime
import requests
import pandas as pd
import numpy as np
import simpy
import random
from statistics import mean
import joblib
from scipy.stats import rv_continuous
import matplotlib.pyplot as plt
import seaborn
import statistics as stc

with open('monitoring_data_12mnth.pkl', 'rb') as f:
    dataset = pickle.load(f)

data = dataset.loc[dataset['birthday'] >= '1850-12-31']
error_data = dataset.loc[dataset['birthday'] <= '1850-12-31']

#Расчет возраста пациента
data['birthday'] = pd.to_datetime(data['birthday'])
NOW = datetime.datetime.now()
data['age'] = round((NOW-data['birthday'])/ np.timedelta64(1, 'Y'),0)

#Список всех уникальных ID с корректной датой рождения
patnum = data['patient_id'].unique()

dsah = pd.read_csv('full_info_log_2018-2019.csv',',',index_col=['Unnamed: 0'],encoding='Windows-1251')

def meas_missing_ratio(series):
    return len(series.loc[series=='Отсутствие измерений'])/len(series)

miss_prob = dsah.groupby('ID пациента')['Название события'].agg(meas_missing_ratio)
miss_prob.head() # вероятности пропуска

countlendata = {}
for i in ppp:
    countlendata[i] = len(dsah[dsah['ID пациента'] == i])
countlen = pd.DataFrame(countlendata.values())

plt.title('Количество событий по каждому пациенту')
plt.hist(countlen)
plt.axvline(stc.median(countlendata.values()), color='k', linestyle='dashed', linewidth=1)

workid = []
for i in uniqid:
    if len(dsah[(dsah['ID пациента'] == i)]) >= 50:
        for j in patnum:
            if i == j:
                workid.append(i)
            else:
                pass

plt.title('Распределение показателей систолического \n давления для пациента id 2496')
plt.ylabel('Количество измерений')#('Давление, мм.рт.ст')
plt.xlabel('Давление, мм.рт.ст')#('Номер измерения')
dsah[dsah['ID пациента']==2496]['САД'].hist()

class rvData(rv_continuous):
    data = np.sort(np.random.rand(100))
    def __init__(self, initdata, *args):
        self.data = np.sort(initdata)
        super().__init__(args)
    def _cdf(self, x, *args):
        idx = int((self.data < x).sum())
        if idx == 0:
            return 0.0
        if idx >= len(self.data):
            return 1.0
        return (idx - 1.0 + (x - self.data[idx - 1]) / (self.data[idx] - self.data[idx - 1])) / len(
            self.data)

#Функция расчета ДАД от среднего значения по САД
def diast_ad_calc(sad, meanad):
    dad = 3*(meanad - 1/3*sad)/2
    return dad

#Функция расчета ЧСС
def max_chss_calc(male, age):
    if male == 'жен':
        max_chss = 206 - (0.88 * age) #Формула Марты Гулати для женщин
    elif male == 'муж':
        max_chss = 208 - (0.7 * age) #Формула Танака
    return max_chss

# Функция расчета средней реальной вариабельности давления (ARV)
def arv(true_press, pat_press):
    arv_val = 0
    arv_pat_val = []
    for i in range(1, len(true_press)):
        arv_val += abs(true_press[i][0] - true_press[i-1][0])
    arv_val += abs(true_press[-1][0] - pat_press[-1][0])
    arv_val = round(arv_val/len(true_press), 2)
    arv_pat_val += [arv_val]
    return arv_pat_val

df = data.loc[data['patient_id'].isin(workid)]
miss_prob = miss_prob.loc[miss_prob.index.isin(workid)]
miss_prob.head() # вероятности пропуск

plt.title('Распределение доли ЧСС для пациента id 2496')
plt.ylabel('Частота')#('Давление, мм.рт.ст')
plt.xlabel('Доля к максимальному ЧСС')#('Номер измерения')
df[df['patient_id']==6337]['perc_chss'].hist()

#Расчет среднего артериального давления, ЧССмакс по всем пациентам
df['mean_ad'] = df[['tonometry_data_systolic_bp', 'tonometry_data_diastolic_bp']].apply(
    lambda x:(x['tonometry_data_systolic_bp']*1/3 + x['tonometry_data_diastolic_bp']*2/3) , axis=1)
df['perc_chss'] = round(df['tonometry_data_heart_rate']/ max_chss_calc(df['male'].values[0], df['age'].values[0]), 3)

partRZ = {}
for i in workid:
    partRZ[i] = len(dsah[(dsah['ID пациента'] == i) & ((dsah['Название события'] == 'Красная зона') |
                                   (dsah['Название события'] == 'Красная зона (Отсутствует контрольное измерение)') |
                                  (dsah['Название события'] == 'Красная зона (ХСН)'))])\
                / len(dsah[dsah['ID пациента'] == i])
RZ = pd.DataFrame(partRZ.values())

plt.title('Distribution of red zones for patients')
plt.hist(RZ)
plt.axvline(stc.median(partRZ.values()), color='k', linestyle='dashed', linewidth=1)

ctrlAH = []
unctrlAH = []
for i in partRZ:
    if partRZ[i]<= 0.8:
        ctrlAH.append(i)
    elif partRZ[i]> 0.8:
        unctrlAH.append(i)

def make_generator(array):
    for i in range(len(array)):
        yield array[i]

gen = make_generator([1 if rd_val < 0.1 else 0 for rd_val in np.random.uniform(0, 1, 2*30)])
next(gen)


def modeling_patient(id_pat):
    df_patient = df[df['patient_id'] == id_pat]
    male = df_patient['male'].values[0]
    age = df_patient['age'].values[0]
    meansad = df_patient['tonometry_data_systolic_bp'].mean()
    sadvar = df_patient['tonometry_data_systolic_bp'].values

    sad = rvData(df_patient['tonometry_data_systolic_bp'])
    meanad = rvData(df_patient['mean_ad'])
    perc_chss = rvData(df_patient['perc_chss'])

    # Запись данных сгенерированного давления
    pat_press = []
    # Запись данных ошибочных измерений
    err = []
    err_old_rule = []
    err_mean = []

    morning_time = 10  # Время утреннего замера давления
    evening_time = 20  # Время вечернего замера давления

    # Массив корректных значений давления пациента
    true_press = []
    true_press_old_rule = []
    true_press_mean = []
    # Запись данных ARV
    arv_pat = []
    # Количество дней
    days = 30
    N = 5

    # какие измерения будут пропущены: 1 - пропуск, 0 - нет
    #     missing_meas_flags = [1 if rd_val < missing_prob else 0 for rd_val in np.random.uniform(0, 1, 2*days)]
    #     gen = make_generator(missing_meas_flags)

    # Начало моделирования
    env = simpy.Environment()

    class Patient:
        def __init__(self, env):
            self.env = env
            self.take_measure = env.event()
            self.pat_answ = [env.process(self.patients())]
            self.time_proc = env.process(self.timemeasure())
            self.pat_error = [env.process(self.error_message())]
            self.measure_done = env.event()

        def timemeasure(self):
            while True:
                self.take_measure.succeed()
                self.take_measure = self.env.event()
                yield self.env.timeout(24)

        def patients(self):
            while True:
                yield self.take_measure
                yield self.env.timeout(morning_time)
                syst_ad = sad.rvs()
                dias_ad = diast_ad_calc(syst_ad, meanad.rvs())
                while dias_ad >= syst_ad:
                    dias_ad = diast_ad_calc(syst_ad, meanad.rvs())
                if not next(gen):  # !!! добавила
                    pat_press.append([int(syst_ad), int(dias_ad), int(max_chss_calc(male, age) * perc_chss.rvs())])
                self.measure_done.succeed()
                self.measure_done = self.env.event()

                yield self.env.timeout(evening_time - morning_time)
                syst_ad = sad.rvs()
                dias_ad = diast_ad_calc(syst_ad, meanad.rvs())
                while dias_ad >= syst_ad:
                    dias_ad = diast_ad_calc(syst_ad, meanad.rvs())
                if not next(gen):  # !!! добавила
                    pat_press.append([int(syst_ad), int(dias_ad), int(max_chss_calc(male, age) * perc_chss.rvs())])
                self.measure_done.succeed()
                self.measure_done = self.env.event()

        def error_message(self):
            #             global pat_press, true_press, true_press_old_rule,true_press_mean,\
            #             err,err_old_rule,err_mean, arv_pat
            while True:
                yield self.measure_done
                if len(len_pat_press) != len(pat_press):  # сгенерировано новое измерение !!!!
                    for i in range(len(pat_press) - len(len_pat_press)):
                        len_pat_press.append(1)  # !!! сравняем длину массивов
                # Проверка по правилу на основе ВАД
                # Получение первого измерения
                if len(pat_press) <= N:
                    true_press.append(pat_press[-1])
                    arv_pat.append(arv(true_press, pat_press)[-1])
                    err.append(0)
                # проверяю вхождения в доверительный интервал
                elif (abs(pat_press[-1][0] - true_press[-1][0]) <= (arv_pat[-1] + 3 * ((np.var(arv_pat)) ** (1 / 2)))):
                    arv_pat.append(arv(true_press, pat_press)[-1])
                    true_press.append(pat_press[-1])
                    err.append(0)
                else:
                    err.append(1)
                # Проверка на основе среднего значения давления
                if len(pat_press) <= N:
                    true_press_mean.append(pat_press[-1])
                    err_mean.append(0)
                elif (abs(pat_press[-1][0] - meansad) <= 2 * (np.var(sadvar) ** (1 / 2))):
                    true_press_mean.append(pat_press[-1])
                    err_mean.append(0)
                else:
                    err_mean.append(1)

                    # Проверка на основе жестких правил
                if (pat_press[-1][0] <= 135) and (pat_press[-1][0] >= 110):
                    true_press_old_rule.append(pat_press[-1])
                    err_old_rule.append(0)
                else:
                    err_old_rule.append(1)

    pats = Patient(env)
    env.run(days * 24)
    env.now

    return err, err_old_rule, err_mean

import time
res4068 = []
start_time = time.time()
res = joblib.Parallel(n_jobs=6)(
                    joblib.delayed(modeling_patient)(4068)
                    for i_run in range(100))

print("--- %s seconds ---" % (time.time() - start_time))
res4068.extend(res)

def modeling(list_pat):
    result = []
    for i in list_pat:
        result.append(modeling_patient(i))
    return result

res_ctrl = []
start_time = time.time()
res1 = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(modeling)(ctrlAH)
                    for i_run in range(100))

print("--- %s seconds ---" % (time.time() - start_time))
res_ctrl.extend(res1)

res_unctrl = []
start_time = time.time()
res2 = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(modeling)(unctrlAH)
                    for i_run in range(100))

print("--- %s seconds ---" % (time.time() - start_time))
res_unctrl.extend(res2)

sum_ctrl = []
for i in res_ctrl: # перебираем по 100 запускам
    ARV_load = 0
    Base_load = 0
    Mean_load = 0
    ARV = 0
    Base = 0
    Mean = 0
    sumlen = 0
    for j in i: # перебираем по 126 пациентам
        ARV += sum(j[0])
        Base += sum(j[1])
        Mean += sum(j[2])
        sumlen += len(j[0])
    ARV_load += ARV/sumlen
    Base_load += Base/sumlen
    Mean_load += Mean/sumlen
    sum_ctrl.append([ARV_load,Base_load,Mean_load])
sum_ctrl


