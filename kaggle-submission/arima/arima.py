# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.colors as mcolors
import time
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import warnings
import math

states = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Florida', 'Georgia', 'Idaho',
    'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Mississippi', 'Missouri',
    'Montana', 'Nebraska', 'Nevada', 'New Mexico', 'North Carolina', 'North Dakota',
    'Ohio', 'Oklahoma', 'Oregon', 'South Carolina', 'South Dakota', 'Tennessee',
    'Texas', 'Utah', 'Virginia', 'Washington', 'Wisconsin', 'Wyoming',]

# if use all states:
df = pd.read_csv('data/train.csv')
states = list(set(df['Province_State'].values))

# order is the param for Arima
orders = [
    (1, 1, 1),
    (1, 1, 2),
    (1, 2, 1),
    (1, 2, 2),
    (2, 1, 1),
    (2, 1, 2),
    (2, 2, 1),
    (2, 2, 2)
]


def train_arima(state, feature, order):
    """### ARIMA"""
    traindf = pd.read_csv('data/train.csv')
    arima_train = traindf.copy()
    arima_train.index = pd.DatetimeIndex(traindf.Date, freq = 'infer')
    arima_train = arima_train.drop(["Date"], axis=1)

    arima_death_sample = arima_train[arima_train.Province_State == state].Deaths
    arima_confirm_sample = arima_train[arima_train.Province_State == state].Confirmed
    if feature == 'Death':
        additive = seasonal_decompose(arima_death_sample, model='additive', extrapolate_trend='freq')
    elif feature == 'Confirmed':
        additive = seasonal_decompose(arima_confirm_sample, model='additive', extrapolate_trend='freq')
    additive_df = pd.concat([additive.seasonal, additive.trend, additive.resid, additive.observed], axis=1)
    additive_df.columns = ['seasonal', 'trend', 'resid', 'actual_values']

    # plt.rcParams.update({'figure.figsize': (8,8)})
    # additive.plot().suptitle('Additive Decompose')
    # plt.show()

    trend = additive.trend
    result = adfuller(trend.values)
    # print('ADF Statistic: %f' % result[0])
    # print('p-value: %f' % result[1])
    #p val ~= 0.99 > 0.05 so trend is stationary

    # from above, p=1, d=2, q=1
    train = trend
    # print(trend)
    model = ARIMA(train, order=order)
    try:
        model = model.fit(disp=0)
    except:
        return "invalid param"
    # print(model.summary())
    try:
        summary = model.summary()
    except:
        return "invalid param 2"
    print(state)
    # print(summary.tables[1][2][4])
    # print(len(summary.tables[1]))
    p_values = []
    for i in range(1, len(summary.tables[1])):
        p_values += [float(str(summary.tables[1][i][4]))]
    print(p_values)
    return str(p_values)
    # #check P>|z| under ar. and ma. to see if model params fit
    # residuals = pd.DataFrame(model.resid)
    # fig, ax = plt.subplots(1,2)
    # plt.rcParams.update({'figure.figsize': (15, 8)})
    # residuals.plot(title="Residuals", ax=ax[0])
    # residuals.plot(kind='kde', title='Density', ax=ax[1])
    # plt.show()


def get_p_vals():
    traindf = pd.read_csv('data/train.csv')
    # states = list(set(traindf['Province_State'].values))
    res = {}
    res_confirm = {}
    res_death = {}
    for order in orders:
        p_values_confirm = {}
        p_values_death = {}
        for state in states:
            p_vals = train_arima(state, 'Confirmed', order)
            p_values_confirm[state] = p_vals

            p_vals = train_arima(state, 'Death', order)
            p_values_death[state] = p_vals
        # print(p_values_confirm)
        # print(p_values_death)
        res[str(order) + '-' + 'confirmed'] = p_values_confirm
        res[str(order) + '-' + 'death'] = p_values_death
        res_confirm[str(order)] = p_values_confirm
        res_death[str(order)] = p_values_death
    print(res)
    df = pd.DataFrame(res)
    df.to_excel('output/res.xlsx')
    confirm_df = pd.DataFrame(res_confirm)
    confirm_df.to_csv('output/res_confirm.csv')
    confirm_df.to_excel('output/res_confirm.xlsx')
    death_df = pd.DataFrame(res_death)
    death_df.to_csv('output/res_death.csv')
    death_df.to_excel('output/res_death.xlsx')


def calc_mean(string):
    if 'invalid param' in string:
        return False
    else:
        lst = string[1:-1].split(', ')
        nums = [float(i) for i in lst]
        aver = np.mean(nums)
        return aver

def find_best_param(df):
    state_to_best_param = {}
    for state in states:
        lowest = 1
        best_param = 'None'
        for order in orders:
            mean = calc_mean(df.loc[state, str(order)])
            if mean < lowest:
                lowest = mean
                best_param = order
        state_to_best_param[state] = best_param
    print(state_to_best_param)
    res_df = pd.DataFrame(state_to_best_param)
    res_df.to_csv('output/best_param.csv')
    return state_to_best_param
    

def forecast(state, feature, order, submission_df):
    """### ARIMA"""
    traindf = pd.read_csv('data/train.csv')
    arima_train = traindf.copy()
    arima_train.index = pd.DatetimeIndex(traindf.Date, freq='infer')
    arima_train = arima_train.drop(["Date"], axis=1)

    arima_death_sample = arima_train[arima_train.Province_State == state].Deaths
    arima_confirm_sample = arima_train[arima_train.Province_State == state].Confirmed
    if feature == 'Deaths':
        additive = seasonal_decompose(arima_death_sample, model='additive', extrapolate_trend='freq')
    elif feature == 'Confirmed':
        additive = seasonal_decompose(arima_confirm_sample, model='additive', extrapolate_trend='freq')
    additive_df = pd.concat([additive.seasonal, additive.trend, additive.resid, additive.observed], axis=1)
    additive_df.columns = ['seasonal', 'trend', 'resid', 'actual_values']

    trend = additive.trend
    result = adfuller(trend.values)
    train = trend
    model = ARIMA(train, order=order)
    try:
        model = model.fit(disp=0)
    except:
        return submission_df    # TODO should process the situation that the param is invalid. return submission here is only for the following iterations to be done.

    full_testdf = pd.read_csv('data/test.csv')
    test = full_testdf.copy()
    test = test[test['Province_State'] == state]
    test.index = test['Date']
    test = test.drop(['Date'], axis=1)
    # print(test)
    # Forecast: 192 forecasting values with 95% confidence
    fc, se, conf = model.forecast(test.shape[0], alpha=0.05)
    # Make as pandas series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(conf[:, 0], index=test.index)
    upper_series = pd.Series(conf[:, 1], index=test.index)
    # print(fc_series)
    # print(fc_series.index)

    # write into submission df
    for index in fc_series.index:
        sub_index = submission_df[submission_df['Province_State'] == state][submission_df['Date'] == index].ForecastID
        # print(sub_index)
        # state_df = submission_df[submission_df['Province_State'] == state]
        feature_index = list(submission_df.columns).index(feature)
        submission_df.iloc[sub_index, feature_index] = fc_series[index]
    # print(state_df)
    # print(submission_df)

    # Plot
    plot = False
    if plot:
        plt.figure(figsize=(24, 10), dpi=100)
        # plt.plot(arima_death_sample, label='training')
        plt.plot(fc_series, label='forecast')
        plt.xticks(rotation=-15)
        # plt.fill_between(lower_series.index, lower_series, upper_series,
        #                  color='k', alpha=.15)
        plt.title('Forecast vs Actuals')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()

    return submission_df


if __name__ == '__main__':
    start_time = time.time()
    warnings.filterwarnings('ignore')
    if not os.path.exists('output'):
        os.makedirs('output')
    ## Comment the next line if alrady ran once and have 'output/res_confirm.csv' and 'output/res_death.csv'
    get_p_vals()

    # train_arima('California', 'Confirmed', (1, 1, 1))
    # train_arima('California', 'Confirmed', (1, 2, 1))
    confirm_df = pd.read_csv('output/res_confirm.csv', index_col=0)
    state_to_best_confirm = find_best_param(confirm_df)
    death_df = pd.read_csv('output/res_death.csv', index_col=0)
    state_to_best_death = find_best_param(death_df)

    submission_df = pd.read_csv('data/test.csv')
    for state in states:
        order = state_to_best_confirm[state]
        # print('----SUBMISSION DF-----')
        # print(submission_df)
        submission_df = forecast(state, 'Confirmed', state_to_best_confirm[state], submission_df)
        submission_df = forecast(state, 'Deaths', state_to_best_death[state], submission_df)
    print(submission_df)
    submission_df.to_csv('output/arima_submission.csv')
    end_time = time.time()
    print('======= Time taken: %f =======' %(end_time - start_time))
