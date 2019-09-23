#!/usr/bin/env python
# encoding: utf-8
'''
@author: zst
@file: Locatation_curve_prediction.py
@date 2019-09-18 15:59
@desc:
'''


import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import joblib


def prediction(LactationNumber, database):
    # 305预测

    X = np.arange(1, 306)[:, np.newaxis]
    # 保加载模型
    model_name = 'Lactation_curve_' + str(LactationNumber) + '_' + database + '.model'
    regr = joblib.load(model_name)

    # 预测值
    ploy = PolynomialFeatures(degree=6)
    X_6 = ploy.fit_transform(X)
    y_six_pre = regr.predict(X_6)
    return y_six_pre

def prediction_yield(LactationNumber, database):
    """

    :param LactationNumber: int 胎次
    :param database: str 数据库
    :return:
            y_six_pre: list 305天单产预测
    """
    y_six_pre = prediction(LactationNumber, database)
    pre_yield = y_six_pre.tolist()
    return pre_yield


def locatation_alpha(t, LactationNumber, database):
    """
    泌乳天数的校正系数
    :param t: int 泌乳天数
    :param LactationNumber: int 胎次
    :param database: str 数据库
    :return: alpha：int 泌乳系数
            dict_every：dict 泌乳天数之后的预测值
    """

    y_six_pre = prediction(LactationNumber, database)
    dict_every = {}
    for key, value in zip(map(str, range(t+1, 306)), y_six_pre[t:]):
        dict_every[str(key)] = value
    # print(dict_every)

    # 泌乳天数的校正系数
    f_305= np.sum(y_six_pre)
    f_t = np.sum(y_six_pre[0:t])
    alpha = f_305 / f_t

    return alpha, dict_every


if __name__ == "__main__":
    # 数据库
    database = "x6_H1170"
    # 胎次
    LactationNumber = 1
    # 泌乳天数
    t = 200
    # 校正系数，泌乳天数之后的预测值
    alpha, pre = locatation_alpha(t, LactationNumber, database)
    print(alpha)
    print(pre)
    pre_yield = prediction_yield(LactationNumber, database)
    print(pre_yield)