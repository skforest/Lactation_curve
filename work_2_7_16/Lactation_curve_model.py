#!/usr/bin/env python
# encoding: utf-8
'''
@author: zst
@file: Lactation_curve_model.py
@date 2019-09-18 17:15
@desc:
'''

import pandas as pd
import pymssql
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import joblib




def build_linear_model(data, LactationNumber, database):
    """

    :param data: dataframe 奶量数据
    :param LactationNumber: int 胎次
    :param database: str 数据库
    :return: Lactation_curve_1_x6_H1170.model 模型
    """

    # 极端异常值
    std = data['DayYield'].std()
    mean = data['DayYield'].mean()

    Q1 = data['DayYield'].quantile(0.25)
    Q3 = data['DayYield'].quantile(0.75)
    IQR = Q3 - Q1
    # 极端异常值上限,下限
    upper_extreme = Q3 + 3 * IQR
    lower_extreme = Q1 - 3 * IQR
    extreme_outlier = [lower_extreme, upper_extreme]

    # 温和异常值上限
    upper_mild = Q3 + 1.5 * IQR
    lower_mild = Q1 - 1.5 * IQR
    mild_outlier = [lower_mild, upper_mild]

    # 去除缺失数据
    data = data.dropna(how='any')

    # 去除重复数据
    data = data.drop_duplicates(keep=False)


    # X = preprocessing.scale(df[['MilkDay']].values)
    # y = preprocessing.scale(df['DayYield'].values)
    X = data[['MilkDay']].values
    y = data['DayYield'].values

    # 模型
    regr = LinearRegression()
    ploy = PolynomialFeatures(degree=6)
    X_6 = ploy.fit_transform(X)
    regr = regr.fit(X_6, y)

    # 保存模型
    model_name = 'Lactation_curve_' + LactationNumber + '_' + database + '.model'
    joblib.dump(regr, model_name)

    # 预测结果
    y_six_pre = regr.predict(X_6)

    ploy_r2 = r2_score(y, y_six_pre)
    ploy_MSE = mean_squared_error(y, y_six_pre)
    print('r^2',ploy_r2)
    print('MSE',ploy_MSE)


if __name__ == "__main__":
    sql_use = """
    SELECT Yield_Id, EarNum, MilkDate, CONVERT(VARCHAR(10),CalvingDate,120) AS CalvingDate, LactationNumber, MilkDay, DayYield
    FROM DairyCow_Yield WHERE LactationNumber =1 AND DayYield >5 AND MilkDay < 306 AND NOT (Season1 IS NULL OR Season2 IS NULL OR Season3 IS NULL)
    ORDER BY MilkDay
    """
    # sql_use = """
    # SELECT Yield_Id, EarNum, MilkDate, CONVERT(VARCHAR(10),CalvingDate,120) AS CalvingDate, LactationNumber, MilkDay, DayYield
    # FROM DairyCow_Yield WHERE LactationNumber =2 AND DayYield >5 AND MilkDay > 4 AND MilkDay < 306 AND NOT (Season1 IS NULL OR Season2 IS NULL OR Season3 IS NULL)
    # ORDER BY MilkDay
    # """
    # sql_use = """
    # SELECT Yield_Id, EarNum, MilkDate, CONVERT(VARCHAR(10),CalvingDate,120) AS CalvingDate, LactationNumber, MilkDay, DayYield
    # FROM DairyCow_Yield WHERE LactationNumber =3 AND DayYield >5 AND MilkDay > 4 AND MilkDay < 306 AND NOT (Season1 IS NULL OR Season2 IS NULL OR Season3 IS NULL)
    # ORDER BY MilkDay
    # """
    server = "43.242.49.7:3789"
    user = "sa"
    password = "huajie789$%^"
    # 数据库
    database = "x6_H1170"
    # 胎次
    LactationNumber = "1"
    conn = pymssql.connect(host=server, user=user, password=password, database=database, charset='utf8', as_dict=True)

    data = pd.read_sql(sql_use, con=conn)
    conn.close()
    # 模型训练
    build_linear_model(data, LactationNumber, database)











