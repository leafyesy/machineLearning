#!/usr/bin/env python
# coding:utf-8

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False


def date_format(dt):
    # dt显示是一个series/tuple；dt[0]是date，dt[1]是time
    import time
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


def draw_pic(x_test, y_test, y_predict, title):
    t = np.arange(len(x_test))
    plt.figure(facecolor='w')
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实值')
    plt.plot(t, y_predict, 'g-', linewidth=2, label=u'预测值')
    plt.legend(loc='lower right')
    plt.title(title, fontsize=20)
    plt.grid(b=True)
    plt.show()


# 最小二乘公式
def least_squares(X, Y, d=0.0):
    X_T_X = X.T * X
    eye = d * np.eye(X_T_X.shape[0], dtype=int)
    print(X.shape)
    print(Y.shape)
    print(X_T_X.shape)
    print(eye.shape)
    theta = (X_T_X + eye).I * X.T * Y
    print(theta.shape)
    return theta


# 时间和功率的关系
def date_power(x_train, x_test, Y, y_test):
    X = np.mat(x_train)
    theta = least_squares(X, Y, 1)
    print(theta)
    y_hat = np.mat(x_test) * theta
    draw_pic(x_test, y_test, y_hat, u'功率预测1')


def date_intensite(x_train, x_test, Y, y_test):
    X1 = np.mat(x_train).reshape(-1, 1)
    # print(X1)
    theta2 = least_squares(X1, Y, 1)
    print(theta2)
    y_hat2 = np.mat(x_test).reshape(-1, 1) * theta2
    draw_pic(x_test2, y_test, y_hat2, u'功率预测2')


# date 日期
# time 时间
# Global_active_power 有功功率
# Global_reactive_power 无功功率
# Voltage 电压
# Global_intentsity 电流
# sub_metering_1 厨房电功率
# sub_metering_2 洗衣机电功率
# sub_metering_3 热水器的电功率
path = 'datas/household_power_consumption_1000.txt'
df = pd.read_csv(path, ';', low_memory=False)

# 处理数据成数值

new_df = df.replace('?', np.nan)
# drop下降 axis 轴
datas = new_df.dropna(axis=0, how='any')  # 只要有一个数据为空，就进行行删除操作

# print(new_df.info())


dt = new_df.iloc[:, 0:2]
intensity = new_df.iloc[:, 5]

# print(intensity)
dt = dt.apply(lambda x: pd.Series(date_format(x)), axis=1)
global_active_power = new_df.iloc[:, 2]
# print(new_df)
# print(dt)
# print(global_active_power)

# 对数据进行分类
x_train, x_test, x_train2, x_test2, global_active_power_train, global_active_power_test = train_test_split(dt,
                                                                                                           intensity,
                                                                                                           global_active_power,
                                                                                                           test_size=0.2,
                                                                                                           random_state=0)
Y = np.mat(global_active_power_train).reshape(-1, 1)

# -------------时间和功率的关系---------------------------
# date_power(x_train, x_test, Y, global_active_power_test)

# ------------------------------------------------------end 时间和功率的关系

# -----------------时间和电流的关系-------------------------
date_intensite(x_train2, x_test2, Y, global_active_power_test)
# X1 = np.mat(x_train2).reshape(-1, 1)
# # print(X1)
# theta2 = least_squares(X1, Y, 0.1)
# # print(theta2)
# y_hat2 = np.mat(x_test2) * theta2
# draw_pic(x_test2, global_active_power_test, y_hat2, u'功率预测2')
