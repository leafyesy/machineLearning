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

path = 'datas/household_power_consumption_1000.txt'
df = pd.read_csv(path, ';', low_memory=False)

# 处理数据成数值

new_df = df.replace('?', np.nan)
# drop下降 axis 轴
datas = new_df.dropna(axis=0, how='any')  # 只要有一个数据为空，就进行行删除操作

print()


# print(new_df.info())

def date_format(dt):
    # dt显示是一个series/tuple；dt[0]是date，dt[1]是time
    import time
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


dt = new_df.iloc[:, 0:2]
dt = dt.apply(lambda x: pd.Series(date_format(x)), axis=1)
global_active_power = new_df.iloc[:, 2]
# print(new_df)
# print(dt)
# print(global_active_power)

# date 日期
# time 时间
# Global_active_power 有功功率
# Global_reactive_power 无功功率
# Voltage 电压
# Global_intentsity 电流
# sub_metering_1 厨房电功率
# sub_metering_2 洗衣机电功率
# sub_metering_3 热水器的电功率

# 对数据进行分类
x_train, x_test, global_active_power_train, global_active_power_test = train_test_split(dt, global_active_power,
                                                                                        test_size=0.2, random_state=0)

# print(x_train)
# print(x_test)
# print(global_active_power_train)
# print(global_active_power_test)

X = np.mat(x_train)
Y = np.mat(global_active_power_train).reshape(-1, 1)
#
# print(X)
# print(Y)

new_X = X.T * X

theta = (new_X + np.eye(new_X.shape[0], dtype=int)).I * X.T * Y

print(theta)

y_hat = np.mat(x_test) * theta

t = np.arange(len(x_test))
plt.figure(facecolor='w')
plt.plot(t, global_active_power_test, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc='lower right')
plt.title(u"功率预测", fontsize=20)
plt.grid(b=True)
plt.show()
