#!/usr/bin/env python
# coding:utf-8
# -*- coding: utf-8 -*-

import numpy as np
from pandas import Series
import pandas as pd

s = Series([1, 2, 3, 4, 5])

# print(s.head())
# periods 周期
# freq 频率
rng = pd.date_range('1/1/2018', periods=10, freq='H')
# print(rng)

ts = Series(np.random.randn(len(rng)), rng)
# print(ts)
# 更改频率和填充间隙
converted = ts.asfreq('2H', method='pad')
# print(converted)
# print("-------------------------------------------------")
# 重新取样
ts = ts.resample(rule='3H').mean()
# print(ts)

# 时间戳
tstamp = pd.Timestamp('2018-01-01 03:00:00')
# 时间区间
period = pd.Period('2018-01-02', freq='D')

print(period)










