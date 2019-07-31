import sys
import json

import requests
import numpy as np
import pandas as pd
from sklearn import cluster, covariance

have_one = False

def data_save(url):
    global have_one
    if have_one:
        return
    res = requests.get(url)
    if res.status_code != 200:
        print("未请求到文件", file=sys.stderr)
    with open("../tmp/股票数据长啥样.csv", 'wb') as file:
        file.write(res.content)
    have_one = True


symbol_dir = "./dat/symbol_map.json"
symbol_dict = None
quotes = list()

# 加载符号映射信息
with open(symbol_dir, 'r') as file:
    symbol_dict = json.loads(file.read())

symbols, names = np.array(sorted(symbol_dict.items())).T

# 获取数据信息
for symbol in symbols:
    print("Fetching quote history for %r" % symbol)
    url = ("https://raw.githubusercontent.com/scikit-learn/examples-data/master/financial-data/{}.csv")   # 03-08年的
    data_save(url.format(symbol))
    quotes.append(pd.read_csv(url.format(symbol)))

close_prices = np.vstack([q["close"] for q in quotes])
open_prices = np.vstack([q["open"] for q in quotes])

# 计算每日股价波动(收盘价-开盘价)
variation = close_prices - open_prices

# 从相关性中建立协方差图模型
edge_model = covariance.GraphicalLassoCV(cv=5)

# 数据标准化
# using correlations rather than covariance is more efficient for structure recovery
X = variation.copy().T
X /= X.std(axis=0)

# 训练模型
# with np.errstate(invalid='ignore'):
edge_model.fit(X)

# 使用affinity propagation, 用临近传播算法家里聚类模型
_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()

# 打印聚类结果
for i in range(n_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))

