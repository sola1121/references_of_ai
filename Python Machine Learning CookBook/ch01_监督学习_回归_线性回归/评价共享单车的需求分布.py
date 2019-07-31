import csv

import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score


def plot_feature_importances(feature_importances, title, feature_names):
    """用于画出特征的重要性"""
    # 将重要性值标准化
    feature_importances = 100.0 * (feature_importances / max(feature_importances))
    # 将得分从高到底排序
    # 使用argsort是由小到大返回标, 使用flipud翻转顺序
    index_sorted = np.flipud(np.argsort(feature_importances))
    # 让X坐标上的标签居中显示
    # shape这里是array的形状, 取第一个就是列的个数, 即数据多少.
    pos = np.arange(index_sorted.shape[0]) + 0.5
    # 画条形图
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted], rotation=45)
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()


def load_dataset(filename):
    file_reader = csv.reader(open(filename, "r"), delimiter=",")
    X, y = list(), list()
    for row in file_reader:
        X.append(row[2:13])
        y.append(row[-1])
    # 提取特征名称
    feature_names = np.array(X[0])
    # 将第一行特征名称移除, 仅保留数值
    return np.array(X[1:]).astype(np.float), np.array(y[1:]).astype(np.float), feature_names

# 读取数据, 并打乱顺序
X, y, feature_names = load_dataset("./dat/bike_day.csv")
X, y = shuffle(X, y, random_state=7)

# 将数据分为训练数据和测试数据
num_trianing = int(0.9*len(X))
X_train, y_train = X[:num_trianing], y[:num_trianing]
X_test, y_test = X[num_trianing:], y[num_trianing:]

# 定义随机森林回归器
rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=20, min_samples_split=2)
# 进行训练
rf_regressor.fit(X_train, y_train)

# 使用X_test预测结果
y_pred = rf_regressor.predict(X_test)

# 评价随机森林回归器训练效果
mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
evs = explained_variance_score(y_true=y_test, y_pred=y_pred)
print("Random Forest regressor performance")
print("均方误差:", mse)
print("解释方差分:", evs)

# 数据重要性分布
plot_feature_importances(rf_regressor.feature_importances_, "Random Forest regressor", feature_names)
