# 建立了回归器, 之后最重要的就是如何评价回归器的拟合效果. 使用误差(error)表示实际值与模型预测值之间的差值.
# 拟合效果的准确性使用指标(metric)表示. 回归器可以使用许多的指标进行衡量.

#   平均绝对误差(mean absolute error): 这是给定数据集的所有数据点的绝对误差平均值.
#   均方误差(mean square error): 这是给定数据集的所有数据点的误差的平方的均值. 这是最流行的指标之一.
#   中位数绝对误差(median absolute error): 这是给定数据集的所有数据点的误差的中位数. 这个指标的主要有点是可以消除异常值(outlier)的干扰. 测试数据集中的单个坏点不会影响整个误差指标.
#                                         均值误差指标会收到异常点的影响.
#   解释方差分(explained variance score): 这个分数用于衡量我们的模型对数据集波动的解释能力. 如果得分为1.0, 那么表明我们的模型是完美的.
#   R方得分(R2 score): 其是指确定性的相关嬉耍, 用于衡量模型对未知样本预测的效果. 最好的得分是1.0, 值也可以是负数.

import numpy as np
import sklearn.metrics as sm

# 两个完全相符的集合, 输出的就是最好的值
y_test = np.arange(1, 100)
y_test_pred = np.arange(1, 100)

print("Mean absolute error =", round(sm.mean_absolute_error(y_true=y_test, y_pred=y_test_pred), 2))
print("Mean square error =", round(sm.mean_squared_error(y_true=y_test, y_pred=y_test_pred), 2))
print("median absolute error =", round(sm.median_absolute_error(y_true=y_test, y_pred=y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_true=y_test, y_pred=y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_true=y_test, y_pred=y_test_pred), 2))
