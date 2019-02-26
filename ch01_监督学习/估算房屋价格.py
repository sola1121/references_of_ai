import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

# 使用sklearn中的接口获取模拟数据, 也可以在https://archive.ics.uci.edu/ml中下载
from sklearn import datasets   # 使用模拟数据
housing_data = datasets.load_boston()
# print(housing_data)
print("影响参数:\n", housing_data.data)
print("价格$:\n", housing_data.target)

# 把输入数据和输出数据分成不同的变量. 通过shuffle函数把数据顺序打乱.
X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

# 生成训练数据集和测试数据集
num_training = int(0.8*len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# 拟合一个决策树
# 选择最大深度为4, 这样可以限制决策树不变成任意深度
dt_regressor = DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(X_train, y_train)

# 代如AdaBoost算法的决策树回归模型进行拟合
ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
ab_regressor.fit(X_train, y_train)

# 使用决策树回归器模型预测
y_pred_dt = dt_regressor.predict(X_test)
# 使用AdaBoost算法改善决策树回归器模型的预测
y_pred_ab = ab_regressor.predict(X_test)


# 对结果的进行评价
mse_dt = mean_squared_error(y_test, y_pred_dt)
evs_dt = explained_variance_score(y_test, y_pred_dt)
print("Decision Tree performance")
print("均方误差:", mse_dt)
print("解释方差分:", evs_dt)

mse_ab = mean_squared_error(y_test, y_pred_ab)
evs_ab = explained_variance_score(y_test, y_pred_ab)
print("\nAdaboost enhanced performance")
print("均方误差:", mse_ab)
print("解释方差分:", evs_ab)


def plot_feature_importances(feature_importances, title, feature_names):
    """用于画出特征的重要性"""
    # 将重要性值标准化
    feature_importances = 100.0 * (feature_importances / max(feature_importances))
    # 将得分从高到底排序
    index_sorted = np.flipud(np.argsort(feature_importances))   # 使用argsort是由小到大返回标, 使用flipud翻转顺序
    # 让X坐标上的标签居中显示
    pos = np.arange(index_sorted.shape[0]) + 0.5   # shape这里是array的形状, 取第一个就是列的个数, 即数据多少.
    # 画条形图
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted], rotation=45)
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()

print("\n决策树使用的各个指标占有率:\n", dt_regressor.feature_importances_)
print("\nAdaBoost算法优化后使用的各个指标占有率:\n", ab_regressor.feature_importances_)

plot_feature_importances(dt_regressor.feature_importances_, "Decision Tree regressor", housing_data.feature_names)
plot_feature_importances(ab_regressor.feature_importances_, "AdaBoot regressor", housing_data.feature_names)
