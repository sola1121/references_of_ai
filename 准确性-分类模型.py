# 交叉验证(cross validation)
# 为了能让模型更贱稳定, 还需要用数据集的不同子集进行反复的验证. 如果只是对特定的子集进行微调, 最终可能会过度拟合(overfitting)模型.
# 过度拟合的模型在已知的数据集上表现很好, 但是在未知数据集上表现不好.
# 精度(precision), 召回率(recall), F1得分(F1 score), 可以用参数评分标准(parameter scoring)获得各项指标得分.

# 精度是指分类器正确分类的样本数量占分类器总分类样本数量的百分比(分类器分类结果中, 有一次额样本分错了)
#                    精度=分类正确的样本数量/总分类样本数量
# 召回率是指本应正确分类的样本数量占某分类总样本数量的百分比(有一些样本属于某分类, 但分类器没有分出来)
#                    召回率=分类正确的样本数量/数据集中我们感兴趣的样本数量
# 这两个指标是二律背反的, 一个指标值达到100%, 另一个将会表现很差. 需要保持两个的合理高度.
# 为了量化两个指标的均衡性, 引入F1得分指标, 是精度和召回率的合成指标, 实际上是精度和召回率的调和均值(harmonic mean)
#                    F1得分=2*精度*召回率/(精度+召回率)

from sklearn.model_selection import cross_val_score

num_validations = 5   # 拆分策略, 影响最后结果集的到小

def cross_validation(classifer, X, y):
    """将所有数据集带入分类器进行交叉验证"""
    # 准确率
    accuracy = cross_val_score(classifer, X, y=y, scoring="accuracy", cv=num_validations)
    print("Accuracy: ", round(100*accuracy.mean(), 2), "%.")
    # 精度
    precision = cross_val_score(classifer, X, y=y, scoring="precision_weighted", cv=num_validations)
    # 召回率
    recall = cross_val_score(classifer, X, y=y, scoring="recall_weighted", cv=num_validations)
    # F1得分
    f1 = cross_val_score(classifer, X, y=y, scoring="f1_weighted", cv=num_validations)

    return round(precision.mean(), 4), round(recall.mean(), 4), round(f1.mean(), 4)



# 混淆矩阵(confusion matrix)
# 混淆矩阵是理解分类模型性能的数据表, 他有助于理解如何把测试数据分成不同的类.
# 当想进行算法调优时, 就需要在对算法做出改变之前了解数据的错误分类情况. 有些分类效果比其他分类效果差, 混淆矩阵可以帮助理解这些问题.
# 理想情况下, 希望矩阵非对角先元素都是0. 即多有的元素都被模型分类到了其对应的类别
import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix   # 用于生成混淆矩阵

# 虚构数据
y_true = [1, 0, 0, 2, 1, 0, 3, 3, 3]
y_pred = [1, 1, 0, 2, 1, 0, 1, 3, 3]


def generate_confusion_matrix(y_true, y_pred):
    confusion_matrix_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)

    # 作图, 将会以分类种类的个数的表格, 这里是直接以颜色来进行映射数目
    plt.imshow(confusion_matrix_mat, interpolation="nearest", cmap=plt.cm.Paired)
    plt.title("Confision matrix")
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, labels=tick_marks)
    plt.yticks(tick_marks, labels=tick_marks)
    plt.xlabel("Predict Label")
    plt.ylabel("True Label")
    plt.show()

generate_confusion_matrix(y_true, y_pred)


# 提取性能报告
# 使用性能报告直接生成精度precision, 召回率recall, f1得分f1 score
from sklearn.metrics import classification_report

targets_names = ["分类0", "分类1", "分类2", "分类3"]

report = classification_report(y_true=y_true, y_pred=y_pred, target_names=targets_names)

print(report)
