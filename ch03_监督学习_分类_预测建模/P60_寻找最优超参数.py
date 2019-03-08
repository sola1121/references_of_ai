import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn   # 不想看到sklearn的warning信息

import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC


def best_metric_classifier(X_train, X_test, y_train, y_test, best_params, metric_name):
    # 使用最好的参数生成训练一个向量机, 然后进行预测
    best_classifier = SVC(**best_params)
    best_classifier.fit(X_train, y_train)

    y_test_pred = best_classifier.predict(X_test)

    report = classification_report(y_true=y_test, y_pred=y_test_pred)
    print("\n当前最优指标", metric_name, "生成报告\n", report)


file_dir = "./dat/data_multivar.txt"
best_params = None   # 用来记录最好的参数集合
X, y = list(), list()

with open(file_dir, "r") as file:
    for line in file.readlines():
        temp_data = [float(i) for i in line.split(",")]
        X.append(temp_data[:-1])
        y.append(temp_data[-1])

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# 通过价差检验设置参数
parameter_grid = [
    {"kernel": ["linear"], "C": [1, 10, 50, 600]},
    {"kernel": ["poly"], "degree": [2, 3]},
    {"kernel": ["rbf"], "gamma": [0.01, 0.001], "C": [1, 10, 50, 600]},
]

# 定义需要使用的指标
metrics = ["precision", "recall_weighted"]


# 开始为每个指标搜索最优超参数
for metric in metrics:
    print("\n### Searching optimal hyperparameters for", metric)
    classifier = GridSearchCV(SVC(C=1), param_grid=parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)

    # 使用的参数们
    print("\n测试的参数", )
    for test_param in classifier.cv_results_["params"]:
        print(test_param)

    print("\n使用的的得分器", classifier.scorer_)

    # 打印出最好的参数集
    print("\nHighest scoring parameter set:", classifier.best_params_)
    best_metric_classifier(X_train, X_test, y_train, y_test, classifier.best_params_, metric)


