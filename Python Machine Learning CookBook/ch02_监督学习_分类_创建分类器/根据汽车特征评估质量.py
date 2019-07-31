import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


file_dir = "./dat/car.data.txt"
origin_data = list()

with open(file_dir, "r") as file:
    for line in file.readlines():
        temp_data = line[:-1].split(",")   # 去掉字符串末尾换行符\n再转换为数组
        origin_data.append(temp_data)

X = np.array(origin_data)   # 包含结果在内的所有数据

# 使用标记编码
label_encoder = list()   # 每一列的标记编码器
X_encoded = np.empty(X.shape)
# 开始对每一列的数据进行编码, 当然, 对应的每一列都使用的是同一个编码器
for i in range(len(X[0])):  # 获取X[0], 即第一行的X集数据, 这里主要用于取i
    label_encoder.append(LabelEncoder())   # 向label_encoder中添加相应数量的LabelEncoder对象
    X_encoded[:, i] = label_encoder[i].fit_transform(X[:, i])   # 对每一列的数据进行标签

X = X_encoded[:, :-1].astype(int)   # 所有行, 截取:-1列
y = X_encoded[:, -1].astype(int)   # 所有行, 截取-1列

# 建立随机森林分类器
params = {"n_estimators": 200, "max_depth": 8, "random_state": 7}
classifier = RandomForestClassifier(**params)
classifier.fit(X, y)


### NOTE: 交叉验证 ###
from sklearn.model_selection import cross_val_score

accuracy = cross_val_score(classifier, X, y, scoring="accuracy", cv=3)
print("Accuracy of the classifier:", round(100*accuracy.mean(), 2), "%.")


### NOTE: 使用训练好的分类器对单一孤立的未知数据进行分类 ##
input_data = ["vhigh", "vhigh", "2", "2", "small", "low"]
input_data_encoded = [-1] * len(input_data)   # 一个形状与输入数据格式相同, 但每个元素都是-1的list
for i, _ in enumerate(input_data):
    input_data_encoded[i] = label_encoder[i].transform([input_data[i]])

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)

# 预测并打印特定数据点的输出
output_class = classifier.predict(input_data_encoded)
print("Output Class:", output_class, label_encoder[-1].inverse_transform(output_class)[0])   # 使用最后一个标签器将结果反向解码


### NOTE: 生成验证曲线 ### 
from sklearn.model_selection import validation_curve

# 对n_estimators作为参数进行验证
parameter_grid1 = np.linspace(25, 200, 8).astype(int)   # 参数的取值范围
classifier = RandomForestClassifier(max_depth=4, random_state=7)
train_estimators_score, validation_estimators_score = validation_curve(classifier, X, y, param_name="n_estimators", param_range=parameter_grid1, cv=5)
print("\n\nValidation curves")
print("Param: n_estimators; Training score:\n", train_estimators_score)
print("Param: n_estimators; Validation score:\n", validation_estimators_score)

# 对max_depth进行验证
parameter_grid2 = np.linspace(2, 10, 5).astype(int)   # 参数的取值范围
classifier = RandomForestClassifier(n_estimators=20, random_state=7)
train_depth_score, validation_depth_score = validation_curve(classifier, X, y, param_name="max_depth", param_range=parameter_grid2, cv=5)
print("\n\nValidation curves")
print("Param: max_depth; Training score:\n", train_depth_score)
print("Param: max_depth; Validation score\n", validation_depth_score)

plt.figure(figsize=(5, 6))
plt.subplot(211)
plt.subplots_adjust(hspace=0.8)
plt.plot(parameter_grid1, 100*np.average(train_estimators_score, axis=1), color="black")
plt.title("Training curve")
plt.xlabel("Number of estimators")
plt.ylabel("Accuracy")

plt.subplot(212)
plt.plot(parameter_grid2, 100*np.average(train_depth_score, axis=1), color="black")
plt.title("Training curve")
plt.xlabel("Maximum depth of the tree")
plt.ylabel("Accuracy")

# plt.show()


# NOTE: 学习曲线
from sklearn.model_selection import learning_curve


classifier = RandomForestClassifier(random_state=7)

parameter_grid3 = np.array([200, 500, 800, 1100])
train_learning_sizes, train_learning_scores, validation_learning_scores = learning_curve(classifier, X, y, train_sizes=parameter_grid3)
print("\n\nLearning curve -", train_learning_sizes)
print("Training scores:\n", train_learning_scores)
print("Validation scores:\n", validation_learning_scores)

plt.figure()
plt.plot(parameter_grid3, 100*np.average(train_learning_scores, axis=1), color="black")
plt.title("Learning curve")
plt.xlabel("Number of training samples")
plt.ylabel("Accuracy")

plt.show()
