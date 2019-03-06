import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

file_dir = "./dat/adult.data.txt"    # 一共14个参数, 其中既有数字值, 也有非数字值, 结果是收入<50k和>50k
X, y = list(), list()
count_lessthan50k, count_morethan50k, num_images_threshold = int(), int(), 10000

with open(file_dir, "r") as file:
    for line in file.readlines():
        if "?" in line:
            continue                 # 跳过数据有缺失的
        data = line[:-1].split(", ") # 去掉str末尾的\n, 再将其转换为list

        # 确保<=50k和>50k的参数都各自有10000个, 保证训练模型没有偏向性.
        if data[-1] == "<=50K" and count_lessthan50k < num_images_threshold:
            X.append(data)
            count_lessthan50k += 1
        elif data[-1] == ">50K"  and count_morethan50k < num_images_threshold:
            X.append(data)
            count_morethan50k += 1

        if count_lessthan50k >= num_images_threshold and count_morethan50k >= num_images_threshold:
            break

X = np.array(X)

# 将字符串转换为数值数据
label_encoders = list()
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():              # 判断列的值是否是数值
        X_encoded[:, i] = X[:, i]
    else:
        label_encoders.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoders[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# 建立并训练朴素贝叶斯分类器
gaussiannb_classifier1 = GaussianNB()
gaussiannb_classifier1.fit(X, y)


# MARK: 交叉验证
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
gaussiannb_classifier2 = GaussianNB()
gaussiannb_classifier2.fit(X_train, y_train)
y_test_pred = gaussiannb_classifier2.predict(X_test)
# 计算分类器F1得分
f1 = cross_val_score(gaussiannb_classifier2, X, y, scoring="f1_weighted", cv=5)
print("F1 Score:", round(100*f1.mean(), 2), "%.")


# MARK: 实际值带入测试
input_data = [
    "39", "State_gov", "77516", "Bachelors", "13", "Nerver-married", "Adm-clerical",
    "Not-in-family", "White", "Male", "2174", "0", "40", "United-States"        
]
count = 0
input_data_encoded = [-1] * len(input_data)

for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoders[count].fit_transform([input_data[i]]))
        count += 1

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)

# 预测并打印预测结果
output_class = gaussiannb_classifier2.predict(input_data_encoded)
print(label_encoders[-1].inverse_transform(output_class))
