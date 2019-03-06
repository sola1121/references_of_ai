import numpy as np
from sklearn import preprocessing


def label_encoder01(X):
    """对大型行列式使用标记编码
    行列式每一行是一组数据集合, 每一列是同意类型数据, 标记编码需要将同类型数据进行标记编码, 即将每一列进行标记编码
    这里将每一列都给出一个标记编码器, 然后对每一列使用对应列的编码器进行训练.
    """
    label_encoders = list()   # 每一列的标记编码器集
    X_encoded = np.empty(X.shape)   # 构建一个和X相同形状的存储集
    for i in range(len(X[0])):   # 获取列数
        label_encoders.append(preprocessing.LabelEncoder())   # 向label_encoder中添加相应列数量的LabelEncoder对象
        X_encoded[: i] = label_encoders[-1].fit_transform(X[:, i])   # 使用对应列的编码器, 编码X中的数据, 并将转换好的放入相同位置x_encoded中

    # 拆分参数X和结果y, 并将其中的值都转换为int, 如果有必要的话
    X = X_encoded[:, :-1].astype(int)   # 所有行, 截取:-1列
    y = X_encoded[:, -1].astype(int)   # 所有行, 截取-1列

    return {"column": label_encoders, "X_set": X, "y_set": y}


def label_encoder02(X):
    """对于既有数字值又有字符值的数据集进行标记编码
    读取数据时要判断读取到的值是否是数值.
    """
    label_encoders = list()
    X_encoded = np.empty(X.shape)
    for i, item in enumerate(X[0]):
        if item.isdigit():
            X_encoded[:, i] = X[:, i]   # 判断字符为数值, 将原值赋予储存集, 如果需要转换类型, 也在这里
            label_encoders.append(None)   # 为了确保对应的labelEncoder对象和列数
        else:
            label_encoders.append(preprocessing.LabelEncoder())
            X_encoded[:, i] = label_encoders[i].fit_transform(X[:, i])   # 也可以使用label_encoder[-1]

    # 拆分参数X和结果y, 如果有必要
    X = X_encoded[:, :-1]
    y = X_encoded[:, -1]

    return {"column": label_encoders, "X_set": X, "y_set": y}


def predict_label(classifier, label_encoders, single_encoded_data):
    """反向解码"""
    output_class = classifier.predict(single_encoded_data)
    print(label_encoders[-1].inverse_transform(output_class))   # 将结果反向解码
