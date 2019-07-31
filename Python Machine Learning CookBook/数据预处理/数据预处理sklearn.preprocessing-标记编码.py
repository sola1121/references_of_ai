from sklearn import preprocessing

# 用于学习的标记
input_classes = ["audi", "ford", "audi", "toyota", "volvo", "ford", "bmw"]

# 标记编码方法
# 在监督学习中, 经常需要处理各种各样的标记. 这些标记可能是数字, 也可能是单词.
# 如果标记是数字, 那么算法可以直接使用他们, 但是, 许多情况下, 标记都需要人们可理解的形式存在, 因此, 人们通常会用单词标记训练数据集. 
# 标记编码就是要把单词标记转换为数值形式, 让算法懂得如何操作标记.
label_encoder = preprocessing.LabelEncoder()   # 定义一个标记编码器
label_encoder.fit(input_classes)   # 对标记进行学习

print("Class mapping")
for i, item in enumerate(label_encoder.classes_):
    print(item, "-->", i)


# 用于测试的标记
labels = ["volvo", "bmw", "toyota"]
encoded_labels = label_encoder.transform(labels)
print("\nLabels =", labels)
print("Encded labels =", list(encoded_labels))


# 通过数字反向获取标签
encoded_labels = [3, 2, 1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print("\nEncoded labels =", encoded_labels)
print("Decoded labels =", decoded_labels)
