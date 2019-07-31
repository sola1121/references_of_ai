from sklearn.datasets import samples_generator   # 样本数据生成器
from sklearn.ensemble import RandomForestClassifier    # 使用随机深林分类器增强
from sklearn.feature_selection import SelectKBest, f_regression   # 特征选择, 使用单变量线性回归测试
from sklearn.pipeline import Pipeline


# 生成样本数据, n_samples=100, n_features=20
X, y = samples_generator.make_classification(n_informative=4, n_features=20, n_redundant=0, n_classes=3, random_state=5)
print(X, "\n\n", y)


# 特征选择器, 选择k个最好的特征, 这里k为10
selector_k_best = SelectKBest(score_func=f_regression, k=10)

# 随机森林分类器
classifier = RandomForestClassifier(n_estimators=50, max_depth=4)

# 构建机器学习流水线, 以输入顺序执行, 并为每一步指定一个名称
pipeline_classifier = Pipeline([("selector", selector_k_best), ("rf", classifier)])

# 通过pipeline调整各运行参数
# pipeline_classifier.set_params(selector__k=6, rf__n_estimators=25)

# 训练分类器
pipeline_classifier.fit(X, y)

# 预测输出结果
prediction = pipeline_classifier.predict(X)
print("\nPredictions:\n", prediction)

# 打印分类器得分
print("\nScore:", pipeline_classifier.score(X, y))

# 打印哪些特征被分类器使用
feature_status = pipeline_classifier.named_steps["selector"].get_support()
selected_features = list()
for count, item in enumerate(feature_status):
    if item:
        selected_features.append(count)
print("\nSelected features (0-indexed):", ", ".join([str(x) for x in selected_features]))

