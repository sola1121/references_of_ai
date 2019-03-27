import json

import numpy as np

from P106_计算皮尔逊相关系数 import pearson_score


# 寻找特定数量的与输入用户相似的用户
def find_similar_users(dataset, user, num_users):
    """dataset: 数据库, user: 查找的用户, num_users: 想要找到的相似用户个数
    第一步将会查看该用户是否包含在数据库中.
    第二步如果用户存在, 则需要计算该用户与数据库中其他所有用户的皮尔逊积矩相关系数
    """
    if user not in dataset:
        raise TypeError("User" + user + " not present in the dataset.")

    # 计算所有用户的皮尔逊相关度
    scores = np.array([[name, pearson_score(dataset, user, name)] for name in dataset if user != name])

    # 评分按照第二列排列
    scores_sorted = np.argsort(scores[:, 1])

    # 评分按照降序排列
    scores_sorted_dec = scores_sorted[::-1]

    # 提取出k个最高分
    top_k = scores_sorted_dec[0:num_users]

    return scores[top_k]


if __name__ == "__main__":

    file_dir = "./dat/movie_ratings.json"

    file = open(file_dir, 'r')
    data = json.loads(file.read())
    file.close()

    user = "John Carson"
    print("\nUsers similar to %s:\n" % user)
    
    similar_users = find_similar_users(data, user, 3)

    print("User\t\t\tSimilar score\n")
    for item in similar_users:
        print(item[0], "\t\t", round(float(item[1]), 2))

