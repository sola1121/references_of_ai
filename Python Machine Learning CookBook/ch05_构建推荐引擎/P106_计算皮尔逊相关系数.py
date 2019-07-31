import json

import numpy as np


def pearson_score(dataset, user1, user2):
    """计算user1和user2的皮尔逊积矩相关系数"""
    if user1 not in dataset:
        raise TypeError("User " + user1 + " not present in dataset.")
    if user2 not in dataset:
        raise TypeError("User " + user2 + " not present in dataset.")

    # 提取两个用户均评过分的电影
    rated_by_both = dict()

    for item in dataset[user1]:
        if item in dataset[user2]:
            rated_by_both[item] = 1
    
    num_ratings = len(rated_by_both)
    # 如果两个用户都没有评分, 得分为0
    if num_ratings == 0:
        return 0

    # 两个用户都评过分的电影的评分之和
    user1_sum = np.sum([dataset[user1][item] for item in rated_by_both])
    user2_sum = np.sum([dataset[user2][item] for item in rated_by_both])

    # 两个用户都评过分的电影的评分的平方和
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in rated_by_both])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in rated_by_both])

    # 计算数据集的乘积之和
    product_sum = np.sum([dataset[user1][item] * dataset[user2][item] for item in rated_by_both])

    # 计算皮尔逊积矩相关系数
    Sxy =product_sum - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings

    # 考虑分母为0的情况
    if Sxx * Syy == 0:
        return 0

    return Sxy / np.sqrt(Sxx * Syy)


if __name__ == "__main__":

    file_dir = "./dat/movie_ratings.json"

    file = open(file_dir, 'r')
    data = json.loads(file.read())
    file.close()

    # 整两个用户来算算
    user1 = "John Carson"
    user2 = "Michelle Peterson"

    print("\nPearson score:", pearson_score(data, user1, user2))
    