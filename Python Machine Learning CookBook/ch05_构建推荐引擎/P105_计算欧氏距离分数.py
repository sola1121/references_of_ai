import json

import numpy as np

"""
处理Json的格式
{
    用户名1:{电影名1: 评分, 电影名2: 评分, ...},
    用户名2:{电影名3: 评分, 电影名4: 评分, ...},
    ...
}
"""

def euclidean_score(dataset, user1, user2):
    """计算user1和user2用户之间的欧几里得距离"""
    if user1 not in dataset:
        raise TypeError("User " + user1 + " not present in dataset.")
    if user2 not in dataset:
        raise TypeError("User " + user2 + " not present in dataset.")

    # 提取两个用户均评过分的电影
    rated_by_both = dict()

    for item in dataset[user1]:        # 获取用户1的电影字典的keys, 即电影名
        if item in dataset[user2]:     # 如果该电影名在用户2的电影字典的keys中
            rated_by_both[item] = 1    # 将电影名作为键, 得分为1
    
    # 如果两个用户没有相同的电影, 不能说这两个用户行为间有关联, 这里设置得分为0
    if len(rated_by_both) == 0:
        return 0

    # 对于每个共同评分, 只计算平方和的平方根, 并将值归一化
    squared_differences = list()

    for item in dataset[user1]:      # 在用户1中的电影字典keys, 即电影名
        if item in dataset[user2]:   # 在用户2中有相同的电影
            squared_differences.append(np.square(dataset[user1][item] - dataset[user2][item]))   # 用户的评分的差的平方
    
    return 1 / (1 + np.sqrt(np.sum(squared_differences)))   # 归一化


if __name__ == "__main__":

    file_dir = "./dat/movie_ratings.json"

    file = open(file_dir, 'r')
    data = json.loads(file.read())
    file.close()

    # 假定两个随机用户, 计算其欧氏距离
    user1 = "John Carson"
    user2 = "Michelle Peterson"

    print("\nEuclidean score:", euclidean_score(data, user1, user2))
