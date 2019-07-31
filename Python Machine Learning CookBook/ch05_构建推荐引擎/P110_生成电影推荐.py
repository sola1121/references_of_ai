import json

import numpy as np

from P105_计算欧氏距离分数 import euclidean_score
from P106_计算皮尔逊相关系数 import pearson_score
from P108_寻找数据集中的相似用户 import find_similar_users


# 为给定用户生成电影推荐
def generate_recommendations(dataset, user):
    if user not in dataset:
        raise TypeError("User " + user + " not present in dataset")

    # 计算该用户与数据库中其他用户的皮尔逊相关系数
    total_scores = {}
    similarity_sums = {}

    for other_user in [u for u in dataset if u != user]:
        similarity_score = pearson_score(dataset, user, other_user)

        if similarity_score <= 0:
            continue

        # 找到未被该用户评分的电影
        un_comment = [m for m in dataset[other_user] if m not in dataset[user] or dataset[user][m] == 0]
        print(un_comment)

        for item in un_comment:
            total_scores.update({item: dataset[other_user][item] * similarity_score})
            similarity_sums.update({item: similarity_score})

    print("total_scores", total_scores)
    print("\nsimilarity_sums", similarity_sums)

    # 如果该用户看过数据库中所有的电影, 那就不能为用户推荐电影
    if len(total_scores) == 0:
        return ['No recommendations possible']

    # 生成一个电影评分标准化列表
    movie_ranks = np.array([[total/similarity_sums[item], item] 
                            for item, total in total_scores.items()])

    # 根据第一列对皮尔逊相关系数进行降序排列
    movie_ranks = movie_ranks[np.argsort(movie_ranks[:, 0])[::-1]]

    # 提取出推荐电影
    recommendations = [movie for _, movie in movie_ranks]

    return recommendations


if __name__ == "__main__":

    file_dir = "./dat/movie_ratings.json"

    file = open(file_dir, 'r')
    data = json.loads(file.read())
    file.close()

    user = "William Reynolds"   # Michael Henry"
    # print("\nRecommandations for " + user + ":")
    movies = generate_recommendations(data, user)
    # for i, movie in enumerate(movies):
    #     print(i+1, " -- ", movie)

    user = "John Carson"
    # print("\nRecommandations for " + user + ":")
    movies = generate_recommendations(data, user)
    # for i, movie in enumerate(movies):
    #     print(i+1, " -- ", movie)

