import argparse

import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from scipy import misc
from sklearn import cluster

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Commpress the input image using clustering")
    parser.add_argument("--input-file", dest="input_file", required=True)
    parser.add_argument("--num-bits", dest="num_bits", required=False, type=int, help="Number of bits used to represent each pixel")
    return parser


def compress_image(img, num_clusters):
    """使用聚类将表示图片的数组拟合, 取其中的中心质点作为新的表示图片的数组的值, 
    要保证新区的质点值在其同一聚类的类中, 所以使用labels_来选出对应的质点.
    """
    # 将输入的图片转换为(样本量, 特征量)数组, 以运行k-means聚类算法
    X = img.reshape((-1, 1))

    k_means = cluster.KMeans(n_clusters=num_clusters, n_init=4, random_state=5)
    k_means.fit(X)
    centroids = np.squeeze(k_means.cluster_centers_)   # 去掉数组中的单维条目
    print("\n原先聚类质点的值\n", k_means.cluster_centers_, "\n\n去掉单维条目后\n", centroids, "\n\n")
    labels = k_means.labels_

    # 为每个数据配置离他最近的中心点, 并转变为图片的形状
    input_image_compressed = np.choose(labels, centroids)  # 使用labels_提供的序号对centroids中的数进行选择
    print("\nlabels指定的是点的下标\n", labels, "\n\n从centroids中选出的\n", input_image_compressed)

    return input_image_compressed.reshape(img.shape) 


def plot_image(img, title):
    vmin = img.min()
    vmax = img.max()
    plt.figure()
    plt.title(title)
    plt.imshow(img, vmin=vmin, vmax=vmax)


if __name__ == "__main__":

    args = build_arg_parser().parse_args()
    input_file = args.input_file
    num_bits = args.num_bits

    if not 1 <= num_bits <= 8:
        raise TypeError("Number of bits should be between 1 and 8")

    num_clusters = pow(2, num_bits)

    # 打印压缩率
    compression_rate = round(100 * (8.0 - args.num_bits) / 8.0, ndigits=2)
    print("\nThe size of the image will be reduced by a factor of", 8.0/args.num_bits)
    print("\nCompression rate =", compression_rate, "%")
    
    # 加载输入图片
    input_image = misc.imread(input_file, flatten=True).astype(np.uint8)
    # 显示原始图片
    # plot_image(input_image, "original image")
    plt.figure()
    original_img = matplotlib.image.imread(input_file) 
    plt.title("original image")
    plt.imshow(original_img)


    #压缩后的图片
    input_image_compressed = compress_image(input_image, num_clusters)
    plot_image(input_image_compressed, "compressed image; compression rate = " + str(compression_rate) + "%")
    plt.show()

