import numpy as np


def pca2dim(array, dim):
    # 平均値算出
    mean = np.mean(array, 0)
    # 共分散行列算出
    covariance = np.mean([np.matrix((a - mean)).T @ np.matrix((a - mean)) for a in array], 0)
    # print(covariance)
    # 共分散行列の固有値と固有ベクトルを算出
    eigen_value, eigen_vector = np.linalg.eigh(covariance)
    # print(eigen_value, eigen_vector)
    # 固有値の大きい順にソート
    idx = eigen_value.argsort()[::-1]
    eigen_value = eigen_value[idx]
    eigen_vector = eigen_vector[:, idx]  # 列ベクトルなことに注意する
    # 引数の次元数分戻す
    return eigen_vector[:, :dim].T


def pca2dim2dim(array, dim):
    t_array = np.transpose(array, (0, 2, 1))
    X = pca2dim(array, dim)
    Z = pca2dim(t_array, dim)
    return X, Z
