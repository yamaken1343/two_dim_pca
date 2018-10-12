import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

import pca_two_dim


def reconstruction_2d(train_img_dir, test_img_path, dim):
    train_faces_dir = train_img_dir
    test_face_path = test_img_path

    # PCAを行う画像複数枚をロード
    train_faces = []
    for path in os.listdir(train_faces_dir):
        img = Image.open(os.path.join(train_faces_dir, path))
        img = img.convert('L')
        img = np.asarray(img)

        train_faces.append(img)

    # PCAを行い共分散行列の固有ベクトルを算出
    U = pca_two_dim.pca2dim(train_faces, dim)

    # 再構成を行う画像をロード
    img = Image.open(test_face_path)
    img = img.convert('L')
    test_face = np.asarray(img)

    # 再構成
    A = test_face
    V = A @ U
    A = V @ U.T
    print(A.shape, V.shape)

    # 表示
    plt.imshow(test_face)
    plt.show()
    plt.clf()

    plt.imshow(A)
    plt.show()


def reconstruction_2d2d(train_img_dir, test_img_path, dim):
    train_faces_dir = train_img_dir
    test_face_path = test_img_path

    # PCAを行う画像複数枚をロード
    train_faces = []
    for path in os.listdir(train_faces_dir):
        img = Image.open(os.path.join(train_faces_dir, path))
        img = img.convert('L')
        img = np.asarray(img)

        train_faces.append(img)

    # PCAを行い共分散行列の固有ベクトルを算出
    X, Z = pca_two_dim.pca2dim2dim(train_faces, dim)

    # 再構成を行う画像をロード
    img = Image.open(test_face_path)
    img = img.convert('L')
    test_face = np.asarray(img)

    # 再構成
    A = test_face
    C = Z.T @ A @ X
    A = Z @ C @ X.T
    print(A.shape, C.shape)

    # 表示
    plt.imshow(test_face)
    plt.show()
    plt.clf()

    plt.imshow(A)
    plt.show()


if __name__ == '__main__':
    reconstruction_2d2d(sys.argv[1], sys.argv[2], 20)
    reconstruction_2d(sys.argv[1], sys.argv[2], 20)
