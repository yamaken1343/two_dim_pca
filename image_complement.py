import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

import pca_two_dim


def restore(src, mask):
    src[mask.nonzero()] = mask[mask.nonzero()]

    return src


def complement_2d2d(train_img_dir, test_img_path, dim):
    train_faces_dir = train_img_dir
    test_face_path = test_img_path
    train_faces = []
    for path in os.listdir(train_faces_dir):
        img = Image.open(os.path.join(train_faces_dir, path))
        img = img.convert('L')
        img = np.asarray(img)

        train_faces.append(img)

    X, Z = pca_two_dim.pca2dim2dim(train_faces, dim)
    img = Image.open(test_face_path)
    img = img.convert('L')
    test_face = np.asarray(img)
    A = test_face
    for i in range(10):
        C = Z.T @ A @ X
        A = Z @ C @ X.T

        A = restore(A, test_face)
    print(A.shape, C.shape)

    plt.imshow(test_face)
    plt.show()
    plt.clf()

    plt.imshow(A)
    plt.show()


if __name__ == '__main__':
    complement_2d2d(sys.argv[1], sys.argv[2], 20)
