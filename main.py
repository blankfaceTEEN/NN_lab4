import os
import random
import re

import numpy as np
from PIL import Image


def matrix_to_vector(x):
    m = x.shape[0] * x.shape[1]
    buff = np.zeros(m)

    c = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            buff[c] = x[i, j]
            c += 1
    return buff


def create_weights(x):
    if len(x.shape) != 1:
        return
    else:
        w = np.zeros([len(x), len(x)])
        for i in range(len(x)):
            for j in range(i, len(x)):
                if i == j:
                    w[i, j] = 0
                else:
                    w[i, j] = x[i] * x[j]
                    w[j, i] = w[i, j]
    return w


def image_to_array(file, size, threshold=145):
    pilIN = Image.open(file).convert(mode="L")
    pilIN = pilIN.resize(size)
    imgArray = np.asarray(pilIN, dtype=np.uint8)
    x = np.zeros(imgArray.shape, dtype=np.float)
    x[imgArray > threshold] = 1
    x[x == 0] = -1
    return x


def array_to_image(data, outFile=None):
    y = np.zeros(data.shape, dtype=np.uint8)
    y[data == 1] = 255
    y[data == -1] = 0
    img = Image.fromarray(y, mode="L")
    if outFile is not None:
        img.save(outFile)
    return img


def update(w, y_vector, theta=0.5, time=100):
    for s in range(time):
        m = len(y_vector)
        i = random.randint(0, m - 1)
        u = np.dot(w[i][:], y_vector) - theta
        if u > 0:
            y_vector[i] = 1
        elif u < 0:
            y_vector[i] = -1

    return y_vector


def network(train_files, test_files, theta=0.5, time=1000, size=(100, 100), threshold=60, current_path=None):
    print("Загружаю изображения")

    files = 0
    for path in train_files:
        print(path)
        x = image_to_array(file=path, size=size, threshold=threshold)
        x_vectors = matrix_to_vector(x)
        if files == 0:
            weights = create_weights(x_vectors)
            files = 1
        else:
            buffer = create_weights(x_vectors)
            weights = weights + buffer
            files += 1

    print("Весы обновлены")

    counter = 0
    for path in test_files:
        y = image_to_array(file=path, size=size, threshold=threshold)
        oshape = y.shape
        y_image = array_to_image(y)
        y_image.show()
        print("Загружаю тестовое изображение")

        y_vector = matrix_to_vector(y)
        print("Нейросеть думает...")
        y_vector_ = update(w=weights, y_vector=y_vector, theta=theta, time=time)
        y_vector_ = y_vector_.reshape(oshape)
        if current_path is not None:
            outfile = current_path + "/" + str(counter) + ".jpeg"
            array_to_image(y_vector_, outFile=outfile)
        else:
            after_image = array_to_image(y_vector_, outFile=None)
            after_image.show()
        counter += 1


current_path = os.getcwd()
train_paths = []
path = current_path + "/train/"
for i in os.listdir(path):
    if re.match(r'[0-9a-zA-Z-]*.jp[e]*g', i):
        train_paths.append(path + i)

test_paths = []
path = current_path + "/test/"
for i in os.listdir(path):
    if re.match(r'[0-9a-zA-Z-_]*.jp[e]*g', i):
        test_paths.append(path + i)

network(train_files=train_paths, test_files=test_paths, theta=0.5, time=20000, size=(100, 100), threshold=90,
        current_path=current_path)
