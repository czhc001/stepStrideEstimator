import numpy as np
import matplotlib.pyplot as plt
import csv

CLASS_NUM = 8
SEQ_MAX_LENGTH = 10
SEQ_MIN_LENGTH = 5
FRAME_SIZE = 1200
FRAME_WIDTH = 200
FRAME_HEIGHT = 6


def load_data_and_label(path):
    data = []
    label = []
    with open(path, 'r') as file:
        content = csv.reader(file, delimiter=',')
        for row in content:
            label.append(row[0])
            data.append([row[i] for i in range(1, FRAME_SIZE + 1)])
    data = np.array(data, dtype=np.float64)
    label = np.array(label, dtype=np.int)

    ii = [i for i in range(label.shape[0])]
    np.random.shuffle(ii)
    data_ = []
    label_ = []
    for i in range(label.shape[0]):
        data_.append(data[ii[i], :])
        label_.append(label[ii[i]])
    data_ = np.array(data_)
    label_ = np.array(label_)
    return [data_, label_]


def generate_data(datatype):
    values = np.array([0.0 for i in range(FRAME_SIZE)])
    if datatype == 0:
        radius = np.random.randint(0, 40, size=[2])
        se = np.random.randint(0, 40, size=[2])

        high = np.random.normal(0.5, 2)
        mid = int((se[0] + se[1])/2)
        for i in range(se[0], mid):
            value = float(i)/float(mid) * high
            values[i] = value
        for i in range(mid, se[1]):
            value = (1.0 - float(i) / float(mid)) * high
            values[i] = value
    elif datatype == 1:
        se = np.random.randint(0, 40, size=[2])
        high = np.random.normal(0.5, 2)
    elif datatype == 2:
        se = np.random.randint(0, 40, size=[2])
        high = np.random.normal(0.5, 2)


def triangle():
    noises = np.random.normal(0, 0.01, size=[FRAME_SIZE])
    values = np.zeros([FRAME_SIZE], np.float64)
    high = np.random.normal(0.0, 1)
    high = np.abs(high) + 0.5
    mid = np.random.randint(2, FRAME_SIZE-2)
    max_r = FRAME_SIZE - mid
    if max_r > mid:
        max_r = mid
    r = np.random.randint(1, max_r)
    st = mid - r
    en = mid + r
    for i0 in range(st, mid):
        value = float(i0 - st) / float(mid - st) * high
        values[i0] = value
    for i1 in range(mid, en):
        value = (1.0 - float(i1 - mid) / float(en - mid)) * high
        values[i1] = value
    values = values + noises
    return values


def oval():
    noises = np.random.normal(0, 0.01, size=[FRAME_SIZE])
    values = np.zeros([FRAME_SIZE], np.float64)
    high = np.random.normal(0.0, 0.5)
    high = np.abs(high) + 0.5
    mid = np.random.randint(2, FRAME_SIZE - 2)
    max_r = FRAME_SIZE - mid
    if max_r > mid:
        max_r = mid
    r = np.random.randint(1, max_r)
    st = mid - r
    en = mid + r
    for i in range(st, en):
        value = np.sqrt(float(r * r - (mid - i) * (mid - i)))
        values[i] = value
    for i in range(st, en):
        values[i] = values[i] * high / float(r)
    values = values + noises
    return values


def echelon():
    noises = np.random.normal(0, 0.01, size=[FRAME_SIZE])
    values = np.zeros([FRAME_SIZE], np.float64)
    high = np.random.normal(0.0, 1)
    high = np.abs(high) + 0.5
    width = np.random.randint(5, FRAME_SIZE - 1)
    st = np.random.randint(1, FRAME_SIZE - width)
    weights = np.random.normal(0, 0.5, size=[3])
    weights = np.abs(weights) + 1
    sum_w = np.sum(weights)
    weights = weights/sum_w
    b = st + int(width * weights[0])
    c = b + int(width * weights[1])
    d = st + width
    print(str(st) + ' ' + str(b) + ' ' + str(c) + ' ' + str(d))
    for i0 in range(st, b):
        value = high * float(i0 - st) / float(b - st)
        values[i0] = value
    for i1 in range(b, c):
        values[i1] = high
    for i2 in range(c, d):
        value = high * (1.0 - float(i2 - c) / float(d - c))
        values[i2] = value
    values = values + noises
    return values


'''
for i in range(100):
    plt.plot(np.array([i for i in range(FRAME_SIZE)]), oval(), color='r')
    plt.plot(np.array([i for i in range(FRAME_SIZE)]), echelon(), color='g')
    plt.plot(np.array([i for i in range(FRAME_SIZE)]), triangle(), color='b')

plt.show()
'''
