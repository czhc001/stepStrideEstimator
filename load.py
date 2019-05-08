import numpy as np
import matplotlib.pyplot as plt
import csv

CLASS_NUM = 2
SEQ_MAX_LENGTH = 10
SEQ_MIN_LENGTH = 5
FRAME_WIDTH = 250
FRAME_HEIGHT = 3
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT


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
    # np.random.shuffle(ii)
    data_ = []
    label_ = []
    for i in range(label.shape[0]):
        data_.append(data[ii[i], :])
        label_.append(label[ii[i]])
    data_ = np.array(data_)
    label_ = np.array(label_)
    return [data_, label_]

