import numpy as np

CLASS_NUM = 5
SEQ_MAX_LENGTH = 10
SEQ_MIN_LENGTH = 5


def load_data_and_label():
    data = []
    label = []

    init = np.random.randint(0, CLASS_NUM)
    seq_l = [init]
    for i in range(1, SEQ_MIN_LENGTH):
        if i >= 3 and  seq_l[i-3] ==0 and  seq_l[i-1] == 0:
            current_l = 1



    data = np.array(data)
    label = np.array(label)
    return [data, label]

def generate_data(datatype):
    if datatype == 0:
        se = np.random.randint(0, 40, size=[2])
        high = np.random.normal(0.5, 2)
        mid = float(se[0]) / 2 + float(se[1]) / 2
        for i in range(se[0], se[1]):

    else if datatype == 2:
        se = np.random.randint(0, 40, size=[2])
        high = np.random.normal(0.5, 2)
    else:
        se = np.random.randint(0, 40, size=[2])
        high = np.random.normal(0.5, 2)