#coding=utf-8
from __future__ import division

import numpy as np

def batch_iter(data, batch_size, num_epochs, shuffle=False):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            #start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            start_index = end_index-batch_size
            yield shuffled_data[start_index:end_index]



def read_data(filename, seq_len, pad_id):
    pad_id = int(pad_id)

    ys = []
    qs = []
    lq = []
    cs = []
    lc = []

    f = open(filename, "r")

    skip = 0
    for line in f.readlines():
        line = line.strip()

        try:
            label, q, c = line.split("\t")
        except:
            skip += 1
            continue

        q = q.strip()
        q = q.split()
        q = q[:seq_len]
        l1 = [len(q)]
        lq.append(l1)
        q = q + [pad_id] * (seq_len - len(q))
        qs.append(q)

        c = c.strip()
        c = c.split()
        c = c[:seq_len]
        l2 = [len(c)]
        lc.append(l2)
        c = c + [pad_id] * (seq_len - len(c))
        cs.append(c)

        label = label.strip()
        if label == "1":
            y = [0, 1]
        elif label == "0":
            y = [1, 0]
        else:
            raise ValueError
        ys.append(y)


    print "read data done..."
    print "skip", skip, "lines"
    return np.array(ys), np.array(qs), np.array(lq), np.array(cs), np.array(lc)



