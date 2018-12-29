import data_loader
import numpy as np


if __name__ == "__main__":
    filename = "data/id_test"
    y, q, lq, c, lc = data_loader.read_data(filename, 20, 0)

    for i in range(1):
        batches = data_loader.batch_iter(list(zip(y, q, lq, c, lc)), 3, 1, False)

        batch = batches.next()

        y_batch, q_batch, lq_batch, c_batch, lc_batch = zip(*batch)

        lq_batch = np.reshape(lq_batch, [-1])


        print "y:\n", y_batch, "\n", "q:\n", q_batch, "\n", "lq:\n", lq_batch
        # print batch


