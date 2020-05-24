import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    valid = np.loadtxt('logs/unet-effb0-gt-best-valid-cube6-15_5valid', delimiter=', ')
    train = np.loadtxt('logs/unet-effb0-gt-best-valid-cube6-15_5train', delimiter=', ')
    min_max_scaler = MinMaxScaler()

    # plt.yscale('log')
    plt.scatter(valid[:, 0], valid[:, 1])
    plt.scatter(train[:, 0], train[:, 1], c='r')
    plt.legend(('Validation error', 'Training error'))
    plt.show()
