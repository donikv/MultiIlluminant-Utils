import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    valid = np.loadtxt('logs/unet-efficientnet-b0-gt-best-valid-cube6-06_5-logvalid', delimiter=', ')
    train = np.loadtxt('logs/unet-efficientnet-b0-gt-best-valid-cube6-06_5-logtrain', delimiter=', ')
    min_max_scaler = MinMaxScaler()

    plt.yscale('log')
    plt.scatter(valid[:, 0], min_max_scaler.fit_transform(valid[:, 1].reshape(-1, 1)))
    plt.scatter(train[:, 0], min_max_scaler.fit_transform(train[:, 1].reshape(-1, 1)), c='r')
    plt.show()
