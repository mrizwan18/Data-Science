import matplotlib.pyplot as plt
import numpy as np


class plot_helper:
    def __init__(self, iris_X_train, iris_y_train, c, g, svc, count):
        self.iris_X_train = iris_X_train
        self.iris_y_train = iris_y_train
        self.c = c
        self.g = g
        self.svc = svc
        self.count = count

    def plot_contours(self, ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    def make_meshgrid(self, x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1

        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy

    def plot(self):
        self.count += 1
        U, V = self.iris_X_train[:, 0], self.iris_X_train[:, 1]
        xx, yy = self.make_meshgrid(U, V)
        figsize = 5
        fig = plt.figure(figsize=(figsize, figsize))
        ax = plt.subplot(111)
        ax.text(3.9, 4.8, 'At C={} and gamma={}'.format(self.c, self.g), style='italic',
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        self.plot_contours(ax, self.svc, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(U, V, c=self.iris_y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        plt.savefig('plots/f' + str(self.count) + '.png')
        plt.close(fig)
