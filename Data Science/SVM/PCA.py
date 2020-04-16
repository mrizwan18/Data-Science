import numpy as np


class PCA:
    def __init__(self, data, k=0, threshold=0.95):
        """
        :param data: Data To be reduced to k dimensions
        :param k: if k not given then by default k will be optimally selected based on the threshold of variance
        :param threshold: variance threshold to accept the optimal k
        """
        self.X = data
        self.k = k
        self.threshold = threshold

    def normalize_features(self):
        feature = np.asarray(self.X)
        mean = (sum(feature) / len(feature))
        feature = (feature - mean)
        return feature

    def PCA(self):
        original = self.X
        self.X = self.normalize_features()
        x_trans_x = np.dot(np.transpose(self.X), self.X)
        sigma = x_trans_x / len(self.X)
        u, s, v = np.linalg.svd(sigma)
        if self.k == 0:  # Means we need to find optimal k. If k is not 0 then it means that user is forcing to reduce to k
            # dimesnions
            self.k = self.select_k(s, self.X.shape[1])
        u_reduce = u[:, :self.k]
        z = np.dot(np.transpose(u_reduce), np.transpose(original))
        return z, u_reduce

    def select_k(self, s, features):
        for k in range(1, features):
            if self.k_variance(s, k) >= self.threshold:
                return k

    def k_variance(self, s, k):
        variance = sum(s[:k]) / sum(s)
        return variance
