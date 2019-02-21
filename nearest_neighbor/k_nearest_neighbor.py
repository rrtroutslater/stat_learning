# Russell Reinart, Stat 760, Spring 2019

import numpy as np 
from scipy.stats import mode

class k_nearest_neighbor():
    def __init__ (self, fname_train, fname_test):

        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data(
                fname_train, fname_test)

    def classify(self, k):
        """ 
        returns predicted labels in test set by choosing k nearest (euclidean distance) neighbors 
        """
        distances = np.zeros(shape=(self.x_train.shape[0]))
        y_predict = np.zeros(shape=(self.x_test.shape[0]))
        
        idx = 0
        for test in self.x_test:
            distances = np.sqrt(np.sum((test- self.x_train)**2, axis=1))  # euclidean distances
            k_nearest_idx = distances.argsort()[:k]                 # indices of nearest distances
            k_nearest_label = self.y_train[k_nearest_idx]           # labels of nearest neighbors
            y_predict[idx] = mode(k_nearest_label)[0][0]            # most common label
            idx += 1
        
        return y_predict

    def compute_loss(self, y_predict):
        """ 
        assumes y_predict[i] corresponds to same data as y_test[i] 
        returns # of misclassified points, and percentage of misclassified points
        """
        loss = np.sum(y_predict != self.y_test)
        loss_percent = loss / self.x_test.shape[0]
        
        return loss, loss_percent

    def compute_loss_statistics(self, y_predict):
        """
        returns dictionary of 1x10 arrays. keys are digits, and arrays represent 
        histograms of the number of misclassified digits for the key.  array[i] is the
        number of times digit i is misclassified as the key.
        """

        stats = {}
        for i in range(10):
            stats[i] = np.zeros(10)

        idx = 0
        for pred in y_predict:
            if pred != self.y_test[idx]:
                stats[int(self.y_test[idx])][int(pred)] += 1
            idx += 1

        return stats

    def load_data(self, fname_train, fname_test):
        """ 
        load training and test data from text files 
        """
        train = np.genfromtxt(fname_train, delimiter=' ')
        test = np.genfromtxt(fname_test, delimiter=' ')
        x_train = train[:, 1:]
        y_train = train[:,0]
        x_test = test[:, 1:]
        y_test = test[:,0]
        
        return x_train, y_train, x_test, y_test
