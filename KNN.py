__authors__ = '1571515, 1568205, 1571619'
__group__ = 'DM.18'


from enum import unique
import numpy as np
import math
import operator
from scipy.spatial.distance import cdist

class KNN:
    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)
        
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################


    def _init_train(self,train_data):
        """
        initializes the train datantrain_data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """

        if(train_data.dtype != float):
            try:
                train_data = train_data.astype(float)
            except:
                print("Posa algo que puguin ser floats")
        self.train_data = np.reshape(train_data,(train_data.shape[0],train_data.shape[1]*train_data.shape[2]*train_data.shape[3]))
                



    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates the k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        test_data = np.reshape(test_data,(test_data.shape[0], -1))

        dist = cdist(test_data, self.train_data)
      
      
        dist = np.argsort(dist, axis=1)
        dist = dist[::,0:k]
        self.neighbors = self.labels[dist]
        
        # sum = np.sum(dist, axis=0)
        # self.neighbors = np.full(k, 9999)
        # for s in range(sum.size):
        #     i = 0
        #     try:
        #         while(s > self.neighbors[i] and i < k):
        #             i+=1 #i++
        #         self.neighbors[i] = dist[s]
        #     except:
        #         pass
         



    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """


        a1 = []
        for i in self.neighbors:
            unique, b = np.unique(i, return_counts = True) # Aixo de sorted es mentida >:(
            a1.append(unique[np.argmax(b)])


        # labels = self.labels[self.neighbors]

        # return1 = np.array()
        # return2 = np.array()

        # for i in range(labels):
        #     label_frequency = np.array()
        #     for j in range(i):
        #         if labels[i][j] not in label_frequency:
        #             label_frequency[labels[i][j]] = 1
        #         else:
        #             label_frequency[labels[i][j]] += 1
        #     return1[i] = np.argmax[label_frequency]
        #     return2[i] = (label_frequency[return1[i]] / np.sum(label_frequency))*100
        
        # return return1 #, return2
        return a1

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """
        self.get_k_neighbours( test_data, k)

        return self.get_class()
