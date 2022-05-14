__authors__ = ['1568205', '1571619', '1571515']
__group__ = 'DM.18'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictºionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options
        self._init_centroids()

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the smaple space is the length of
                    the last dimension
        """
        if X.dtype != float:
            try:
                X = X.astype(float)
            except:
                print("Posa algo que siguin floats")

        if len(X.shape) == 3:
            self.X = np.reshape(X, (-1, X.shape[2]))
        else:
            self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other prameter you can add it to the options dictionary
        self.options = options

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        index = 1
        counter = 1
        
        if self.options['km_init'].lower() == 'first':
            self.centroids = np.zeros([1, self.X.shape[1]])
            self.centroids[0] = self.X[0]
            while counter != self.K:
                if (self.centroids != np.array([self.X[index]])).any(axis=1).all():
                    self.centroids = np.concatenate((self.centroids, np.array([self.X[index]])))
                    counter += 1
                index += 1

        if self.options['km_init'].lower() == 'random':
            index = np.random.randint(self.X.shape[0], size=self.K)
            self.centroids = self.X[index]

    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        distances = distance(self.X, self.centroids)
        self.labels = np.argmin(distances, axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = np.copy(self.centroids)

        for c in range(self.K):
            idexes = np.where(self.labels == c)
            clss = self.X[idexes]
            if clss.size == 0:
                self.centroids[c] = 0
            else:
                self.centroids[c] = clss.mean(axis=0)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        diff = distance(self.centroids, self.old_centroids)

        for i in range(diff.shape[0]):
            if self.options['tolerance'] < diff[i][i]:
                return False
        return True

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        while self.num_iter < self.options['max_iter']:
            """1.Per a cada punt de la imatge, troba quin és el centroide més proper."""
            self.get_labels()
            """2. Calcula nous centroides utilitzant la funció get_centroids"""
            self.get_centroids()
            """3. Augmenta en 1 el número d’iteracions"""
            self.num_iter += 1
            """4. Comprova si convergeix, en cas de no fer-ho torna al primer pas."""
            if self.converges():
                break

    def whitinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """
        wcd = 0
        n = len(self.X)
        for i, point in enumerate(self.X):
            cx = self.labels[i]
            diff = point - self.centroids[cx]
            wcd += np.matmul(diff, diff.transpose())
        wcd = 1/n * wcd

        return wcd

    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        wcd = 0

        for k in range(2, max_K):
            self.K = k
            self._init_centroids()
            self.fit()
            wcd_old = wcd
            wcd = self.whitinClassDistance()
            if wcd_old != 0:
                decrement = 100 - 100 * (wcd / wcd_old)
                if 20 > decrement:
                    self.K = k - 1
                    break



def distance(X, C):
    """
        Calculates the distance between each pixcel and each centroid
        Args:
            X (numpy array): PxD 1st set of data points (usually data points)
            C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

        Returns:
            dist: PxK numpy array position ij is the distance between the
            i-th point of the first set an the j-th point of the second set
        """

    distance = np.zeros([len(X), len(C)])

    for i in range(C.shape[0]):
        distance[:, i] = np.sum((X - C[i]) ** 2, axis=1) ** 0.5

    return distance


def get_colors(centroids):
    """
        for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
        Args:
            centroids (numpy array): KxD 1st set of data points (usually centroind points)

        Returns:
            lables: list of K labels corresponding to one of the 11 basic colors
        """
    labels = []
    probabilities = utils.get_color_prob(centroids)

    for i in probabilities:
        max_prob = np.argmax(i, axis=None)
        labels.append(utils.colors[max_prob])

    return labels