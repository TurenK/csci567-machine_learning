import numpy as np
from collections import Counter
import time
from copy import deepcopy


#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def get_eucli_distance(point1, point2):
    return np.sum(np.power((point1 - point2), 2))

def get_eucli_distance_matrix(points1, points2):
    return np.sum(np.power((points1 - points2), 2), axis=1)


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    ###############################################
    # TODO: implement the Kmeans++ initialization
    ###############################################
    centers = [generator.randint(0, n)]
    for index in range(1, n_cluster):
        probabilities = np.zeros(n)
        for x_i in range(n):
            probabilities[x_i] = min(get_eucli_distance(x[x_i], x[center]) for center in centers)
        probabilities = probabilities / np.sum(probabilities)
        probabilities_cul = np.array([np.sum(probabilities[0:i]) for i in range(1, n + 1)])
        r = generator.rand()
        centers.append(np.argmin(np.where(probabilities_cul <= r, 1, 0)))

    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)


class KMeans:
    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array, 
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0), 
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        ###################################################################
        # TODO: Update means and membership until convergence 
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################
        centers_coor = x[self.centers]
        before_j = 0
        iteration = 0
        membership = np.zeros(N)
        # put each point in its nearest center -- initial membership
        for x_i in range(N):
            matrix_x_i = np.concatenate(([x[x_i] for k in range(self.n_cluster)])).reshape((self.n_cluster, D))
            membership[x_i] = np.argmin(get_eucli_distance_matrix(matrix_x_i, centers_coor))

        for i in range(self.max_iter):
            # update centers to average of each group
            print("start iteration "+str(i))
            a = time.time()
            for c_ind in range(self.n_cluster):
                group = np.flatnonzero(membership == c_ind)
                if len(group):
                    centers_coor[c_ind] = np.divide(np.sum(x[group], axis=0), len(group))
            print("update center time: " + str(time.time()-a))

            # put each point in its nearest center
            b = time.time()
            for x_i in range(N):
                matrix_x_i = np.concatenate(([x[x_i] for k in range(self.n_cluster)])).reshape((self.n_cluster,D))
                membership[x_i] = np.argmin(get_eucli_distance_matrix(matrix_x_i,centers_coor))
            print("assign membership time: "+str(time.time()-b))

            # calculate average distortion objective value
            c = time.time()
            now_j = np.sum(
                [np.abs(get_eucli_distance(x[x_i], centers_coor[int(membership[x_i])])) for x_i in range(N)]) / N
            print("calculate j time: "+str(time.time()-c))

            # if |now_j-before_j| is less than a threshold, break
            if np.abs(now_j - before_j) < self.e:
                break
            # else update j and continue
            before_j = now_j
            iteration += 1

        return centers_coor, membership, iteration


class KMeansClassifier:
    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented, 
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################
        kmeans = KMeans(self.n_cluster, e=self.e, max_iter=self.max_iter, generator=self.generator)
        centroids, member, iteration = kmeans.fit(x, centroid_func=centroid_func)

        centroid_labels = np.array([Counter(y[np.flatnonzero(member == c_ind)]).most_common()[0][0] for c_ind in range(self.n_cluster)])

        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored 
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################
        labels = np.zeros((N,))
        for x_i in range(N):
            labels[x_i] = self.centroid_labels[
                np.argmin([get_eucli_distance(x[x_i], self.centroids[c_ind]) for c_ind in range(self.n_cluster)])]

        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)


def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################
    new_im = deepcopy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_im[i][j] = code_vectors[np.argmin(get_eucli_distance(image[i][j], code_vectors[k]) for k in range(code_vectors.shape[0]))]

    return new_im
