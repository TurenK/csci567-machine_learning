import numpy as np
from knn import KNN


############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    temp_pre = np.array(predicted_labels)
    temp_rea = np.array(real_labels)

    TP = np.sum((temp_rea == 1) & (temp_pre == 1))
    FP = np.sum((temp_rea == 0) & (temp_pre == 1))
    FN = np.sum((temp_rea == 1) & (temp_pre == 0))
    f1 = TP / (TP + 1 / 2 * (FP + FN))
    return f1


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        if type(point1) in (int, float):
            point1 = [point1]
        if type(point2) in (int, float):
            point2 = [point2]
        summ = 0
        for i in range(len(point1)):
            summ += pow(abs(point1[i] - point2[i]), 3)
        final = pow(summ, (1 / 3))
        return final

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        if type(point1) in (int, float):
            point1 = [point1]
        if type(point2) in (int, float):
            point2 = [point2]
        summ = 0
        for i in range(len(point1)):
            summ += pow(abs(point1[i] - point2[i]), 2)
        final = pow(summ, (1 / 2))
        return final

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        if type(point1) in (int, float):
            point1 = [point1]
        if type(point2) in (int, float):
            point2 = [point2]
        summ_dotx = 0
        summ_x1_pow = 0
        summ_x2_pow = 0
        for i in range(len(point1)):
            summ_dotx += (point1[i] * point2[i])
            summ_x1_pow += pow(point1[i], 2)
            summ_x2_pow += pow(point2[i], 2)
        x1_sqrt = pow(summ_x1_pow, (1 / 2))
        x2_sqrt = pow(summ_x2_pow, (1 / 2))
        if x1_sqrt == 0.0 or x2_sqrt == 0.0:
            return 1.0
        else:
            final = 1.0 - summ_dotx / (x1_sqrt * x2_sqrt)
            return final


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29),
        and find the best model with the highest f1-score on the given validation set.

        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance.
        Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
        (this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        f1_scores = []
        ks = []
        diss = []
        i = 0
        for k_i in range(1, 30, 2):
            for d_i in distance_funcs:
                knn = KNN(k_i, distance_funcs[d_i])
                knn.train(x_train, y_train)
                predicted_labels = knn.predict(x_val)
                real_labels = y_val
                f1_scores.append([f1_score(real_labels, predicted_labels), i])
                ks.append(k_i)
                diss.append(d_i)
                i += 1
        indexes = sorted(f1_scores, key=lambda x: (x[0], -x[1]))

        # You need to assign the final values to these variables
        self.best_k = ks[indexes[len(indexes) - 1][1]]
        self.best_distance_function = diss[indexes[len(indexes) - 1][1]]
        distance_funcs = {
            'euclidean': Distances.euclidean_distance,
            'minkowski': Distances.minkowski_distance,
            'cosine_dist': Distances.cosine_similarity_distance,
        }
        self.best_model = KNN(self.best_k, distance_funcs[self.best_distance_function])
        self.best_model.train(x_train, y_train)

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically,
        before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them.

        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance.
        Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k,
        self.best_distance_function, best_scaler, and self.best_model respectively.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes).
        Then follow the same rule as in "tuning_without_scaling".
        """
        f1_scores = []
        ks = []
        diss = []
        scals = []
        i = 0
        for s_i in scaling_classes:
            scaling_class = scaling_classes[s_i]()
            x_train_new = scaling_class(x_train)
            x_val_new = scaling_class(x_val)
            for d_i in distance_funcs:
                for k_i in range(1, 30, 2):
                    knn = KNN(k_i, distance_funcs[d_i])
                    knn.train(x_train_new, y_train)
                    predicted_labels = knn.predict(x_val_new)
                    real_labels = y_val
                    f1_scores.append([f1_score(real_labels, predicted_labels), i])
                    ks.append(k_i)
                    diss.append(d_i)
                    scals.append(s_i)
                    i += 1
        indexes = sorted(f1_scores, key=lambda x: (x[0], -x[1]))

        # You need to assign the final values to these variables
        self.best_k = ks[indexes[len(indexes) - 1][1]]
        self.best_distance_function = diss[indexes[len(indexes) - 1][1]]
        distance_funcs = {
            'euclidean': Distances.euclidean_distance,
            'minkowski': Distances.minkowski_distance,
            'cosine_dist': Distances.cosine_similarity_distance,
        }
        self.best_model = KNN(self.best_k, distance_funcs[self.best_distance_function])
        self.best_model.train(x_train, y_train)
        self.best_scaler = scals[indexes[len(indexes) - 1][1]]


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        new_features = []
        for point in features:
            if type(point) == int:
                point = [point]
            point = np.array(point, float)
            if np.all(point == 0):
                new_features.append(point.tolist())
                continue
            point = np.divide(point, np.power(np.sum(np.power(point, 2)), 1 / 2))
            new_features.append(point.tolist())
        return new_features


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        new_features = np.array(features, float)
        for i in range(new_features.shape[1]):
            min_i = np.min(new_features[:, i])
            max_i = np.max(new_features[:, i])
            if max_i == min_i:
                new_features[:, i] = new_features[:, i] * 0.0
                continue
            new_features[:, i] = np.divide((new_features[:, i] - min_i), (max_i - min_i) * 1.0)
        return new_features.tolist()


def testF1():
    prediction_labels = [1, 1, 0, 0, 1, 1, 1, 1, 1]
    real_labels = [1, 0, 1, 1, 1, 0, 0, 1, 0]
    print(f1_score(real_labels, prediction_labels))


def testminko():
    point1 = [1, 2.5, 3, 4]
    point2 = [4, 1, 6, 7]
    print(Distances.minkowski_distance(point1, point2))


def testeu():
    point1 = [1, 2.5, 3, 4]
    point2 = [4, 1, 6.9, 7]
    # point1 = 1
    # point2 = 4
    print(Distances.euclidean_distance(point1, point2))


def testco():
    point1 = [1, 2, 3, 4]
    point2 = [4, 1, 6, 7]
    # point1 = 1
    # point2 = 4
    print(Distances.cosine_similarity_distance(point1, point2))


def testknn():
    features = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
    labels = [0, 1, 0, 1]

    knn = KNN(3, Distances.euclidean_distance)
    knn.train(features, labels)
    print(knn.predict([[0, 0]]))


def testNorm():
    features = [[3, 4], [1, -1], [0, 0]]
    norm = NormalizationScaler()
    print(norm(features))


def testMinMx():
    features = [[2, -1, 3], [-1, 5, 3], [0, 0, 3]]
    minmax = MinMaxScaler()
    print(minmax(features))


if __name__ == '__main__':
    # testF1()
    # testminko()
    # testeu()
    # testco()
    testknn()
    # testNorm()
    # testMinMx()
