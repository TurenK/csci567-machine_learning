import numpy as np
import pandas as pd


############################################################################
# DO NOT MODIFY CODES ABOVE 
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here                    #
    #####################################################
    err = np.mean(np.square(np.subtract(np.matmul(X, w), y)))
    return err


###### Part 1.2 ######
def linear_regression_noreg(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    #	TODO 2: Fill in your code here                    #
    #####################################################
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), y)
    return w


###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 4: Fill in your code here                    #
    #####################################################
    w = np.matmul(np.matmul(np.linalg.inv(np.add(np.matmul(np.transpose(X), X), (lambd * np.identity(np.size(X, 1))))),
                            np.transpose(X)), y)
    return w


###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    #####################################################
    # TODO 5: Fill in your code here                    #
    #####################################################		
    bestlambda = None
    least_error = float('Inf')
    tune_range = [2 ** (-14), 2 ** (-13), 2 ** (-12), 2 ** (-11), 2 ** (-10), 2 ** (-9), 2 ** (-8), 2 ** (-7), 2 ** (-6),
                  2 ** (-5), 2 ** (-4), 2 ** (-3), 2 ** (-2), 2 ** (-1), 1]
    for lambd in tune_range:
        w = regularized_linear_regression(Xtrain, ytrain, lambd)
        error = mean_square_error(w, Xval, yval)
        if error < least_error:
            least_error = error
            bestlambda = lambd
    return bestlambda


###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    #####################################################
    # TODO 6: Fill in your code here                    #
    #####################################################		
    temp = X
    for power_X in range(2, p + 1):
        temp = np.concatenate((temp, np.power(X, power_X)), axis=1)
    X = temp
    return X


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""
