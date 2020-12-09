import numpy as np


#######################################################
# DO NOT MODIFY ANY CODE OTHER THAN THOSE TODO BLOCKS #
#######################################################

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data (either 0 or 1)
    - loss: loss type, either perceptron or logistic
	- w0: initial weight vector (a numpy array)
	- b0: initial bias term (a scalar)
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the final trained weight vector
    - b: scalar, the final trained bias term

    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
    multiplied by the step_size to update parameters.	
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    y[y == 0] = -1
    y = np.reshape(y, (N, 1))
    x_new = np.concatenate((X, np.ones(N).reshape(N, 1)), axis=1)
    w_1 = w.reshape((D, 1))
    w_new = np.concatenate((w_1, np.array(b).reshape((1, 1))), axis=0)

    if loss == "perceptron":
        ################################################
        # TODO 1 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize perceptron loss                  # 
        ################################################
        for iter in range(max_iterations):
            y_Xw = np.multiply(np.matmul(x_new, w_new), y)
            avg_loss_w_new = np.matmul(np.transpose(x_new), y * np.where(y_Xw <= 0, 1, 0)) / N
            w_new = w_new + step_size * avg_loss_w_new


    elif loss == "logistic":
        ################################################
        # TODO 2 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize logistic loss                    # 
        ################################################
        for iter in range(max_iterations):
            sigmoid_y_Xw = sigmoid(-np.multiply(np.matmul(x_new, w_new), y))
            avg_loss_w = np.matmul(np.transpose(x_new), np.multiply(y, sigmoid_y_Xw)) / N
            w_new = w_new + step_size * avg_loss_w

    else:
        raise Exception("Undefined loss function.")

    assert w_new.shape == (D + 1, 1)

    w = w_new[0:D].reshape((D,))
    b = w_new[D][0]

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after applying the sigmoid function 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : fill in the sigmoid function    #
    ############################################
    # z = np.array(z, dtype=np.float128)
    # handle overflow -- ignore
    np.seterr(over='ignore')
    value = np.divide(1, 1 + np.exp(-z))
    return value


def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    
    Returns:
    - preds: N-dimensional vector of binary predictions (either 0 or 1)
    """
    N, D = X.shape

    #############################################################
    # TODO 4 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    if loss == "perceptron":
        preds = np.where((np.matmul(X, w) + b) > 0, 1, 0)

    elif loss == "logistic":
        preds = np.where(sigmoid(np.matmul(X, w) + b) > 0.5, 1, 0)
    else:
        raise Exception("Loss Function is undefined.")

    assert preds.shape == (N,)
    return preds


def softmax(z):
    temp = np.exp(z - np.max(z))
    return np.divide(temp, np.sum(temp, axis=0))


def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data (0, 1, ..., C-1)
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform (stochastic) gradient descent

    Returns:
    - w: C-by-D weight matrix, where C is the number of classes and D 
    is the dimensionality of features.
    - b: a bias vector of length C, where C is the number of classes
	
    Implement multinomial logistic regression for multiclass 
    classification. Again for GD use the *average* of the gradients for all training 
    examples multiplied by the step_size to update parameters.
	
    You may find it useful to use a special (one-hot) representation of the labels, 
    where each label y_i is represented as a row of zeros with a single 1 in
    the column that corresponds to the class y_i. Also recall the tip on the 
    implementation of the softmax function to avoid numerical issues.
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)  # DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION

    x_new = np.concatenate((X, np.ones(N).reshape(N, 1)), axis=1)
    w_new = np.concatenate((w, b.reshape(C, 1)), axis=1)
    if gd_type == "sgd":

        for it in range(max_iterations):
            n = np.random.choice(N)
            ####################################################
            # TODO 5 : perform "max_iterations" steps of       #
            # stochastic gradient descent with step size       #
            # "step_size" to minimize logistic loss. We already#
            # pick the index of the random sample for you (n)  #
            ####################################################
            x = x_new[n].reshape(D + 1, 1)
            softmax_wx = softmax(np.matmul(w_new, x))
            softmax_wx[y[n]] -= 1  # subtract 1 for corresponding wx
            w_new = w_new - step_size * np.matmul(softmax_wx, np.transpose(x))

    elif gd_type == "gd":
        ####################################################
        # TODO 6 : perform "max_iterations" steps of       #
        # gradient descent with step size "step_size"      #
        # to minimize logistic loss.                       #
        ####################################################
        one_hot_y = np.transpose(np.eye(C)[y])  # one-hot y
        for iter in range(max_iterations):
            softmax_wx = softmax(np.matmul(w_new, np.transpose(x_new)))
            softmax_wx = np.subtract(softmax_wx, one_hot_y)
            w_new = w_new - (step_size * np.matmul(softmax_wx, x_new)) / N

    else:
        raise Exception("Undefined algorithm.")

    w = w_new[:,0:D].reshape((C,D))
    b = w_new[:,D].reshape(C,)

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained model, C-by-D 
    - b: bias terms of the trained model, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Predictions should be from {0, 1, ..., C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    #############################################################
    # TODO 7 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    C = b.shape[0]
    x_new = np.concatenate((np.ones(N).reshape(N, 1), X), axis=1)
    w_new = np.concatenate((b.reshape(C, 1), w), axis=1)
    preds = np.argmax(np.matmul(w_new, np.transpose(x_new)), axis=0)
    assert preds.shape == (N,)
    return preds
