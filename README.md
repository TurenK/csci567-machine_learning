# csci567-machine_learning

Machine learning models -- CSCI 567 by Prof. Luo, Haipeng
**Language :** Python 3.6.4
**Tools :** only Numpy

## KNN

**Structure :** 
- f1_score**
- class Distance**
    - euclidean_distance
    - minkowski_distance
    - cosine_similarity_distance
**- class KNN**
    - train
    - get_k_neighbors
    - predict
**- class NormalizationScaler**
    - __call__ -- x′ = x / ||x||2
**- class MinMaxScaler**
    - __call__ -- x' = (x - min) / (max - min)
**- class HyperparameterTuner**
    - tuning_without_scaling
    - tuning_with_scaling
	
**Test :** supervised learning; test on heart disease to predict whether or not

## Linear Classifier

**Structure :** 
**Binary Classification**
    - Perceptron vs. Logistic
**Multiclass Classification**
    - SGD vs. GD

**Test :** supervised learning; test on synthetic data, two moon data and binarized MNIST data

## Regression

**Structure :** 
    - Regression without regularization
    - Regression with regularization
    - Tune the regularization parameter
    - Polynomial regression -- mapping

**Test :** supervised learning; test on wine data to predict wine quality

## Neural Network

**Structure :** 
    ![image](https://github.com/TurenK/csci567-machine_learning/blob/main/Neural_Networks/structure.png)
**Test :** supervised learning; test on a subset of MNIST

## K-Means

**Structure :** 
- K-means algorithm
    ![image](https://github.com/TurenK/csci567-machine_learning/blob/main/K_means/clusters.png)
- Classification with K-means
    ![image](https://github.com/TurenK/csci567-machine_learning/blob/main/K_means/Algo.png)
- K-means++ initialization -- most spread out initialization
- Image compression with K-means -- compression by k-means on a picture

**Test :** supervised learning; test on a toy dataset and some handwritten digit

## Hidden Markov Model

**Structure :** 
- HMM
    - calculate forward message, backward message
    - sequence probability -- observing a particular sequence
    - posterior probability -- the state at a particular time step given the observation sequence
    - likelihood probability -- state transition at a particular time step given the observation sequence
    - viterbi algorithm -- the most likely hidden state path
    
- Application to speech tagging
    - tagging and predicting

**Test :** supervised learning; test on sentences and predicting

