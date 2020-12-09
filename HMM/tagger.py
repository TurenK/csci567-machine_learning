import numpy as np
from hmm import HMM
from collections import Counter

def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)
    
    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    ###################################################
    # TODO: build two dictionaries
    #   - from a word to its index 
    #   - from a tag to its index 
    # The order you index the word/tag does not matter, 
    # as long as the indices are 0, 1, 2, ...
    ###################################################
    indx = 0
    for line in train_data:
        for word in line.words:
            if word not in word2idx:
                word2idx[word] = indx
                indx+=1

    pi = np.zeros(S)
    L = len(unique_words)
    A = np.full([S, S], 1/len(train_data))
    B = np.zeros((S, L))
    ###################################################
    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if  
    #   "divided by zero" is encountered, set the entry 
    #   to be zero.
    ###################################################
    tag_count = Counter([line.tags[0] for line in train_data])

    for indx, tag in enumerate(tags):
        tag2idx[tag] = indx
        pi[indx] = np.divide(tag_count[tag], len(unique_words))

    for line in train_data:
        for indx in range(line.length - 1):
            A[tag2idx[line.tags[indx]], tag2idx[line.tags[indx + 1]]] += 1

        for indx in range(line.length):
            B[tag2idx[line.tags[indx]], word2idx[line.words[indx]]] += 1

    for i in range(S):
        sum_a = np.sum(A[i])
        if sum_a < 0.00001:
            A[i] = 0
        sum_b = np.sum(B[i])
        if sum_b < 0.00001:
            B[i] = 0
        A[i] = np.divide(A[i], sum_a)
        B[i] = np.divide(B[i], sum_b)


    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ######################################################################
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    ######################################################################
    for line in test_data:
        count = 0
        for word in line.words:
            if word not in model.obs_dict:
                model.obs_dict[word] = len(model.obs_dict)
                count += 1
        if count:
            model.B = np.hstack([model.B, np.full([len(tags), count], 0.000001)])
        tagging.append(model.viterbi(line.words))

    return tagging

# DO NOT MODIFY BELOW
def get_unique_words(data):

    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
