import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances as sk_euclidean_distances
import math

from scipy.stats import entropy as kl_div


def accuracy(y_true, y_pred, num_classes):
    '''
    Class value should start from 1.
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # results = np.zeros(num_classes, dtype=float)
    results = []
    for i in range(num_classes):
        cls = i + 1
        index = np.where(y_true == cls)
        true = y_true[index]
        if true.size == 0:
            # We do not consider this class because it is not in the ground truth
            print("Class {} not in the ground truth".format(cls))
            # TODO: May set result value to 'nan'
        else:
            # true = y_true[index]
            pred = y_pred[index]
            matched = (true == pred)
            tp = float(matched.sum())
            total = len(true)
            # results[i] = tp / total
            result = tp / total
            results.append(result)
    return np.array(results)


def _euclidean_distance(vec1, vec2):
    diff = np.subtract(vec1, vec2)
    return np.sqrt(np.sum(diff ** 2))


def euclidean_similarity(vec1, vec2):
    # d = euclidean_distance(vec1, vec2)
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    d = sk_euclidean_distances(vec1, vec2)
    # print("Distance", d[0][0])
    max_d = math.sqrt(2)
    assert d[0][0] <= max_d, print("Euclidean distance is ", d)
    # Rescale distance to 0 - 1
    d = d[0][0]
    alpha = (max_d - d) / max_d
    # alpha = 1.0 / (1.0 + d)
    assert 0.0 <= alpha <= 1.0, print("alpha is ", alpha)
    return alpha


def kl_similarity(vec1, vec2):
    alpha = kl_div(vec1, vec2)
    # return (1 - alpha)
    return alpha


def bhatta_similarity(prev_vec, cur_vec):
    alpha = np.sum(np.sqrt(prev_vec*cur_vec))
    alpha = round(alpha, 5)
    assert 0.0 <= alpha <= 1.0, print("alpha is ", alpha)
    return alpha


def _cosine_similarity(vec1, vec2):
    dot_prod = vec1 * vec2
    numerator = np.sum(dot_prod)
    vec1_square_root = np.sqrt(np.sum(vec1 ** 2))
    vec2_square_root = np.sqrt(np.sum(vec2 ** 2))
    denominator = vec1_square_root * vec2_square_root

    return (numerator / denominator)


def cosine_similarity(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    alpha = sk_cosine_similarity(vec1, vec2)
    alpha = round(alpha[0][0], 5)
    # print("Custom sim: ", _cosine_similarity(vec1, vec2), "Sk sim:", alpha)
    # Custom cosine function and sk learn are almost similar
    assert alpha >= 0.0 and alpha <= 1.0, print("alpha is ", alpha)
    return alpha


def softmax(scores):
    max = np.max(scores)
    stable_x = np.exp(scores - max)
    prob = stable_x / np.sum(stable_x)
    return prob


def sigmoid(x):
    x = np.array(x)
    return 1.0 / (1.0 + np.exp(-1.0 * x))


def get_error(vec1, vec2):
    return np.abs(vec1 - vec2)

# todo change the get_action to get_class


def get_action(window_scores, use_softmax=False, avg=True):
    if use_softmax:
        for i in range(len(window_scores)):
            window_scores[i] = softmax(window_scores[i])

    if avg:
        avg_score = np.sum(
            window_scores, axis=0) / len(window_scores)
        axis = None
    else:
        avg_score = window_scores
        axis = 1

    return np.argmax(avg_score, axis=axis)
