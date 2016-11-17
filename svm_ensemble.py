'''
A really basic approach using the default data parameters
(cutting, 4mm smoothing, 0.25 zoom factor, divide dimensions in 3)

Train an SVM on each of the resulting subcubes, then average their results.
'''

import data
import numpy as np
from sklearn.svm import SVC

'''
Loads the given range/type of data.

The result has shape (n_cubes ^ 3, sample_size, cube_size), where
    n_cubes: number of cubes per dimension
    n_samples: number of samples in rng
    cube_size: final cube size, which depends on the cutting and zooming out.

'''
def load_all(typ, rng):
    X = []
    for i in rng:
        x = data.load_and_preprocess(typ, i)
        sh = x.shape

        # Flatten each inner cube
        X.append(x.reshape(sh[0], sh[1] * sh[2] * sh[3]))

    return np.rollaxis(np.stack(X), 1)

'''
Input: an array of the same shape as returned by load_all, and the labels.
Output: a list of n_cubes^3 classifiers
'''
def fit_bunch(X, y):
    predictors = []
    for i in range(len(X)):
        #print('Fitting classifier ' + str(i))

        pred = SVC(kernel='linear', probability=True)
        pred.fit(X[i], y)

        predictors.append(pred)

    return predictors

'''
Input: an array of the same shape as returned by load_all, and the classifiers
Output: an array of shape (n_predictors, n_samples, 2)
'''
def predict_bunch(Z, predictors):
    predictions = []

    for i in range(len(Z)):
        #print('Predicting with classifier ' + str(i))

        pred = predictors[i].predict_proba(Z[i])
        predictions.append(pred)

    return np.array(predictions)

'''
Combine the predictions from each classifier using a mean of all of them.
Input: an array of shape (n_predictors, n_samples, 2)
Output: an array of shape (n_samples, 2)
'''
def combine_predictions_mean(preds):
    avg = np.mean(preds, axis=0)
    return avg[:, 1] # Only take the first element of the second dimension

# TODO have a different combination of predictions?


# Testing

def validate():
    X = load_all('train', range(250))
    y = data.get_targets(range(250))
    clas = fit_bunch(X, y)

    Z = load_all('train', range(250, 278))
    actual = data.get_targets(range(250, 278))
    predicted_indiv = predict_bunch(Z, clas)
    predicted_ensem = combine_predictions_mean(predicted_indiv)

    print(data.evaluate(predicted_ensem, actual))

def test():
    X = load_all('train', range(278))
    y = data.get_targets(range(278))
    clas = fit_bunch(X, y)

    Z = load_all('test', range(138))
    predicted_indiv = predict_bunch(Z, clas)
    predicted_ensem = combine_predictions_mean(predicted_indiv)
    data.print_prediction(predicted_ensem)

if __name__ == '__main__':
    #validate()
    test()
