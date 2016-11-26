'''
A really basic approach using the default data parameters
(cutting, 4mm smoothing, 0.25 zoom factor, divide dimensions in 3)

Train an SVM on each of the resulting subcubes, then average their results.
'''

import data
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold

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
Output: a list of n_cubes^3 tuples of (classifier, transformer)
'''
def fit_bunch(X, y):
    predictors = []
    for i in range(len(X)):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X[i])

        pred = SVC(kernel='linear', probability=True)
        pred.fit(Xs, y)

        predictors.append((pred, scaler))

    return predictors

'''
Input: an array of the same shape as returned by load_all, and the output
of fit_bunch
Output: an array of shape (n_predictors, n_samples, 2)
'''
def predict_bunch(Z, predictors):
    predictions = []

    for i in range(len(Z)):
        #print('Predicting with classifier ' + str(i))
        Zs = predictors[i][1].transform(Z[i])

        pred = predictors[i][0].predict_proba(Zs)
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



'''
Train a model on data X and class labels y, then use the model to predict
class labels for Z.

Returns the predicted class labels.
'''
def train_and_predict(X, y, Z):
    clas = fit_bunch(X, y)
    predicted_indiv = predict_bunch(Z, clas)
    predicted_ensem = combine_predictions_mean(predicted_indiv)
    return predicted_ensem

# Testing

'''
Trains a model on data X and labels y, predicts the labels for Z, and computes
the loss with the actual labels.

Returns the loss.
'''
def validate(X, y, Z, actual):
    predicted = train_and_predict(X, y, Z)
    return data.evaluate(predicted, actual)

'''
Performs cross-validation on the entire training dataset.

Prints the mean loss.
'''
def xval(splits=5):
    X = load_all('train', range(278))
    y = data.get_targets(range(278))
    total_loss = 0

    # Stratified: keep classes somewhat balanced within folds

    # scikit-learn 0.18
    #kf = StratifiedKFold(n_splits=splits)
    #for i, (train, test) in enumerate(kf.split(X, y)):

    kf = StratifiedKFold(y, splits)
    for i, (train, test) in enumerate(kf):
        loss = validate(X[:,train,:], y[train], X[:,test,:], y[test])

        print('XVal fold ' + str(i) + ' loss: ' + str(loss))
        total_loss += loss

    print('Mean loss: ' + str(total_loss / splits))

def test():
    X = load_all('train', range(278))
    y = data.get_targets(range(278))
    Z = load_all('test', range(138))

    predictions = train_and_predict(X, y, Z)
    data.print_prediction(predicted_ensem)

if __name__ == '__main__':
    xval(3)
    #test()
