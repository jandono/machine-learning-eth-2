import sklearn.metrics
import sklearn.model_selection

def cross_entropy(actual, predicted):
    epsilon = 0.00001
    class0 = (1 - actual) * np.log((1 + epsilon)- predicted)
    class1 = actual * np.log(predicted + epsilon)
    return -np.mean(class0 + class1)


class AvgPredictor():
    def fit(__self__, X, y):
        #fit with svm
        #fit with rf
        #return self
        pass

    def predict(__self__, Z):
        #predict svm
        #predict rf
        #return result
        pass

xent_scorer = sklearn.metrics.make_scorer(cross_entropy, greater_is_better=False)
cross_val_score(AvgPredictor(), X, y, scoring=xent_scorer)
