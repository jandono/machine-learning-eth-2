import numpy as np
import nilearn.image
import nilearn.plotting
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import svm
from skimage.measure import block_reduce
from random import shuffle
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection

def cross_entropy_proba(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted[:,1])
    epsilon = 0.00001
    class0 = (1 - actual) * np.log((1 + epsilon)- predicted)
    class1 = actual * np.log(predicted + epsilon)
    return np.mean(class0 + class1)

def cross_entropy(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    epsilon = 0.00001
    class0 = (1 - actual) * np.log((1 + epsilon)- predicted)
    class1 = actual * np.log(predicted + epsilon)
    return np.mean(class0 + class1)

class AvgPredictor():

    def __init__(self):
        self.clf_svm = svm.SVC(kernel='linear', probability=True)
        self.clf_rf = RandomForestClassifier(n_estimators=1000, max_depth=10)

    def fit(self, X, y):
        # fit with svm
        self.clf_svm.fit(X, y)
        # fit with rf
        self.clf_rf.fit(X, y)

        return self

    def get_params(self, deep=True):
        if deep:
            return self.clf_rf.get_params(deep) | self.clf_svm.get_params(deep)
        else:
            return {}

    def predict(self, t):
        final_results = []

        # predict svm
        final_results.append(self.clf_svm.predict_proba(t))
        final_results[0] = final_results[0][:, 1]

        # predict rf
        final_results.append(self.clf_rf.predict_proba(t))
        final_results[1] = final_results[1][:, 1]


        samples = len(t)

        result = list(np.zeros(samples))
        for i in xrange(0, samples):
            result[i] = (final_results[0][i] + final_results[1][i]) / 2

        return result


path_train = 'data/set_train/mri/'#4mm/'
path_test = 'data/set_test/mri/'#4mm/'

train_size = 278
test_size = 138

cut = {}
cut['cut_x'] = (16, 160) # original 18 158
cut['cut_y'] = (19, 190) # original 19 189
cut['cut_z'] = (11, 155) # original 13 153


y = np.genfromtxt('data/targets.csv', delimiter=',')
y = y[0:train_size]
y = list(np.array(y))

x = []
t = []

for i in xrange(0, train_size):
    img = nilearn.image.load_img(path_train + 'mwp1train_' + str(i + 1) + '.nii')
    d = img.get_data()
    d = d[cut['cut_x'][0]:cut['cut_x'][1],  cut['cut_y'][0]:cut['cut_y'][1],  cut['cut_z'][0]:cut['cut_z'][1]]
    d = block_reduce(d, block_size=(8, 8, 8), func=np.mean)
    x.append(np.ravel(d))

for i in xrange(0, test_size):
    img = nilearn.image.load_img(path_test + 'mwp1test_' + str(i + 1) + '.nii')
    d = img.get_data()
    d = d[cut['cut_x'][0]:cut['cut_x'][1],  cut['cut_y'][0]:cut['cut_y'][1],  cut['cut_z'][0]:cut['cut_z'][1]]
    d = block_reduce(d, block_size=(8, 8, 8), func=np.mean)
    t.append(np.ravel(d))

# clf_knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
# clf_knn.fit(x,y)
# results = clf_knn.predict_proba(t)
# results = results[:, 1]


final_results = []

# clf1 = svm.SVC(kernel='linear', probability=True)
# clf1.fit(x, y)
# final_results.append(clf1.predict_proba(t))
# final_results[0] = final_results[0][:, 1]

clf2 = RandomForestClassifier(n_estimators=300, max_depth=5)
clf2.fit(x, y)
results = clf2.predict_proba(t)
results = results[:, 1]
# final_results.append(clf2.predict_proba(t))
# final_results[1] = final_results[1][:, 1]

# clf3 = svm.SVC(kernel='rbf', probability=True)
# clf3.fit(x, y)
# final_results.append(clf2.predict_proba(t))
# final_results[2] = final_results[2][:, 1]

xent_scorer = sklearn.metrics.make_scorer(cross_entropy_proba, needs_proba=True, greater_is_better=False)
# xent_scorer = sklearn.metrics.make_scorer(cross_entropy, greater_is_better=False)
crs_val_s = cross_val_score(clf2, x, y, scoring=xent_scorer, cv=5)
print crs_val_s
print np.mean(crs_val_s)
print np.std(crs_val_s)

f = open('submission.txt', 'w+')
f.write('ID,Prediction\n')
for i in xrange(0, test_size):
    f.write(str(i+1) + ',' + str(results[i]) + '\n')
    i += 1


# f = open('submission.txt', 'w+')
# f.write('ID,Prediction\n')
#
# for i in xrange(0, test_size):
#     f.write(str(i+1) + ',' + str((final_results[0][i] + final_results[1][i])/2) + '\n')
#     i += 1


