import numpy as np
import nilearn.image
import nilearn.plotting
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from skimage.measure import block_reduce
from scipy.stats import mode


path_train = 'data/set_train/mri/'#4mm/'
path_test = 'data/set_test/mri/'#4mm/'

train_size = 278
test_size = 138

cut = {}
cut['cut_x'] = (16, 160) # original 18 158
cut['cut_y'] = (19, 190) # original 19 189
cut['cut_z'] = (11, 155) # original 13 153

# cut['cut_x'] = (16, 160) # original 18 158
# cut['cut_y'] = (19, 190) # original 19 189
# cut['cut_z'] = (11, 155) # original 13 153


y = np.genfromtxt('data/targets.csv', delimiter=',')
y = y[0:train_size]
y = np.array(y)

x = []
t = []

for i in xrange(0, train_size):
    img = nilearn.image.load_img(path_train + 'mwp1train_' + str(i + 1) + '.nii')
    d = img.get_data()
    d = d[cut['cut_x'][0]:cut['cut_x'][1],  cut['cut_y'][0]:cut['cut_y'][1],  cut['cut_z'][0]:cut['cut_z'][1]]
    cut_img = nib.Nifti1Image(d, affine=np.eye(4))
    d = block_reduce(d, block_size=(8, 8, 8), func=np.mean)
    x.append(np.ravel(d))

nilearn.plotting.show()

for i in xrange(0, test_size):
    img = nilearn.image.load_img(path_test + 'mwp1test_' + str(i + 1) + '.nii')
    d = img.get_data()
    d = d[cut['cut_x'][0]:cut['cut_x'][1],  cut['cut_y'][0]:cut['cut_y'][1],  cut['cut_z'][0]:cut['cut_z'][1]]
    d = block_reduce(d, block_size=(8, 8, 8), func=np.mean)
    t.append(np.ravel(d))

#clf = svm.SVC(kernel='linear', probability=True)
clf = AdaBoostClassifier(svm.SVC(kernel='linear', probability=True))
clf.fit(x, y)
final_results = clf.predict_proba(t)
final_results = final_results[:, 1]

f = open('submission.txt', 'w+')
f.write('ID,Prediction\n')
i = 1
for result in final_results:
    f.write(str(i) + ',' + str(result) + '\n')
    i += 1

