import nilearn.image
import numpy as np
import scipy.ndimage

DATA = '../data'
TARGETS = DATA + '/targets.csv'

# To load a file and do all possible preprocessing on it, call:
#   load_and_preprocess(['train'|'test'], <file_id>)
#
# See individual functions for more details.

# =========== Loading images, parameters ===========

def load_and_preprocess(typ, n, params=None):
    if params is None:
        params = construct_defaults()

    check_consistent_params(params)

    img = load_datum(typ, n)
    if params['fwhm'] != 1:
        img = smooth(img)
    raw = np.squeeze(img.get_data()) # Remove extraneous 4th dimension
    cooked = cut_irrelevant_dimensions(raw, params['cut_x'], params['cut_y'], params['cut_z'])

    if params['fwhm'] != 1:
        cooked = reduce(cooked, 1 / float(params['fwhm']))

    return cubeify(cooked, params['n_cubes'])


def load_datum(typ, n):
    path = '%s/set_%s/%s_%d.nii' % (DATA, typ, typ, n)
    return nilearn.image.load_img(path)

def check_consistent_params(p):
    # We would like that every small cube at the end be the same size,
    # if we want to use them independently (e.g. train one estimator for each
    # of them or something like that)
    #
    # Therefore we want that for all dimensions:
    #   ((cut_end - cut_start) * zoom) / n_cubes ----> is a whole number
    #

    def check_dim(a, b, fwhm, n_cubes):
        make_dim(a, b, fwhm, n_cubes).is_integer()

    if (not check_dim(p['cut_x'][0], p['cut_x'][1], p['fwhm'], p['n_cubes']) or
        not check_dim(p['cut_y'][0], p['cut_y'][1], p['fwhm'], p['n_cubes']) or
        not check_dim(p['cut_z'][0], p['cut_z'][1], p['fwhm'], p['n_cubes'])):
        pass # raise Exception('Inconsistent parameters')

def make_dim(a, b, fwhm, n_cubes):
        return ((b - a) / float(fwhm) / float(n_cubes))

'''
Mostly for debugging. Indicates the shape one of these small cubes.
'''
def final_dimension(p):
    return (make_dim(p['cut_x'][0], p['cut_x'][1], p['fwhm'], p['n_cubes']),
            make_dim(p['cut_y'][0], p['cut_y'][1], p['fwhm'], p['n_cubes']),
            make_dim(p['cut_z'][0], p['cut_z'][1], p['fwhm'], p['n_cubes']))

def construct_defaults():
    return {
        # Which ranges to keep
        'cut_x': (16,160), # original 18 158
        'cut_y': (20,188), # original 19 189
        'cut_z': (11,155), # original 13 153

        # How much smoothing to apply
        'fwhm': 4,

        # Zoom factor is defined as 1 / fwhm
        #'zoom_factor': 0.25

        # Split in how many cubes for each dimension
        'n_cubes': 3
    }

# =========== Preprocessing ===========

'''
Apply the nilearn smoothing.
'''
def smooth(img, fwhm):
    return nilearn.image.smooth_img(img, fwhm)

'''
Shorten the axis to remove areas that have almost no information.
'''
def cut_irrelevant_dimensions(dat, cut_x, cut_y, cut_z):
    return dat[cut_x[0]:cut_x[1],  cut_y[0]:cut_y[1],  cut_z[0]:cut_z[1]]

'''
Reduce the dimensions of the image.
'''
def reduce(data, zoom_factor):
    return scipy.ndimage.zoom(data, zoom_factor)

'''
Transform this cube into several smaller cubes.
Returns a 4-D array, where all the small cubes are stacked one after the other.

Assumes that n_cubes evenly divides the data in every dimension.
'''
def cubeify(data, n_cubes):
    szx = int(data.shape[0] / n_cubes)
    szy = int(data.shape[1] / n_cubes)
    szz = int(data.shape[2] / n_cubes)
    cubes = []
    for x in range(n_cubes):
        for y in range(n_cubes):
            for z in range(n_cubes):
                sx = szx * x
                sy = szy * y
                sz = szz * z
                cub = data[sx:(sx + szx),  sy:(sy + szy),  sz:(sz + szz)]
                cubes.append(cub)

    return np.array(cubes)


# ============ Targets and predictions ============

def get_targets():
    np.genfromtxt(TARGETS, delimiter=',')

def print_prediction(preds):
    print('ID,Prediction')
    for i, p in enumerate(preds):
        print(str(i + 1) + ',' + str(p))
