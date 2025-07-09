# -*- coding: utf-8 -*-
#%% classes

import numpy as np
from itertools import product, combinations
from scipy.special import binom

class ValidationSplit():
    def __init__(self, labels_train, labels_test):

        self.labels_train = labels_train
        self.labels_test = labels_test

    def __iter__(self):
        yield np.arange(len(self.labels_train)), np.arange(len(self.labels_train), len(self.labels_train) + len(self.labels_test))

    def split(self, X, y=None):
        return self.__iter__()

    def __len__(self):
        return 1

    def get_n_splits(self, X=None, y=None):
        return 1


class ShuffleBinLeaveOneOut:

    def __init__(self, labels, n_iter=10, n_pseudo=5):

        """
        Parameters
        ----------
        labels : List<int>, np.ndarray()
            Label for each trial
        n_iter : int
            Number of permutations
        n_pseudo : int
            How many trials belong to one bin (aka pseudo-trial)
        """

        self.labels = np.array(labels)
        self.n_iter = n_iter

        self.classes, self.n_trials = np.unique(labels, return_counts=True)
        self.n_classes = self.classes.shape[0]
        self.n_pseudo = n_pseudo
        self._compute_pseudo_info()

    def _compute_pseudo_info(self):
        """
        Compute indices and labels for the pseudo-trial matrix
        The pseudo-trial matrix is the resulting matrix *after* having grouped the data into
        randomly permuted pseudo-trial bins and averaging trials within each bin. Thus, no
        additional permutation is necessary for pseudo-trial indices, which are somewhat trivial.
        """
        self.ind_pseudo_train = np.full((self.n_classes, self.n_classes, 2*(self.n_pseudo-1)),
                                        np.nan, dtype=int)
        self.ind_pseudo_test = np.full((self.n_classes, self.n_classes, 2), np.nan, dtype=int)
        self.labels_pseudo_train = np.full((self.n_classes, self.n_classes, 2*(self.n_pseudo-1)),
                                           np.nan, dtype=int)
        self.labels_pseudo_test = np.full((self.n_classes, self.n_classes, 2), np.nan, dtype=int)
        for c1 in range(self.n_classes):
            range_c1 = range(c1*(self.n_pseudo-1), (c1+1)*(self.n_pseudo-1))
            for c2 in range(self.n_classes):
                range_c2 = range(c2*(self.n_pseudo-1), (c2+1)*(self.n_pseudo-1))
                self.ind_pseudo_train[c1, c2, :2*(self.n_pseudo - 1)] =                     np.concatenate((range_c1, range_c2))
                self.ind_pseudo_test[c1, c2] = [c1, c2]

                self.labels_pseudo_train[c1, c2, :2*(self.n_pseudo - 1)] =                     np.concatenate((self.classes[c1] * np.ones(self.n_pseudo - 1),
                                    self.classes[c2] * np.ones(self.n_pseudo - 1)))
                self.labels_pseudo_test[c1, c2] = self.classes[[c1, c2]].astype(self.labels_pseudo_train.dtype)

    def __iter__(self):

        """
        Generator function for the cross-validation object. Each fold corresponds to a new random
        grouping of trials into pseudo-trials.
        """
        _ind_train = np.full(self.n_classes*(self.n_pseudo-1), np.nan, dtype=object)
        _ind_test = np.full(self.n_classes, np.nan, dtype=object)
        for perm in range(self.n_iter):
            for c1 in range(self.n_classes):  # separate permutation for each class
                prm = np.array(np.array_split(np.random.permutation(self.n_trials[c1]), self.n_pseudo),dtype=object)
                ind = prm + np.sum(self.n_trials[:c1])
                for i, j in enumerate(range(c1*(self.n_pseudo-1), (c1+1)*(self.n_pseudo-1))):
                    _ind_train[j] = ind[i]
                _ind_test[c1] = ind[-1]
            yield _ind_train, _ind_test

    def split(self, X, y=None):
        return self.__iter__()

    def get_n_splits(self, X=None, y=None):
        return self.n_iter

    def __len__(self):
        return self.n_iter

class ShuffleBinLeaveOneOutWithin:

    def __init__(self, labels, n_iter=10, n_pseudo=5):

        """
        Parameters
        ----------
        labels : List<int>, np.ndarray()
            Label for each trial
        n_iter : int
            Number of permutations
        n_pseudo : int
            How many trials belong to one bin (aka pseudo-trial)
        """

        self.labels = np.array(labels)
        self.n_iter = n_iter

        self.classes, self.n_trials = np.unique(labels, return_counts=True)
        self.n_classes = self.classes.shape[0]
        self.n_pseudo = n_pseudo
        self._compute_pseudo_info()

    def _compute_pseudo_info(self):
        """
        Compute indices and labels for the pseudo-trial matrix
        The pseudo-trial matrix is the resulting matrix *after* having grouped the data into
        randomly permuted pseudo-trial bins and averaging trials within each bin. Thus, no
        additional permutation is necessary for pseudo-trial indices, which are somewhat trivial.
        """
        self.ind_pseudo_train = np.full((self.n_classes, 1, self.n_pseudo-2),
                                        np.nan, dtype=int)
        self.ind_pseudo_test = np.full((self.n_classes, 1, 2), np.nan, dtype=int)
        self.labels_pseudo_train = np.full((self.n_classes, 1, self.n_pseudo-2),
                                           np.nan, dtype=int)
        self.labels_pseudo_test = np.full((self.n_classes, 1, 2), np.nan, dtype=int)
        for c1 in range(self.n_classes):
            range_c1 = range(c1*(self.n_pseudo-2), (c1+1)*(self.n_pseudo-2))

            self.ind_pseudo_train[c1, 0, :self.n_pseudo - 2] = range_c1
            self.ind_pseudo_test[c1, 0] = [c1*2, c1*2+1]

            self.labels_pseudo_train[c1, 0, :self.n_pseudo - 2] = self.classes[c1] * np.ones(self.n_pseudo - 2)
            self.labels_pseudo_test[c1, 0] = self.classes[[c1, c1]].astype(self.labels_pseudo_train.dtype)

    def __iter__(self):

        """
        Generator function for the cross-validation object. Each fold corresponds to a new random
        grouping of trials into pseudo-trials.
        """
        _ind_train = np.full(self.n_classes*(self.n_pseudo-2), np.nan, dtype=object)
        _ind_test = np.full(self.n_classes*2, np.nan, dtype=object)
        for perm in range(self.n_iter):
            for c1 in range(self.n_classes):  # separate permutation for each class
                prm = np.array(np.array_split(np.random.permutation(self.n_trials[c1]), self.n_pseudo))
                ind = prm + np.sum(self.n_trials[:c1])
                for i, j in enumerate(range(c1*(self.n_pseudo-2), (c1+1)*(self.n_pseudo-2))):
                    _ind_train[j] = ind[i]
                for i, j in enumerate(range(c1*2, (c1+1)*2)):
                    _ind_test[j] = ind[-i-1]
            yield _ind_train, _ind_test


    def split(self, X, y=None):
        return self.__iter__()

    def __len__(self):
        return self.n_iter

    def get_n_splits(self, X=None, y=None):
        return self.n_iter

class ShuffleBin:

    def __init__(self, labels, n_iter=10, n_pseudo=5):

        """
        Parameters
        ----------
        labels : List<int>, np.ndarray()
            Label for each trial
        n_iter : int
            Number of permutations
        n_pseudo : int
            How many trials belong to one bin (aka pseudo-trial)
        """

        self.labels = np.array(labels)
        self.n_iter = n_iter

        self.classes, self.n_trials = np.unique(labels, return_counts=True)
        self.n_classes = self.classes.shape[0]
        self.n_pseudo = n_pseudo
        self._compute_pseudo_info()

    def _compute_pseudo_info(self):
        """
        Compute indices and labels for the pseudo-trial matrix
        The pseudo-trial matrix is the resulting matrix *after* having grouped the data into
        randomly permuted pseudo-trial bins and averaging trials within each bin. Thus, no
        additional permutation is necessary for pseudo-trial indices, which are somewhat trivial.
        """
        self.ind_pseudo_test = np.full((self.n_classes, self.n_classes, 2 * self.n_pseudo),
                                       np.nan, dtype=int)
        self.labels_pseudo_test = np.full((self.n_classes, self.n_classes, 2 * self.n_pseudo),
                                          np.nan, dtype=int)

        for c1 in range(self.n_classes):
            range_c1 = range(c1*self.n_pseudo, (c1+1)*self.n_pseudo)
            for c2 in range(self.n_classes):
                range_c2 = range(c2*self.n_pseudo, (c2+1)*self.n_pseudo)
                self.ind_pseudo_test[c1, c2, :2 * self.n_pseudo] = np.concatenate((range_c1, range_c2))
                self.labels_pseudo_test[c1, c2, :2 * self.n_pseudo] =                     np.concatenate((self.classes[c1] * np.ones(self.n_pseudo),
                                    self.classes[c2] * np.ones(self.n_pseudo)))
        self.ind_pseudo_train = []
        self.labels_pseudo_train = []


    def __iter__(self):

        """
        Generator function for the cross-validation object. Each fold corresponds to a new random
        grouping of trials into pseudo-trials.
        """
        _ind_test = np.full(self.n_classes*self.n_pseudo, np.nan, dtype=object)
        _ind_train = []
        for perm in range(self.n_iter):
            for c1 in range(self.n_classes):  # separate permutation for each class
                prm = np.array(np.array_split(np.random.permutation(self.n_trials[c1]), self.n_pseudo))
                ind = prm + np.sum(self.n_trials[:c1])
                for i, j in enumerate(range(c1*self.n_pseudo, (c1+1)*self.n_pseudo)):
                    _ind_test[j] = ind[i]
            yield _ind_train, _ind_test


    def split(self, X, y=None):
        return self.__iter__()

    def __len__(self):
        return self.n_iter

    def get_n_splits(self, X=None, y=None):
        return self.n_iter


class ShuffleBinTest:

    def __init__(self, labels, n_iter=10, n_pseudo=5):

        """
        Parameters
        ----------
        labels : List<int>, np.ndarray()
            Label for each trial
        n_iter : int
            Number of permutations
        n_pseudo : int
            How many trials belong to one bin (aka pseudo-trial)
        """

        self.labels = np.array(labels)
        self.n_iter = n_iter

        self.classes, self.n_trials = np.unique(labels, return_counts=True)
        self.n_classes = self.classes.shape[0]
        self.n_pseudo = n_pseudo
        self._compute_pseudo_info()

    def _compute_pseudo_info(self):
        """
        Compute indices and labels for the pseudo-trial matrix
        The pseudo-trial matrix is the resulting matrix *after* having grouped the data into
        randomly permuted pseudo-trial bins and averaging trials within each bin. Thus, no
        additional permutation is necessary for pseudo-trial indices, which are somewhat trivial.
        """
        self.ind_pseudo_test = np.full((self.n_classes, self.n_classes, 2 * self.n_pseudo),
                                       np.nan, dtype=int)
        self.labels_pseudo_test = np.full((self.n_classes, self.n_classes, 2 * self.n_pseudo),
                                          np.nan, dtype=int)

        for c1 in range(self.n_classes):
            range_c1 = range(c1*self.n_pseudo, (c1+1)*self.n_pseudo)
            for c2 in range(self.n_classes):
                range_c2 = range(c2*self.n_pseudo, (c2+1)*self.n_pseudo)
                self.ind_pseudo_test[c1, c2, :2 * self.n_pseudo] = np.concatenate((range_c1, range_c2))
                self.labels_pseudo_test[c1, c2, :2 * self.n_pseudo] =                     np.concatenate((self.classes[c1] * np.ones(self.n_pseudo),
                                    self.classes[c2] * np.ones(self.n_pseudo)))
        self.ind_pseudo_train = []
        self.labels_pseudo_train = []


    def __iter__(self):

        """
        Generator function for the cross-validation object. Each fold corresponds to a new random
        grouping of trials into pseudo-trials.
        """
        _ind_test = np.full(self.n_classes*self.n_pseudo, np.nan, dtype=object)
        _ind_train = []
        for perm in range(self.n_iter):
            for c1 in range(self.n_classes):  # separate permutation for each class
                prm = np.array(np.array_split(range(self.n_trials[c1]), self.n_pseudo))
                ind = prm + np.sum(self.n_trials[:c1])
                for i, j in enumerate(range(c1*self.n_pseudo, (c1+1)*self.n_pseudo)):
                    _ind_test[j] = ind[i]
            yield _ind_train, _ind_test


    def split(self, X, y=None):
        return self.__iter__()

    def __len__(self):
        return self.n_iter

    def get_n_splits(self):
        return self.n_iter


class ShuffleBinWithin:

    def __init__(self, labels, n_iter=10, n_pseudo=5):

        """
        Parameters
        ----------
        labels : List<int>, np.ndarray()
            Label for each trial
        n_iter : int
            Number of permutations
        n_pseudo : int
            How many trials belong to one bin (aka pseudo-trial)
        """

        self.labels = np.array(labels)
        self.n_iter = n_iter

        self.classes, self.n_trials = np.unique(labels, return_counts=True)
        self.n_classes = self.classes.shape[0]
        self.n_pseudo = n_pseudo
        self._compute_pseudo_info()

    def _compute_pseudo_info(self):
        """
        Compute indices and labels for the pseudo-trial matrix
        The pseudo-trial matrix is the resulting matrix *after* having grouped the data into
        randomly permuted pseudo-trial bins and averaging trials within each bin. Thus, no
        additional permutation is necessary for pseudo-trial indices, which are somewhat trivial.
        """
        self.ind_pseudo_test = np.full((self.n_classes, 1, self.n_pseudo),
                                       np.nan, dtype=int)
        self.labels_pseudo_test = np.full((self.n_classes, 1, self.n_pseudo),
                                          np.nan, dtype=int)
        for c1 in range(self.n_classes):
            range_c1 = range(c1*self.n_pseudo, (c1+1)*self.n_pseudo)

            self.ind_pseudo_test[c1, 0, :self.n_pseudo] = range_c1
            self.labels_pseudo_test[c1, 0, :self.n_pseudo] = self.classes[c1] * np.ones(self.n_pseudo)
        self.ind_pseudo_train = []
        self.labels_pseudo_train = []


    def __iter__(self):

        """
        Generator function for the cross-validation object. Each fold corresponds to a new random
        grouping of trials into pseudo-trials.
        """
        _ind_test = np.full(self.n_classes*self.n_pseudo, np.nan, dtype=object)
        _ind_train = []
        for perm in range(self.n_iter):
            for c1 in range(self.n_classes):  # separate permutation for each class
                prm = np.array(np.array_split(np.random.permutation(self.n_trials[c1]), self.n_pseudo))
                ind = prm + np.sum(self.n_trials[:c1])
                for i, j in enumerate(range(c1*self.n_pseudo, (c1+1)*self.n_pseudo)):
                    _ind_test[j] = ind[i]
            yield _ind_train, _ind_test

    def split(self, X, y=None):
        return self.__iter__()

    def __len__(self):
        return self.n_iter

    def get_n_splits(self, X=None, y=None):
        return self.n_iter


class ShuffleBinWithinTest:

    def __init__(self, labels, n_iter=10, n_pseudo=5):

        """
        Parameters
        ----------
        labels : List<int>, np.ndarray()
            Label for each trial
        n_iter : int
            Number of permutations
        n_pseudo : int
            How many trials belong to one bin (aka pseudo-trial)
        """

        self.labels = np.array(labels)
        self.n_iter = n_iter

        self.classes, self.n_trials = np.unique(labels, return_counts=True)
        self.n_classes = self.classes.shape[0]
        self.n_pseudo = n_pseudo
        self._compute_pseudo_info()

    def _compute_pseudo_info(self):
        """
        Compute indices and labels for the pseudo-trial matrix
        The pseudo-trial matrix is the resulting matrix *after* having grouped the data into
        randomly permuted pseudo-trial bins and averaging trials within each bin. Thus, no
        additional permutation is necessary for pseudo-trial indices, which are somewhat trivial.
        """
        self.ind_pseudo_test = np.full((self.n_classes, 1, self.n_pseudo),
                                       np.nan, dtype=int)
        self.labels_pseudo_test = np.full((self.n_classes, 1, self.n_pseudo),
                                          np.nan, dtype=int)
        for c1 in range(self.n_classes):
            range_c1 = range(c1*self.n_pseudo, (c1+1)*self.n_pseudo)

            self.ind_pseudo_test[c1, 0, :self.n_pseudo] = range_c1
            self.labels_pseudo_test[c1, 0, :self.n_pseudo] = self.classes[c1] * np.ones(self.n_pseudo)
        self.ind_pseudo_train = []
        self.labels_pseudo_train = []


    def __iter__(self):

        """
        Generator function for the cross-validation object. Each fold corresponds to a new random
        grouping of trials into pseudo-trials.
        """
        _ind_test = np.full(self.n_classes*self.n_pseudo, np.nan, dtype=object)
        _ind_train = []
        for perm in range(self.n_iter):
            for c1 in range(self.n_classes):  # separate permutation for each class
                prm = np.array(np.array_split(range(self.n_trials[c1]), self.n_pseudo))
                ind = prm + np.sum(self.n_trials[:c1])
                for i, j in enumerate(range(c1*self.n_pseudo, (c1+1)*self.n_pseudo)):
                    _ind_test[j] = ind[i]
            yield _ind_train, _ind_test

    def split(self, X, y=None):
        return self.__iter__()

    def __len__(self):
        return self.n_iter

    def get_n_splits(self, X=None, y=None):
        return self.n_iter

class XClassSplit():

    def __init__(self, runs, sets):
        self.sets = np.atleast_2d(sets)
        self.runs = np.array(runs, dtype=int)

        self.unique_runs = np.unique(self.runs)
        self.unique_sets = np.atleast_2d([np.unique(s) for s in self.sets])
        self.n = sum([len(s) * len(self.unique_runs) for s in self.unique_sets])

    def __iter__(self):

        for s, set in enumerate(self.sets):
            for set_id in self.unique_sets[s]:
                for run in self.unique_runs:
                    test_index = np.where((set == set_id) & (self.runs == run))[0]
                    train_index = np.where((set != set_id) & (self.runs != run))[0]
                    yield train_index, test_index

    def split(self, X, y=None):
        return self.__iter__()

    def __len__(self):
        return self.n

    def get_n_splits(self, X=None, y=None):
        return self.n


# class ExhaustiveLeave2Out:
#
#     def __init__(self, labels):
#         self.labels = labels
#         self.classes = np.unique(self.labels)
#         self.n_samples = len(self.labels)
#         n_samples1 = np.sum(self.labels == self.classes[0])
#         n_samples2 = np.sum(self.labels == self.classes[1])
#         self.n_iter = 2 * n_samples1 * n_samples2
#
#     def __iter__(self):
#         for i, l in enumerate(self.labels):
#             other_class = self.classes[0] if l == self.classes[1] else self.classes[1]
#             ind_other_class = np.where(self.labels == other_class)[0]
#             for i in ind_other_class:
#                 test_ind = [i, i]
#                 train_ind = np.setdiff1d(range(self.n_samples), test_ind)
#                 yield train_ind, test_ind
#
#     def split(self, X, y=None):
#         return self.__iter__()
#
#     def __len__(self):
#         return self.n_iter
#
#     def get_n_splits(self):
#         return self.n_iter


class SuperExhaustiveLeaveNOut:

    def __init__(self, N):
        self.N = N

    def __iter__(self, y):
        for test_ind in combinations(range(len(y)), self.N):
            train_ind = np.setdiff1d(range(len(y)), test_ind)
            yield train_ind, np.array(test_ind)

    def split(self, X, y, groups=None):
        return self.__iter__(y)

    def get_n_splits(self, X, y, groups=None):
        return int(binom(len(y), self.N))


class ExhaustiveLeave2Out:

    def __init__(self):
        pass

    def __iter__(self, y):
        classes = np.unique(y)
        ind1 = np.where(y == classes[0])[0]
        ind2 = np.where(y == classes[1])[0]
        for test_ind in product(ind1, ind2):
            train_ind = np.setdiff1d(range(len(y)), test_ind)
            yield train_ind, np.array(test_ind)

    def split(self, X, y, groups=None):
        return self.__iter__(y)

    def get_n_splits(self, X, y, groups=None):
        classes = np.unique(y)
        n_samples1 = np.sum(y == classes[0])
        n_samples2 = np.sum(y == classes[1])
        return n_samples1 * n_samples2


class SubsetLeave2Out:

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def __iter__(self, y):
        classes = np.unique(y)
        ind1 = np.where(y == classes[0])[0]
        ind2 = np.where(y == classes[1])[0]
        combos = list(product(ind1, ind2))
        order = np.random.choice(len(combos), self.n_splits, replace=False)
        for i in range(self.n_splits):
            test_ind = np.array(combos[order[i]])
            train_ind = np.setdiff1d(range(len(y)), test_ind)
            yield train_ind, test_ind

    def split(self, X, y, groups=None):
        return self.__iter__(y)

    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits


class ProxyCV:

    def __init__(self, train_ind, test_ind):
        self.train_ind = train_ind
        self.test_ind = test_ind

    def __iter__(self):
        yield self.train_ind, self.test_ind

    def split(self, X, y=None):
        return self.__iter__()

    def __len__(self):
        return 1

    def get_n_splits(self):
        return 1


class DummyCV:

    def __init__(self, n_samples):
        self.n_samples = n_samples

    def __iter__(self):
        yield list(range(self.n_samples)), []

    def split(self, X, y=None):
        return self.__iter__()

    def __len__(self):
        return 1

    def get_n_splits(self):
        return 1

#%% functions 
def run_eeg_svm_guggenmos(X, y, n_pseudo, n_perm = 0,n_jobs = -1):
    
    import numpy as np
    import scipy
    from sklearn.discriminant_analysis import _cov
    from sklearn.svm import SVC
    import time
    import pdb
    np.random.seed(10)
    tStart = time.time()
    
    svm = SVC(kernel='linear')
    CV = ShuffleBinLeaveOneOut
    print('Npseudo:',n_pseudo)

    n_sensors = X.shape[1]
    n_time = X.shape[2]
    n_conditions = len(np.unique(y))
    cv = CV(y, n_iter=n_perm, n_pseudo=n_pseudo)
    result = np.full((n_perm, n_conditions, n_conditions,n_time), np.nan)
    dissimilarity    = np.full((n_perm, n_conditions, n_conditions,n_time), np.nan)
    coefs  = np.full((n_perm, n_conditions, n_conditions,n_sensors,n_time), np.nan)
    coefs_scaled  = np.full((n_perm, n_conditions, n_conditions,n_sensors,n_time), np.nan)
    coefs_scaled_whitened  = np.full((n_perm, n_conditions, n_conditions,n_sensors,n_time), np.nan)

    for f, (train_indices, test_indices) in enumerate(cv.split(X)):
                print('\tPermutation %g / %g' % (f + 1, n_perm))

                # 1. Compute pseudo-trials for training and test
                Xpseudo_train = np.full((len(train_indices), n_sensors, n_time), np.nan)
                Xpseudo_test = np.full((len(test_indices), n_sensors, n_time), np.nan)

                # pdb.set_trace()
                for i, ind in enumerate(train_indices):
                    Xpseudo_train[i, :, :] = np.mean(X[ind.astype(int), :, :], axis=0)
                for i, ind in enumerate(test_indices):
                    Xpseudo_test[i, :, :] = np.mean(X[ind.astype(int), :, :], axis=0)
                
                # save the original values before whitening
                Xpseudo_all = np.concatenate((Xpseudo_train.copy(),Xpseudo_test.copy()),axis = 0)
                # 2. Whitening using the Epoch method
                sigma_conditions = cv.labels_pseudo_train[0, :, n_pseudo-1:].flatten()
                sigma_ = np.empty((n_conditions, n_sensors, n_sensors))
                for k,c in enumerate(np.unique(y)):
                    # compute sigma for each time point, then average across time
                    sigma_[k] = np.mean([_cov(Xpseudo_train[sigma_conditions==c, :, t], shrinkage='auto')
                                         for t in range(n_time)], axis=0)
                sigma = sigma_.mean(axis=0)  # average across conditions
                sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)
                Xpseudo_train = (Xpseudo_train.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)
                Xpseudo_test = (Xpseudo_test.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)

            
                for t in range(n_time):
                    for c1 in range(n_conditions-1):
                        for c2 in range(min(c1 + 1, n_conditions-1), n_conditions):
                                # 3. Fit the classifier using training data
                                data_train = Xpseudo_train[cv.ind_pseudo_train[c1, c2], :, t]
                                svm.fit(data_train, cv.labels_pseudo_train[c1, c2])                            
                                # 4. Compute and store classification accuracies
                                data_test = Xpseudo_test[cv.ind_pseudo_test[c1, c2], :, t]
                                predictions = svm.predict(data_test)
                                result[f, c1, c2, t] = np.mean(predictions == cv.labels_pseudo_test[c1, c2]) - 0.5                
                                # 5. compute dissimilarity based on equation 3: DV weighted decoding accuracy
                                dist = svm.decision_function(data_test) / np.linalg.norm(svm.coef_)
                                dissimilarity[f, c1, c2, t] = np.mean(np.multiply((predictions == cv.labels_pseudo_test[c1, c2]).astype(int)-.5,dist))

                                # 6. Extract the model coefs (channel weights)
                                coefs[f,c1,c2,:,t]  = svm.coef_
                                coefs_scaled[f,c1,c2,:,t] = np.dot(np.cov(Xpseudo_all[:,:,t].transpose()),svm.coef_.transpose()).transpose()
                                
                                
    # average across permutations
    result = np.nanmean(result, axis=0)
    dissimilarity = np.nanmean(dissimilarity, axis = 0)
    coefs  = np.nanmean(coefs, axis=0)
    coefs_scaled  = np.nanmean(coefs_scaled, axis=0)
    coefs_scaled_whitened  = np.nanmean(coefs_scaled_whitened, axis=0)

    duration=time.time() - tStart
    duration=duration/60


    return result, dissimilarity, coefs, coefs_scaled, duration


def run_eeg_svm_guggenmos_tempGen(X, y, n_pseudo, n_perm = 0,n_jobs = -1):
    
    import numpy as np
    import scipy
    from sklearn.discriminant_analysis import _cov
    from sklearn.svm import SVC
    import time
    import pdb
    from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    np.random.seed(10)
    tStart = time.time()
    
    CV = ShuffleBinLeaveOneOut
    clf = make_pipeline(StandardScaler(),LinearSVC(C=1.0, max_iter = 10000));
    time_gen = GeneralizingEstimator(clf, n_jobs=1, scoring=None, verbose=True)
    n_sensors = X.shape[1]
    n_time = X.shape[2]
    n_conditions = len(np.unique(y))
    cv = CV(y, n_iter=n_perm, n_pseudo=n_pseudo)
    result = np.full((n_perm, n_conditions, n_conditions,n_time,n_time), np.nan)
    dissimilarity    = np.full((n_perm, n_conditions, n_conditions,n_time), np.nan)
    coefs  = np.full((n_perm, n_conditions, n_conditions,n_sensors,n_time), np.nan)
    coefs_scaled  = np.full((n_perm, n_conditions, n_conditions,n_sensors,n_time), np.nan)
    coefs_scaled_whitened  = np.full((n_perm, n_conditions, n_conditions,n_sensors,n_time), np.nan)

    for f, (train_indices, test_indices) in enumerate(cv.split(X)):
                print('\tPermutation %g / %g' % (f + 1, n_perm))

                # 1. Compute pseudo-trials for training and test
                Xpseudo_train = np.full((len(train_indices), n_sensors, n_time), np.nan)
                Xpseudo_test = np.full((len(test_indices), n_sensors, n_time), np.nan)

                # pdb.set_trace()
                for i, ind in enumerate(train_indices):
                    Xpseudo_train[i, :, :] = np.mean(X[ind.astype(np.int64), :, :], axis=0)
                for i, ind in enumerate(test_indices):
                    Xpseudo_test[i, :, :] = np.mean(X[ind.astype(np.int64), :, :], axis=0)
                
                # save the original values before whitening
                Xpseudo_all = np.concatenate((Xpseudo_train.copy(),Xpseudo_test.copy()),axis = 0)
                # 2. Whitening using the Epoch method
                sigma_conditions = cv.labels_pseudo_train[0, :, n_pseudo-1:].flatten()
                sigma_ = np.empty((n_conditions, n_sensors, n_sensors))
                for k,c in enumerate(np.unique(y)):
                    # compute sigma for each time point, then average across time
                    sigma_[k] = np.mean([_cov(Xpseudo_train[sigma_conditions==c, :, t], shrinkage='auto')
                                         for t in range(n_time)], axis=0)
                sigma = sigma_.mean(axis=0)  # average across conditions
                sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)
                Xpseudo_train = (Xpseudo_train.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)
                Xpseudo_test = (Xpseudo_test.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)

            
                for c1 in range(n_conditions-1):
                    for c2 in range(min(c1 + 1, n_conditions-1), n_conditions):
                        # 3. Fit the classifier using training data
                        data_train = Xpseudo_train[cv.ind_pseudo_train[c1, c2], :, :]
                        time_gen.fit(data_train, cv.labels_pseudo_train[c1, c2])                            
                        # 4. Compute and store classification accuracies
                        data_test = Xpseudo_test[cv.ind_pseudo_test[c1, c2], :, :]           
                        pred_array = np.full((2, n_time,n_time), np.nan)
                        pred_array[0,:,:] = cv.labels_pseudo_test[c1, c2][0]
                        pred_array[1,:,:] = cv.labels_pseudo_test[c1, c2][1]           
                        predictions = time_gen.predict(data_test)
                        result[f, c1, c2, :,:] = np.nanmean(predictions == pred_array,axis=0) - 0.5        
    result = np.nanmean(np.nanmean(np.nanmean(result,axis=0),axis=0),axis=0)

    duration=time.time() - tStart
    duration=duration/60
    return result, duration

def run_eeg_svm_guggenmos_SVR(X, y, n_pseudo, n_perm = 20,n_jobs = -1):
    
    from sklearn.svm import SVR
    from sklearn.model_selection import permutation_test_score
    from scipy.stats import pearsonr
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold, permutation_test_score
    import numpy as np
    import scipy
    from sklearn.discriminant_analysis import _cov
    from sklearn.svm import SVC
    import time
    import pdb
    np.random.seed(10)
    tStart = time.time()



    svr_lin = SVR(kernel="linear", C=1)
    CV = ShuffleBinLeaveOneOut
    n_sensors = X.shape[1]
    n_time = X.shape[2]
    n_pseudo = 5
    n_conditions = len(np.unique(y))
    cv = CV(y, n_iter=n_perm, n_pseudo=n_pseudo)
    result = np.full((n_perm, n_conditions, n_conditions,n_time), np.nan)
    dissimilarity    = np.full((n_perm, n_conditions, n_conditions,n_time), np.nan)
    coefs  = np.full((n_perm, n_conditions, n_conditions,n_sensors,n_time), np.nan)
    coefs_scaled  = np.full((n_perm, n_conditions, n_conditions,n_sensors,n_time), np.nan)
    coefs_scaled_whitened  = np.full((n_perm, n_conditions, n_conditions,n_sensors,n_time), np.nan)
    cv = StratifiedKFold(2, shuffle=True, random_state=0)

    # 1. Compute pseudo-trials for training and test          
    sigma_conditions = y.flatten()
    sigma_ = np.empty((n_conditions, n_sensors, n_sensors))
    for k,c in enumerate(np.unique(y)):
        # compute sigma for each time point, then average across time
        sigma_[k] = np.mean([_cov(X[sigma_conditions==c, :, t], shrinkage='auto')
                                        for t in range(n_time)], axis=0)
    sigma = sigma_.mean(axis=0)  # average across conditions
    sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)
    X_whitened = (X.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)

    overall_score = []
    coefficients = np.full((64,220),np.nan)
    coefficients_scaled = np.full((64,220),np.nan)
    for t in range(0,220):
        thisScore = []
        thisWeight = np.full((20,64),np.nan)
        for perm in range(0,20):
            X_train, X_test, y_train, y_test = train_test_split(X_whitened, y, test_size=0.2, random_state=perm)
            svr_lin.fit(X_train[:,:,t],y_train)
            thisWeight[perm,:] = svr_lin.coef_
            predictions = svr_lin.predict(X_test[:,:,t])
            corrCoef,stat = pearsonr(predictions,y_test)
            thisScore.append(.5*np.log(1+corrCoef)/(1-corrCoef))
        overall_score.append(np.nanmean(thisScore))
        coefficients[:,t] = np.nanmean(thisWeight,axis=0)
        coefficients_scaled[:,t] = np.dot(np.cov(X_whitened[:,:,t].transpose()),coefficients[:,t].transpose()).transpose()
    overall_score = np.array(overall_score)
    return overall_score,coefficients, coefficients_scaled
