##################################################################################
# The script is used to define a class for data generation for clustering or other
# applications
#
# @author Wang Shuai
# @date 2019.06.06
##################################################################################

from __future__ import division
import pandas as pd
import re
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn import manifold

import math
import numpy as np
#import hypertools as hyp  # used to visualize data
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import pairwise_kernels
from sklearn.metrics.pairwise import euclidean_distances
from file_manager import FileManager
import numpy.linalg as LA
from scipy.spatial import distance_matrix
from equal_groups import EqualGroupsKMeans

class DataGenerator(object):
    """ This class is design for generating data for distributed computing.

    There are two kinds of centralized data to be generated:
        1. synthetic data: we can generate it following some rules
        2. real data: we should download these data files in advance,
                read them from files, and preprocess them
    More importantly, in order for distributed computing, the class should be
    able to partition the centralized data and distribute it into several
    partitions each of which correponds to one local node.

    Attributes:
        root_dir (string): root directory under which the data files should be stored
        is_real (boolean): the flag indicates whether the data to be used is synthetic or real
        data_name (string): the name of the data
        data_mat (numpy array or mat): the data set with rows being features
        num_of_pars (int): the number of partitions on the data
        par_idx (numpy array): a 1D array indicating the partition of the data
                                with each element being chosen from 0~ (num_of_pars - 1)
        num_of_cls (int): the number of clusters the data are chosen from
        true_labels (numpy array): a 1D array indicating the ground truth for cluster assignments
        existed (boolean): a flag indicates whether the data has been generated and saved before
                        False: this is the first time to generated the data, and True otherwise
                        Note that the data includes data mat, labels and partition index. Any of
                        them has not been generated will make the flag being False
    """
    def __init__(self, root_dir, is_real, data_name, num_of_features = 2000, num_of_samples = 1000, \
            num_of_cls = 10, num_of_pars = 100, seed = 0):
        """ This function is to initialize the class.
        Specifically, it will generate the centralized data set with input args

        Args:
            root_dir (string): root dir to read and write data sets
            is_real (boolean): the flag indicating to use synthetic data or real data
                            True: real data, False: synthetic data
            data_name (string): It has the form of "XXX1" in which the first substring is the folder
                            name (dataset name) and the second one is data number of this kind of data
            num_of_features (int): used when is_real is False. The number of features for the dataset
            num_of_samples (int): used when is_real is False. The number of data samples for the dataset
            num_of_cls (int): used when is_real is False.
                            the number of clusters from which the synthetic dataset is chosen
            seed (int): the seed for random data generation. Since we will use randomness for patitioning
                        and synthetic data generation, the seed is set at first for reproducation.
        """
        self.root_dir = root_dir
        self.is_real = is_real
        self.num_of_cls = num_of_cls
        self.data_name = data_name
        self.num_of_features = num_of_features
        self.num_of_samples = num_of_samples
        if is_real:  # save the newly generated data set so that we don't need to regenerate it again
            # we process the real name to get the kind (name) of the data set and the number
            #data_kind, data_num = self.__get_data_kind_and_num(self.data_name)
            if self.data_name.startswith('mnist'):
                self.data_kind = 'mnist'
                # get the paths for real dataset and its label
                if not self.data_name.endswith('unbal'):
                    bal_str = 'balanced_noniid'
                else:
                    bal_str = 'nonbalanced_noniid'
                data_path = os.path.join(self.root_dir, 'real_data', self.data_kind, 'data_' + bal_str + '_#' + str(num_of_samples) + '_seed' + str(seed) + '.csv')
                label_path = os.path.join(self.root_dir, 'real_data', self.data_kind, 'label_' + bal_str + '_#' + str(num_of_samples) + '_seed' + str(seed) + '.csv')
                par_path = os.path.join(self.root_dir, 'real_data', self.data_kind, 'par' + str(num_of_pars) \
                        + '_' + bal_str + '_#' + str(num_of_samples) + '_seed' + str(seed) + '.csv')
                print (par_path)
            elif self.data_name.startswith('tcga'):
                data_kind, data_num = self.__get_data_kind_and_num(self.data_name)
                self.data_kind = 'tcga'
                data_path = os.path.join(self.root_dir, 'real_data', self.data_kind, self.data_name + '_' + str(self.num_of_features) \
                        + 'x' + str(self.num_of_samples) + '_K' + str(self.num_of_cls) + '_seed' + str(seed) + '.csv')
                label_path = os.path.join(self.root_dir, 'real_data', self.data_kind, self.data_name + '_' + str(self.num_of_features) \
                        + 'x' + str(self.num_of_samples) + '_K' + str(self.num_of_cls) + '_seed' + str(seed) + '_label.csv')
                #par_path = os.path.join(self.root_dir, 'real_data', self.data_kind, 'par' + str(num_of_pars) + '_seed' + str(seed) + '.csv')
            elif self.data_name.startswith('coil'):
                data_kind, data_num = self.__get_data_kind_and_num(self.data_name)
                self.data_kind = 'coil'
                data_path = os.path.join(self.root_dir, 'real_data', self.data_kind, self.data_name + '_' + str(self.num_of_features) \
                        + 'x' + str(self.num_of_samples) + '_K' + str(self.num_of_cls) + '_seed' + str(seed) + '.csv')
                label_path = os.path.join(self.root_dir, 'real_data', self.data_kind, self.data_name + '_' + str(self.num_of_features) \
                        + 'x' + str(self.num_of_samples) + '_K' + str(self.num_of_cls) + '_seed' + str(seed) + '_label.csv')
            elif self.data_name.startswith('ypmsd'):
                data_kind, data_num = self.__get_data_kind_and_num(self.data_name)
                self.data_kind = 'ypmsd'
                data_path = os.path.join(self.root_dir, 'real_data', self.data_kind, self.data_name + '_' + str(self.num_of_features) \
                        + 'x' + str(self.num_of_samples) + '_K' + str(self.num_of_cls) + '_seed' + str(seed) + '.csv')
                label_path = os.path.join(self.root_dir, 'real_data', self.data_kind, self.data_name + '_' + str(self.num_of_features) \
                        + 'x' + str(self.num_of_samples) + '_K' + str(self.num_of_cls) + '_seed' + str(seed) + '_label.csv')
            elif self.data_name.startswith('tdt'):
                data_kind, data_num = self.__get_data_kind_and_num(self.data_name)
                self.data_kind = 'tdt'
                data_path = os.path.join(self.root_dir, 'real_data', self.data_kind, self.data_name + '_' + str(self.num_of_features) \
                        + 'x' + str(self.num_of_samples) + '_K' + str(self.num_of_cls) + '_seed' + str(seed) + '.csv')
                label_path = os.path.join(self.root_dir, 'real_data', self.data_kind, self.data_name + '_' + str(self.num_of_features) \
                        + 'x' + str(self.num_of_samples) + '_K' + str(self.num_of_cls) + '_seed' + str(seed) + '_label.csv')
            else:
                print (self.data_name)
                raise ValueError('Other data kinds are not supported now!')
            
            
            print (os.path.exists(data_path))
            if os.path.exists(data_path): # the data file exists, just read it
                self.data_mat = self.read_data_from_csvfile(data_path)
                self.true_labels = self.read_data_from_csvfile(label_path)
                self.true_labels = self.true_labels[0, :] # since labels are stored as matrix, we just extrac row 0
                if self.data_kind == 'mnist':
                    self.par_idx = self.read_data_from_csvfile(par_path)
                    self.par_idx = self.par_idx[0, :] 
                self.existed = True
            else:
                orig_data_path = os.path.join(self.root_dir, 'real_data', self.data_kind, 'data' + str(data_num) + '.csv')
                orig_label_path = os.path.join(self.root_dir, 'real_data', self.data_kind, 'label' + str(data_num) + '.csv')
                data_mat = self.read_data_from_csvfile(orig_data_path)
                #self.data_mat = self.data_mat.transpose()[:, 0:20001] # just for testing
                labels = self.read_data_from_csvfile(orig_label_path)
                print (data_mat.shape)
                if self.data_kind == 'mnist':
                    labels = labels.transpose()[0, :]
                    self.data_mat, self.true_labels, self.par_idx = self.extract_mnist_subdata_and_partition(data_mat, labels, \
                            num_of_samples, num_of_pars, is_balanced, 0)
                elif self.data_kind == 'coil':
                    self.data_mat = data_mat
                    self.true_labels = labels.transpose()[0, :]
                    self.par_idx = None
                else:    
                    self.data_mat = data_mat
                    self.true_labels = labels[0, :]
                    self.par_idx = None
                    
                f_manager = FileManager(self.root_dir)
                f_manager.add_file(data_path)
                np.savetxt(data_path, np.asmatrix(self.data_mat), delimiter = ',')
                f_manager.add_file(label_path)
                np.savetxt(label_path, np.asmatrix(self.true_labels), delimiter = ',')
                if not self.par_idx is None:
                    f_manager.add_file(par_path)
                    np.savetxt(par_path, np.asmatrix(self.par_idx), delimiter = ',')
                self.existed = False

            print ('reading data complete!----')
            #print (self.true_labels[0:10])
        else:
            print ('seed: ' + str(seed))
            np.random.seed(seed) # set the seed
            # at first, we check whether the data file has been generated or not
            data_path = os.path.join(self.root_dir, 'synthetic_data', self.data_name + \
                    '_' + str(self.num_of_features) + 'x' + str(self.num_of_samples) + '_K' + str(self.num_of_cls) + '_seed' + str(seed) + '.csv')
            label_path = os.path.join(self.root_dir, 'synthetic_data', self.data_name +
                    '_' + str(self.num_of_features) + 'x' + str(self.num_of_samples) + '_K' + str(self.num_of_cls) + '_seed' + str(seed) + '_label.csv')
            if os.path.exists(data_path): # the data file exists, just read it
                self.data_mat = self.read_data_from_csvfile(data_path)
                self.true_labels = self.read_data_from_csvfile(label_path)
                self.true_labels = self.true_labels[0, :] # since labels are stored as matrix, we just extrac row 0
                self.existed = True
            else:
                if data_name.startswith('snr'): # we should generate synthetic data with the linear model
                    # process the data name to get SNR
                    data_kind, data_num = self.__get_data_kind_and_num(data_name)
                    self.data_mat, self.true_labels = self.gen_data_by_linearModel(num_of_features, num_of_samples, self.num_of_cls, data_num)
                else:
                    raise ValueError('Error: other ways to generate synthetic datasets is not avaliable!')

                # save the newly generated data set so that we don't need to regenerate it again
                print (self.root_dir)
                f_manager = FileManager(self.root_dir)
                f_manager.add_file(data_path)
                np.savetxt(data_path, np.asmatrix(self.data_mat), delimiter = ',')
                f_manager.add_file(label_path)
                np.savetxt(label_path, np.asmatrix(self.true_labels), delimiter = ',')
                self.existed = False

        if self.num_of_cls != len(np.unique(self.true_labels)):
            print (self.num_of_cls)
            print (len(np.unique(self.true_labels)))
            print (self.true_labels)
            raise ValueError('Error: the specified cluster number is inconsistent with that in labels!')
       
    def __get_data_kind_and_num(self, data_name):
        ''' The function is used to process input data name string to return the data kind and data number

        Args:
            data_name: a string following the form of "XXX123" in which "XXX" indicates the data kind (e.g.
                        tdt2, mnist) and "123" indicates the data number
        Returns
            a pair of data kind and data number (e.g. ("XXX", "123")
        '''
        match = re.match(r"([a-z]+)(-?[0-9]+)", data_name, re.I)
        if match:
            items = match.groups()
        else:
            raise ValueError('Error, the data name does not follow the form XXX12!')
        return items[0], int(items[1])

    def gen_inits_WH(self, init = 'random', seed = 1):
        ''' The function is to initialize the factors W, H for nonnegative matrix factorization
        There are some options:
            1. random ------  generate W, H randomly
            2. kmeans ------  generate H based on cluster assignments obtained by Kmeans
                            then W = data_mat * H (since H is orthogonal)
            3. nmf    ------  use sklearn.nmf on data matrix firstly to get W, H for initialization
            4. kmeans++ ----  use heuristic strategy kmeans++ to get cluster assignment
                                    which can be used for H and W = data_mat * H

        Args:
            data (numpy array or mat): the input data
            init (string): the name of method used for generating the initializations
            rank (int): the rank for decomposition
            seed (float): the seed for random generator
        Returns:
            numpy matrix W and H
        '''

        initW_path = os.path.join(self.root_dir, 'inits', self.data_name, 'W' + str(seed) + '.csv')
        initH_path = os.path.join(self.root_dir, 'inits', self.data_name, 'H' + str(seed) + '.csv')
        np.random.seed(seed)
        if os.path.exists(initW_path) and os.path.exists(initW_path):
            W_init = self.read_data_from_csvfile(initW_path)
            H_init = self.read_data_from_csvfile(initH_path)
        else:
            (m, n) = self.data_mat.shape
            if init == 'random':
                avg = np.sqrt(abs(self.data_mat).mean() / self.num_of_cls)
                W_init = np.asmatrix(avg * np.random.random((m, self.num_of_cls)))
                H_init = np.asmatrix(avg * np.random.random((n, self.num_of_cls)))
                H_init = H_init.transpose()
                f_manager = FileManager(self.root_dir)
                f_manager.add_file(initW_path)
                np.savetxt(initW_path, np.asmatrix(W_init), delimiter = ',')
                f_manager.add_file(initH_path)
                np.savetxt(initH_path, np.asmatrix(H_init), delimiter = ',')
            else:
                raise ValueError('Error: other methods are not supported!')
        return np.asmatrix(W_init), np.asmatrix(H_init)
            
    def gen_data_by_linearModel(self, num_of_features, num_of_samples, num_of_cls, SNR = 1):
        '''The function aims to generate the synthetic data following the model
                X = W * H + E
        where:
            columns of W are generated by uniform distribution in [0, 1]
            H is set to be [I, I, ..., I]  then performn row normalization
            E is drawn iid from standard guassian distribution
        so that X is obtained by W * H + E and then repeating the following step
            X = [X]_{+}
            E = X - W * H
            E = gamma * E
            X = W * H + E
        where is a scaling constant determined by the desired SNR, and {+} takes the nonnegative part
        of its argument. We may need to repeat the above steps several times, till we get a nonnegative
        H with desired SNR (SNR is defined by ||WH||_{F}^{2} / ||E||_{F}^{2})

        Detailed explanation can refer to the paper Learning From Hidden Traits: Joint Factor Analysis
        and Latent Clustering

        Args:
            num_of_features (int): the number of features of the generated dataset
            num_of_samples (int): the number of samples of the generated dataset
            num_of_cls (int): the number of clusters for the generated dataset
            SNR (int): signal to noise ratio
        Returns:
            the dataset generated and its labels
        '''
        # set the range of elements in basis vector
        low_val = 0.0
        high_val = 1.0

        # randomly generate the cluster centroids
        W = np.random.uniform(low_val, high_val, size = (num_of_features, num_of_cls))

        #print ('data generated:')
        #print (W[0:10, 0:5])
        # initialize the cluster assignments
        H = np.zeros((num_of_cls, num_of_samples))

        # randomly assign the size of each cluster
        #cls_sizes = np.random.randint(np.maximum(1, 0.01 * num_of_samples / num_of_cls), 1 * num_of_samples / num_of_cls, 0.5 * num_of_cls).tolist()
        cls_sizes = []
        res_count = num_of_samples
        for i in range(num_of_cls):
            temp = np.random.randint(np.maximum(1, 0.01 * res_count / (num_of_cls - i)), 1.5 * res_count / (num_of_cls - i), 1)[0]
            cls_sizes.append(temp)
            res_count -= temp
        cls_sizes.append(res_count)
        '''    
        cls_size1 = np.random.randint(np.maximum(1, 0.1 * num_of_samples / num_of_cls), 2 * num_of_samples / num_of_cls, int(0.4 * num_of_cls)).tolist()

        cls_size2 = np.random.randint(np.maximum(1, 1 * num_of_samples / num_of_cls), 1.8 * num_of_samples / num_of_cls, int(0.3 * num_of_cls)).tolist()
        cls_size3 = np.random.randint(np.maximum(1, 1.8 * num_of_sample / num_of_cls), 2.5 * num_of_samples / num_of_cls, int
        cls_sizes = cls_size1 + cls_size2
        '''
        #res_num = num_of_samples - np.sum(cls_sizes)
        #las = int(res_num / 2)
        #cls_sizes.append(las)
        #cls_sizes.append(res_num - las)
        #cls_sizes.append(res_num)
        
        print ('The size of each cluster: ' + str(cls_sizes))

        # for each cluster, set the columns of H accoridng to its size
        start_pos = 0
        for i in range(num_of_cls):
            end_pos = int(start_pos + cls_sizes[i])
            H[i, start_pos:end_pos] = np.ones(int(cls_sizes[i]))
            start_pos = end_pos

        # construct the gaussian error for each cluster
        E = np.zeros((num_of_samples, num_of_features))
        # cov_list = []  # a list of covariance matrix
        mean = np.zeros(num_of_features)   # an array of mean values for each cluster
        start_row = 0
        cov_arr = np.eye(num_of_features)
        for i in range(num_of_cls):
            end_row = start_row + int(cls_sizes[i])
            #cov_list.append(np.eye(num_of_features)) # set covariance matrix to delta * I
            # draw samples based on multivariate guassian distribution with mean 0 and covariance matrix I
            #print ('start to generate data')
            dat = np.random.multivariate_normal(mean, cov_arr, cls_sizes[i])
            #print (dat[0:9, 0:9])
            E[start_row:end_row, :] = np.asarray(dat) # copy data to E
            start_row = end_row # update the start position for next dat
        E = E.transpose()

        # then, we reorder H and error
        arr_idx = np.random.choice(num_of_samples, num_of_samples, replace = False)
        H = H[:, arr_idx]
        E = E[:, arr_idx]
        H = np.asmatrix(H)

        # the true labels can be obtained by the max index in columns
        true_labels = np.argmax(np.asarray(H), axis = 0)

        # here, we try to control the scale of errors using the SNR (signal to noise ratio)
        print ('SNR: ' + str(SNR))
        ratio = 10 **(SNR / 10)
        while(True):
            X = W * H + E
            X = np.maximum(0, X)
            E = X - W * H
            test_SNR = LA.norm(W * H, 'fro') ** 2 / LA.norm(E, 'fro') ** 2
            print ('test_SNR :' + str(test_SNR))
            print ('ratio:' + str(ratio))
            print ('SNR - ratio : ' + str(abs(test_SNR - ratio)))
            if abs(test_SNR - ratio) > 1e-10:
                gamma = np.sqrt(test_SNR / ratio)
                E = E * gamma
            else:
                break
        print ('**SNB**: ' + str(LA.norm(W * H, 'fro') ** 2 / LA.norm(X - W * H, 'fro') ** 2))

        # to make the dataset difficult to cluster, we add some outliers by scaling randomly selected points
        #num_of_outliers = int(0.05 * num_of_samples) # %5 of data samples of outliers
        num_of_outliers = int(0.03 * num_of_samples) # %3 of data samples of outliers
        idx_list = np.random.randint(0, num_of_samples, size = num_of_outliers) # randomly select data samples as outliers
        print ('where to insert outliers? ' + str(idx_list))
        X = np.asarray(X)
        # set the values of outliers by scaling randomly generated vectors
        ol = np.random.uniform(size = (num_of_features, num_of_outliers))
        for j in range(num_of_outliers):
            X[:, idx_list[j]] = ol[:, j] * 5

        if len(true_labels) != num_of_samples:
            raise ValueError('Error: the length of data labels is not correct!')
        return np.asmatrix(X), true_labels


    def read_data_from_csvfile(self, f_path):
        """ This function is used to read data from a .csv file and
	return an array-like structure

        Args:
	    f_path (string): the absolute path of the .csv file

        Returns:
	    data_arr: data array (numpy array)

        Raises:
	    ValueError: if the file does not exist
        """
        if not os.path.exists(f_path):
            raise ValueError("Error: cannot find file: " + f_path)

        print (f_path)
        # read csv file into a pandas dataframe
        df = pd.read_csv(f_path, header = None)

        # convet the dataframe into a numpy mat
        data_arr = df.as_matrix()

        print (data_arr.shape)

        return data_arr

    def data_split(self, data, num_split):
        """ This function aims to split data into a number of partitions

        Args: 
            data (a 1-D array or list): the data to be split
            num_split (int): number of partitions
        Returns:
            a list with elements being list (e.g. [[1, 4, 5], [2, 3], [7, 8]])
        """
        # get the size of each split and the rest after the split
        delta, r = len(data) // num_split, len(data) % num_split
        data_lst = [] # the list to store the result
        i, used_r = 0, 0
        while i < len(data): # for each data
            if used_r < r: # we will put the rest evenly into the first partitions
                data_lst.append(data[i:i+delta+1])
                i += delta + 1
                used_r += 1
            else:
                data_lst.append(data[i:i+delta])
                i += delta
        return data_lst

    def choose_digits(self, split_data_lst, num):
        """ This function aims to randomly choose two list of data index and return

        Args:
            split_data_list (list): a list of elements being a list of indexes
        Returns:
            a list
        """
        available_digit = [];
        for i, digit in enumerate(split_data_lst):
            if len(digit) > 0:
                available_digit.append(i)
        if len(available_digit) == 0:
            return -1
        elif len(available_digit) < num: # in case that only one digit has splits
            na_splits = 0
            for k in range(len(available_digit)):
                d = available_digit[k]
                na_splits += len(split_data_lst[d])
            if na_splits >= num:
                for k in range(len(available_digit)):
                    d = available_digit[k]
                    for j in range(len(split_data_lst[d]) - 1):
                        available_digit.append(d)
            else:
                return -1
            '''
            i = available_digit[0]
            if len(split_data_lst[i]) >= 2: 
                available_digit.append(i)
            else: return -1
            '''
        try:
            lst = np.random.choice(available_digit, num, replace = False).tolist()
        except:
            print (split_data_lst)
            print (available_digit)

        return lst

    def extract_mnist_subdata_and_partition(self, data_mat, labels, num_data, num_of_pars, is_balanced, seed = 0):
        """ This function aims to extract a sub-data matrix from the original one.
        Specifically, the extraction is performed according to the labels of data
        samples. 

        Args:
            data_mat (numpy array): rows are features and columns are data samples
            labels (numpy array): labels of data samples
            num_data (int): the number data samples to be extracted
            num_of_pars (int): the number of partitions
            is_balanced (boolean): a flag to indicate whether the size of each class
                                    is the same or not.
            seed (float): the seed for randomness
        Returns:
            a numpy array with elements being index of data samples extracted.
        """

        num_of_cls = len(np.unique(labels))
        size_of_cls = np.ones(num_of_cls).astype(int)
        if is_balanced:
            if num_data % num_of_cls != 0:
                raise ValueError('Error: the number to be extract is not multiple of ' + str(num_of_cls))
            esize = int(num_data / num_of_cls)
            print (esize)
            size_of_cls = size_of_cls * esize
            print (size_of_cls)
        else:
            if num_of_cls == 10:  # 10 classes
                size_of_cls = (np.array([0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.15, 0.15, 0.2]) * num_data).astype(int)
            else:
                raise ValueError('Error: the class number is not 10!')

        # put index of data samples with the same class to a list
        cls_data = []
        split_cls_data = []
        print (size_of_cls)
        for cls_num in range(num_of_cls):
            # find the index of data samples with class cls_num
            idx = np.where(labels == cls_num)[0]
            idx = idx[:size_of_cls[cls_num]] # extract some data
            cls_data.append(idx)
            # split the idx into 20 even partitions
            split_cls_data.append(self.data_split(idx, 20))

        print(">>> Data is non-i.i.d. distributed")
        if is_balanced:
            print(">>> Data is balanced")
        else:
            print(">>> Data is not balanced")

        par_idx = np.ones(len(labels)) * -1
        np.random.seed(seed)
        # partition extract data
        for par_id in range(num_of_pars):
            # assign par_id to index of data samples to that partition
            for d in self.choose_two_digit(split_cls_data):
                l = len(split_cls_data[d][-1])
                idx = split_cls_data[d].pop().tolist()
                par_idx[idx] = par_id

        if data_mat.shape[0] > data_mat.shape[1]:
            data_mat = data_mat.transpose()
        print (np.where(par_idx == -1))

        data_mat = data_mat[:, par_idx != -1]
        labels = labels[par_idx != -1]
        par_idx = par_idx[par_idx != -1]
        print (par_idx[0:100])
        print (len(par_idx))

        if len(par_idx) != np.sum(size_of_cls):
            raise ValueError('Error: the number of data samples is wrong!')

        return data_mat, labels, par_idx
        

    def partition_data_array(self, num_of_pars = 1, m_name = 'uniform', seed = 0):
        """ This function is used to partion a data array in a column-wise fashion

        The partition is performed based on the following methods:
            1. "uniform":
		Each data point is assigned to one of the partitions with equal probability
	    2. "similarity":
		Each partition has an associted data point randomly selected from the data
		array. Then, each data point is assigned to one of the partitions with
		probability proportional to its similarity to the associated point of each
		partition, where the similarity is computed by the Euclidean distance
	    3. "weighted":
		Each partition is assigned a weight chosen by |N(0, 1)| and then each data
		point is assigned to one of the partions with probability proportional to
		its weight

        Args:
	    data_arr (numpy array):  the data array to be partitioned
	    num (int): the number of partitions
            m_name (string): the method used to perform partition
            seed (float): the seed for partition

        Returns:
	    None

        Raises:
	    ValueError: if the data_arr is none or num < 1
        """
        if self.data_mat is None or num_of_pars < 1:
            raise ValueError("Error: the input data array is none or the number of partitions < 1")
        np.random.seed(seed)

        print ('before partition: ' + str(self.data_mat.shape))
        # we aims to put the csv file of partition result under the same dir as that of
        # data file and label file. So, we get the corresponding path for partition result
        # using the same strategy
        #data_kind, data_num = self.__get_data_kind_and_num(self.data_name)
        if self.is_real:
            pars_path = os.path.join(self.root_dir, 'real_data', self.data_kind, self.data_name + \
                    '_' + m_name + '_#' + str(num_of_pars) + '_seed'+ str(seed) + '_pars.csv')
        else:
            pars_path = os.path.join(self.root_dir, 'synthetic_data', self.data_name + \
                    '_' + str(self.num_of_features) + 'x' + str(self.num_of_samples) + \
                    '_' + m_name + '_#' + str(num_of_pars) + '_seed' + str(seed)+ '_pars.csv')

        print ('---------read ars-------------')
        print (pars_path)

        if os.path.exists(pars_path): # if the result is already stored, just read it
            self.par_idx = self.read_data_from_csvfile(pars_path)
            self.par_idx = self.par_idx[0, :] # since par_idx is stored as matrix, we just extract row 0
            self.num_of_pars = len(np.unique(self.par_idx))
            print (np.unique(self.par_idx))
            print (num_of_pars)
            print (self.num_of_pars)
            if self.num_of_pars != num_of_pars:
                raise ValueError('Error: the number of partitions is not consistent with last try!')
            self.existed = True
        else:
            self.num_of_pars = num_of_pars # set the number of partitions

            if 'noniid' in m_name:
                m_str, cls_cover = m_name.split('#')[0], int(m_name.split('#')[1])
                if cls_cover > self.num_of_cls:
                    raise ValueError('Error: each cluster covers too much digits!!!')
                if m_str == 'bal_noniid':
                    self.par_idx = self.balanced_noniid_partitioning(self.true_labels, self.num_of_pars, cls_cover)
                elif m_str == 'ubal_noniid':
                    self.par_idx = self.unbalanced_noniid_partitioning(self.true_labels, self.num_of_pars, cls_cover)
                else:
                    raise ValueError(m_name + ' does not exist!!!')
            elif m_name == "uniform":
                self.par_idx = self.uniform_partitioning()
            elif m_name == "similarity":
                self.par_idx = self.similarity_partitioning()
            elif m_name == "sc":
                self.par_idx = self.sc_partitioning()
            elif m_name == 'descan':
                self.par_idx = self.density_partitioning()
            elif m_name == "weighted":
                self.par_idx = self.weighted_partitioning()
            elif m_name == "cluster":
                self.par_idx = self.cluster_partitioning()
            else:
                raise ValueError('Error: the method ' + str(m_name) + ' is not supported!')

            # print out the size of each partition
            # print (np.bincount(idx_arr.astype(int)))

            # save the partition result so that we can read it directly
            f_manager = FileManager(self.root_dir)
            f_manager.add_file(pars_path)
            np.savetxt(pars_path, np.asmatrix(self.par_idx), delimiter = ',')
            self.existed = False


    def uniform_partitioning(self):
        """ This function aims to split the data points (columns) of data_arr into a number of
        partitions with uniform distribution

        Args:
	    None
        Returns:
	    idx_arr (numpy array)
        """
        if self.data_name.startswith('tcga'):
            np.random.seed(20)

        N = self.data_mat.shape[1]    # the number of data samples
        size_of_pi = math.floor(N / self.num_of_pars) # the size of each partition except the last one
        idx_arr = np.ones(N) * -1     # initialize the partition index to be -1
        data_idx = range(N)
        for i in range(self.num_of_pars - 1):   # for each partition
            temp = np.random.choice(data_idx, size_of_pi, replace = False)   # randomly select size_of_pi samples to partition i
            idx_arr[temp] = i
            data_idx = [elem for elem in data_idx if not elem in temp]   # remove indexes of selected samples from data_idx
        idx_arr[data_idx] = self.num_of_pars - 1  # assign the rest of data samples to the last partition
        return idx_arr

    def sc_partitioning(self):
        """ This funtion aims to split the data points (columns) of data arr into a number of 
        partitions by clustering them using equal sized k-means
        """
        #equal_KM = EqualGroupsKMeans(n_clusters = self.num_of_pars, random_state = 0).fit(self.data_mat.transpose())
        sc = SpectralClustering(n_clusters = self.num_of_pars, random_state = 0).fit(self.data_mat.transpose())
        return sc.labels_

    def density_partitioning(self):
        descan = DBSCAN().fit(self.data_mat.transpose())
        return descan.labels_

    def balanced_noniid_partitioning(self, labels, num_of_pars, cls_cover = 2, seed = 0):
        """ This function aims to split the data points (columns) of data arr into a number of 
        partitions, in which each partition contains equal sized points of two random clusters

        Args:
            labels (1-D numpy array): the true cluster labels of the data matrix
            num_of_pars (int) : the number of partitions
            cls_cover (int): the number of clusters data points in one partition belong to
            seed (float): the seed for randomness
        """
        # compute the size of each split
        data_num = self.data_mat.shape[1]
        size_of_split = int(data_num / (cls_cover * num_of_pars)) - 1
        if size_of_split < 10: 
            raise ValueError('Error: the split is less than 10!!!')
        # put index of data samples with the same class to a list
        cls_data = []
        split_cls_data = []
        res_data = [] 
        cls_idx_list = np.unique(labels)
        for cls_num in cls_idx_list:
            # find the index of data samples with class cls_num
            idx = np.where(labels == cls_num)[0]
            cls_data.append(idx)

            # split the idx into 20 even partitions
            num_of_splits, r = len(idx) // size_of_split, len(idx) % size_of_split
            '''
            print ('cluster num : ' + str(cls_num))
            print (idx)
            print ('num_of_splits : ' + str(num_of_splits))
            print ('size_of_splits : ' + str(size_of_split))
            '''
            if num_of_splits < 1:
                num_of_splits += 1
            elif r / num_of_splits > 1:
                num_of_splits += 1
            else:
                pass
            splits = self.data_split(idx, num_of_splits)
            #print (splits)
            split_cls_data.append(splits)

        #print ('length of split cls data: ' + str(len(split_cls_data)))
        c_num = 0
        for j in range(len(split_cls_data)):
            c_num += len(split_cls_data[j])
        print ('num of splits: ' + str(c_num))

        if c_num < cls_cover * num_of_pars:
            raise ValueError('Error: the splits are not enough!!!')

        par_idx = np.ones(len(labels)) * -1
        np.random.seed(seed)
        par_counts = {}
        # partition extract data
        for par_id in range(num_of_pars):
            # assign par_id to index of data samples to that partition
            #print ('choosing splits for partition ' + str(par_id))
            digits = self.choose_digits(split_cls_data, cls_cover)
            if digits == -1: pass
            else:
                for d in digits:
                    l = len(split_cls_data[d][-1])
                    idx = split_cls_data[d].pop().tolist()
                    par_idx[idx] = par_id
        for par_id in range(num_of_pars):
            #par_num = len(np.where(par_idx == par_id))
            unique, counts = np.unique(par_idx, return_counts = True)
            dt = dict(zip(unique, counts))
            if not par_id in dt.keys():
                par_counts[par_id] = 0
            else:
                par_num = dt[par_id]
                if par_num <= cls_cover * size_of_split + 2:
                    par_counts[par_id] = par_num

        print (par_counts)

        print (np.unique(par_idx, return_counts = True))

        print ('-----------------------')
        print (np.where(par_idx == 0))

        if (par_idx == -1).any(): # some data points are not partitioned
            #print ('num of data points : ' + str(len(np.where(par_idx == -1))))
            #print (np.where(par_idx == -1))
            #print (split_cls_data)
            for i in range(len(split_cls_data)):
                for j in range(len(split_cls_data[i])):
                    #l = len(split_cls_data[i][j])
                    #print (split_cls_data[i][j])
                    if len(split_cls_data[i][j]) > 0:
                        tmp = split_cls_data[i][j]
                        while True:
                            print (tmp)
                            l = len(tmp)
                            ky = min(par_counts, key=par_counts.get)
                            old_count = par_counts[ky]
                            print ('min: ' + str(ky) + ' counts ' + str(old_count) + ' l:' + str(l))
                            if old_count + l <= cls_cover * size_of_split + 2:
                                par_counts[ky] += l
                                #end_idx = l
                                par_idx[tmp] = ky
                                #print (np.where(par_idx == 0))
                                break
                            else:
                                if l <= 2:
                                    par_idx[tmp] = ky
                                    par_counts[ky] += l
                                    tmp = []
                                else:
                                    par_idx[tmp[0:cls_cover * size_of_split + 2 - old_count]] = ky
                                    tmp = tmp[cls_cover * size_of_split + 2 - old_count:]
                                    print ('par_id : ' + str(ky) + ' num:' + str(np.where(par_idx == ky)) + ', need: ' + str(cls_cover * size_of_split))
                                    par_counts[ky] += cls_cover * size_of_split + 1 - old_count
                            if par_counts[ky] > cls_cover * size_of_split + 5:
                                del par_counts[ky]
                            if len(tmp) <= 0: break
                            #par_idx[tmp] = ky
            if (par_idx == -1).any():
                print (np.where(par_idx == -1))
                raise ValueError('Error: some data points are not partitioned!!!')
        
        return par_idx


    def unbalanced_noniid_partitioning(self, labels, num_of_pars, cls_cover = 2, seed = 0):
        """ This function aims to split the data points (columns) of data arr into a number of 
        partitions, in which each partition contains unevenly sized points of two random clusters

        Args:
            labels (1-D numpy array): the true cluster labels of the data matrix
            num_of_pars (int) : the number of partitions
            cls_cover (int): the number of clusters data points in one partition belong to
            seed (float): the seed for randomness
        """
        
        cls_vals, counts = np.unique(labels, return_counts = True)
        if np.std(counts) < 10:
            raise ValueError('Error: the size of each cluster is approximately the same !!!')
        num_of_cls = len(cls_vals)
        n_splits_cls, r = cls_cover * num_of_pars // num_of_cls, cls_cover * num_of_pars % num_of_cls
        if r > 0: n_splits_cls += 1
            #raise ValueError('Error: the rest is not 0!!!')
        if np.min(counts) <= 2 * n_splits_cls:
            raise ValueError('Error: one cluster cannot be splited well !')

        # put index of data samples with the same class to a list
        cls_data = []
        split_cls_data = []
        #print (size_of_cls)
        for cls_num in cls_vals:
            # find the index of data samples with class cls_num
            idx = np.where(labels == cls_num)[0]
            # split the idx into 20 even partitions
            #size_of_split, r = len(idx) // num_splits_cls, len(idx) % num_splits_cls
            cls_data.append(idx)
            # split the idx into 20 even partitions
            split_cls_data.append(self.data_split(idx, n_splits_cls))

        c_num = 0
        for j in range(len(split_cls_data)):
            c_num += len(split_cls_data[j])
        print ('num of splits: ' + str(c_num))

        par_idx = np.ones(len(labels)) * -1
        np.random.seed(seed)
        par_counts = {}
        # partition extract data
        for par_id in range(num_of_pars):
            # assign par_id to index of data samples to that partition
            #print ('choosing splits for partition ' + str(par_id))
            digits = self.choose_digits(split_cls_data, cls_cover)
            if digits == -1: pass
            else:
                for d in digits:
                    l = len(split_cls_data[d][-1])
                    idx = split_cls_data[d].pop().tolist()
                    par_idx[idx] = par_id

        for par_id in range(num_of_pars):
            #par_num = len(np.where(par_idx == par_id))
            unique, counts = np.unique(par_idx, return_counts = True)
            dt = dict(zip(unique, counts))
            par_num = dt[par_id]
            if par_num <= 5:
                par_counts[par_id] = par_num
        
        
        if (par_idx == -1).any(): # some data points are not partitioned
            for i in range(len(split_cls_data)):
                for j in range(len(split_cls_data[i])):
                    #l = len(split_cls_data[i][j])
                    #print (split_cls_data[i][j])
                    if len(split_cls_data[i][j]) > 0:
                        tmp = split_cls_data[i][j]
                        if len(par_counts) <= 0:
                            ky = np.random.choice(num_of_cls, 1, replace = False)
                        else: 
                            ky = min(par_counts, key=par_counts.get)
                            del par_counts[ky]
                        par_idx[tmp] = ky
            if (par_idx == -1).any():
                print (np.where(par_idx == -1))
                raise ValueError('Error: some data points are not partitioned!!!')


        return par_idx

    def similarity_partitioning1(self):
        """ This function aims to split the data points (columns) of data arr into a number of 
        partitions by clustering them using k-means

        Args:
            None

        Returns:
            idx_arr(numpy array)
        """
        if self.data_name.startswith('tcga'):
            kmeans = KMeans(n_clusters = self.num_of_pars, init = 'k-means++', n_init = 20, random_state = 0).fit(self.data_mat.transpose())
        else:
            kmeans = KMeans(n_clusters = self.num_of_pars, random_state = 0).fit(self.data_mat.transpose())
        return kmeans.labels_

    def similarity_partitioning(self):
        """ This function aims to split the data points (columns) of data_arr into a number of
        partitions based on similarity to their associated points

        Args:
	    None
        Returns:
	    idx_arr (numpy array)
        """
        N = self.data_mat.shape[1]   # the number of data samples
        data_idx = range(N)
        dat_of_pars = np.random.choice(data_idx, self.num_of_pars, replace = False)   # each partition has an associated point randomly sampled from data_irr
        dat_of_pars = np.sort(dat_of_pars)  # sort it


        #print ('selected points for each partition: ')
        #print (dat_of_pars)


        # calculate the affinity matrix in which the similarity of a pair of data samples
        # is calculated with Guassian Kernal (gamma = 1)
        #total_aff_mat = pairwise_kernels(self.data_mat.transpose(), self.data_mat.transpose(), metric = 'rbf', gamma = 1.0)
        data_mat = self.data_mat.transpose() / LA.norm(self.data_mat, 'fro') # scale the data at first in case the similarity = 0
        #total_aff_mat = rbf_kernel(data_mat, data_mat, gamma = 1.0)
        total_aff_mat = pairwise_kernels(data_mat, data_mat[dat_of_pars, :], metric = 'rbf')

        #print (total_aff_mat[0, 0:100])
        #print (data_mat.shape)
        #print (np.dot(data_mat[0,:].T, data_mat[1, :]) / (LA.norm(data_mat[0,:]) * LA.norm(data_mat[1,:])))

        #aff_mat = total_aff_mat[:, dat_of_pars]   # only cares about the similarity with the partitions' associated points
        aff_mat = total_aff_mat
        #print (aff_mat[0:3, :])
        # calculate the probabilities that each point should be assigned to these partitions
        row_sum = aff_mat.sum(axis = 1)   # sum up each row

        #print (row_sum[0:3])
        prob_mat = aff_mat.astype(float) / row_sum.reshape(N, 1)  # normalized each row of aff_mat such that the sum of each row = 1
        #print (prob_mat.shape)
        #print (prob_mat[0, :])

        #print (np.argmax(prob_mat, axis = 1))

        idx_arr = np.argmax(prob_mat, axis = 1)


        '''
        idx_arr = np.ones(N) * -1  # initialize the partition index to be -1
        for i in range(N): # for each data point
            if i not in dat_of_pars:  # except the assoicated points
                idx_arr[i] = np.random.choice(self.num_of_pars, replace = False, p = prob_mat[i, :]) # sample based on probability
                if i == 0:
                    print (prob_mat[i, :])
                    print (idx_arr[i])
            else:
                idx_arr[i] = np.where(dat_of_pars == i)[0]
        '''
        return idx_arr

    def cluster_partitioning(self):
        """ This function aims to split the data points (columns) of data_arr into a number of
        partitions based on their cluster labels. Specficially, we will assign each cluster of
        data points to a partition. Note that the number of partitions be less than or equal to the
        number of clusters in the data. If the number of partitions is less than the number of
        clusters, the data points in each cluster (rest) will be assinged a random partition

        Args:
            None
        Returns:
            idx_arr (numpy array)
        """
        N = self.data_mat.shape[1]  # the number of data samples
        cls_labels = np.unique(self.true_labels) # get all cluster labels
        cls_labels = np.sort(cls_labels)
        idx_arr = np.ones(N) * -1
        if self.num_of_pars > len(cls_labels):
            raise ValueError('Error: the number of partitions should <= the number of clusters!!!')
        for j in range(len(cls_labels)):
            cls_id = cls_labels[j] # cluster id
            cls_data_idx = np.where(self.true_labels == cls_id)[0]  # get the index of data samples belong to a certain cluster
            if j < self.num_of_pars:
                idx_arr[cls_data_idx] = j
            else:
                # randomly choose a par_id
                par_id = np.random.choice(self.num_of_pars, 1)
                idx_arr[cls_data_idx] = par_id
        return idx_arr

    def weighted_partitioning(self):
        """ This function aims to split the data points (columns) of data_arr into a number of
        partitions based on the weight of each partition

        Args:
	    None
        Returns:
	    idx_arr (numpy array)
        """
        N = self.data_mat.shape[1]   # the number of data samples
        while (True):
            weights = abs(np.random.normal(0, 1, self.num_of_pars))   # draw random weights for each partition by normal distribution
            size_of_pars = (weights / np.sum(weights)) * N    # the size of each partition according to the weights
            size_of_pars = size_of_pars.astype(int)  # convert all sizes to integer
            if np.all(size_of_pars > 0.1 * N / self.num_of_pars):
                break
        data_idx = range(N)
        idx_arr = np.ones(N) * -1  # initialize the partition index to be -1
        for i in range(self.num_of_pars - 1):  # for each partition
            temp = np.random.choice(data_idx, size_of_pars[i])
            idx_arr[temp] = i
            data_idx = [elem for elem in data_idx if not elem in temp]   # remove indexes of selected samples from data_idx
        idx_arr[data_idx] = self.num_of_pars - 1  # assign the rest of data samples to the last partition
        print (np.unique(idx_arr, return_counts = True))
        return idx_arr

    def get_data_and_labels(self):
        ''' This function is used to return the generated data matrix and its associated labels

        Args:
            None
        Returns:
            a list of data matrix and labels
        '''
        return self.data_mat, self.true_labels

    def get_partitions(self):
        ''' This function is used to return the partition index
        Args:
            None
        Returns:
            a numpy array with element being the index of partitions
        '''
        return self.par_idx

    def existed_or_not(self):
        ''' This function returns the flag to see whether the dataset is the first time generated

        Args:
            None
        Returns:
            self.existed
        '''
        return self.existed

    def visualize_data(self, dat_embeded = None, dim = 2, partition_idx = None, dat_path = None, \
        xlabel = '', ylabel = '', title = '', data_points = None):
        """ This function is used to visulize the data points in a 2-D space. Specifically, it
        will plot these data points in a figure after dimension-reduction with T-SNE. The
        data points belonging to the same partition will be plotted with the same color.
        Args:
            data_arr (numpy array): the data array to be plotted with each column being a data point
            dim (int): the dimension on which the data points to be plotted, default: 2
            partition_idx (list): a list of indexes indicating the partition of these data points
            dat_path (string): an absolute file path to store the 2D figure
            xlabel (string): a string to be shown on the x-axis of the 2D figure
            ylabel (string): a string to be shown on the y-axis of the 2D figure
            title (string): a string as the title of the figure
        Returns:
            None
        """
        if dim != 2:
            raise ValueError('Error: only 2-D figures are allowed now!')

        if dat_embeded is None: # generate it using tsne on data_arr
            if self.data_mat.shape[0] > 2:
                print ('transform---------------')
                tsne = manifold.TSNE(n_components = dim, random_state = 0)
                print ('data_mat:'+ str(self.data_mat.shape))
                dat_embeded = tsne.fit_transform(self.data_mat.transpose())
            else:
                dat_embeded = np.asmatrix(np.copy(self.data_mat.transpose()))
            if partition_idx is None:
                print ('partiton idx is none, use true labels!')
                partition_idx = self.true_labels
            if dat_path is None:
                dat_folder = 'real_data' if self.is_real else 'synthetic_data'
                dr_str = 'DR' if self.dim_reduced else ''
                dat_path = os.path.join(self.root_dir, dat_folder, self.data_kind, 'data' + dr_str + '#' + str(self.data_num) + '.pdf')
                title = self.data_kind + '#' + str(self.data_num)

        colormap = np.array(['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', \
                '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', \
                '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', \
                '#000000'])
        
        print ('data_embeded: ' + str(dat_embeded.shape))
        #map cluster indexs to 0-20
        pars = np.asarray(map(int, partition_idx))
        print (pars)
        u_elms = np.unique(pars)
        ind = -1
        num = len(list(u_elms))
        if num > 20:
            raise ValueError('The number of pars is greater than 20!!')
        step = int(20 / num)
        p_list = []
        for i in range(num):
            if not u_elms[i] in p_list:
                ind = ind + step
                pars[pars == u_elms[i]] = ind
                p_list.append(u_elms[i])
                
        print (pars)
        if len(list(u_elms)) > self.num_of_cls or len(p_list) > self.num_of_cls:
            print (len(list(u_elms)), len(p_list), self.num_of_cls)
            raise ValueError('Error: the number of clusters are not consistent')
        
        color_used = colormap[list(pars)]
        dat_embeded = np.array(dat_embeded)
        print (dat_embeded[0:5, 0:2])
        print (dat_embeded.shape)
        plt.figure()
        plt.scatter(dat_embeded[:, 0], dat_embeded[:, 1], c = color_used)
        if not data_points is None:
            print (data_points.shape)
            plt.scatter(data_points[:, 0], data_points[:, 1], c = 'blue', s=40)
        #plt.ylim(0, 20)
	#plt.xlim(0, 20)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(dat_path, bbox_inches = 'tight')

if __name__ == "__main__":
    data_gen = DataGenerator(root_dir = '/home/ubuntu/work/data', is_real = False, data_name = 'snr-1')
    data_mat, labels = data_gen.get_data_and_labels()
    print (data_mat.shape)
    print (labels.shape)
    #data_gen.visualize_data(data_mat, dim = 2, partition_idx = labels.tolist())
    data_gen.partition_data_array(num_of_pars = 3, m_name = 'uniform')
    partitions = data_gen.get_partitions()
    print (partitions)
    print (labels)
    #data_gen.visualize_data(data_mat, dim = 2, partition_idx = labels.tolist())
