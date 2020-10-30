###########################################################################################
# The script gives an implementation of the paper "Distributed k-Means and k-Median clustering
# on general topologies" with the authros Marua Florina Balcan etc.
#
# The code is extracted from https://github.com/harshadrai/Distributed_kMeans and modified
# for a special communcation topology- star newtork.
#
# As a counterpart of distributed_coreset.py, the script just provides for a sequential version
# which indeed executes in a central fashion to simululate the process instead of a practical
# distributed setting.
#
# @author Wang Shuai
# @date 2019.07.19
###########################################################################################

from __future__ import division

import numpy as np
import pandas as pd
import math
from numpy import linalg as LA
# import two helper class
from sklearn.metrics.pairwise import euclidean_distances
from data_generator import DataGenerator as DataGen # manage the data generation
from sklearn.cluster import KMeans
from file_manager import FileManager
from cluster_metrics import *
import csv
import collections
import os

from cluster_tsne import *

class Node(object):
    ''' This class is designed for the local computations on local sites

    Attributes:
        data (numpy array): the local data stored on the local node with
                        rows being samples and columns being features
        centers : the centers of local data obtained by local approximation solution (kmeans)
        labels: the labels of local data by the local clustering
        local_coreset: the local coreset constructed (centers + sampling points)
        weights: the weight for each point in local coreset
        local_cost (float): the local cost of all local data points
        cost_of_each_data : the local cost of each local point
        num_of_cls (int): the number of clusters
        comm (object): the communicator which is responsible for communication
        #proc_id (int): the id of processor which the node is executed on
    '''
    def __init__(self, data, seed_num = 1):
        ''' __init__ function

        Args
            data (numpy array): the local data with rows being samples
        Returns:
            None
        '''
        #self.neighbors = neighbors
        #self.degree = degree
        self.data = data            # Holds the local data Pi
        self.size = self.data.shape[0]
        np.random.seed(seed_num)

    def nearest_neighbor(self, x, Si):
        ''' Find nearest_neighbor in Si

        Args:
            x - int, the point index
            Si - list of point indexes
        Returns:
            nearest neighbor, distance
        '''
        if x in Si:
            return x, 0
        else:
            Si_data = self.data[Si, ]
            dist = LA.norm(Si_data - self.data[x, ], axis = 1).tolist() # array of distances of all points in Si to x
            min_idx = np.argmin(dist)
            return Si[min_idx], dist[min_idx]


    def summary_outliers(self, k, t):
        ''' Aggregate local data points

        Args:
            k -int, the number of clusters
            t -int, the number of outliers
        Returns:
            A summary with centers and weights
        '''
        Xi = list(range(self.size))
        beta = 0.45
        alpha = 2
        kv = np.maximum(np.log(self.size) , k)
        S = []
        S_weights = []
        while True:
            if len(Xi) <= 8 * t: break 
            Si_try = np.random.choice(Xi, int(alpha * kv), replace = True).tolist()
            Si = []
            [Si.append(x) for x in Si_try if x not in Si] # remove duplicates in Si
            Si_weights = np.zeros(len(Si))  # the weights for each sample in Si
            dist_to_Si = []
            nn_in_Si = []
            for x in Xi: # for each point in Xi, calculate its disance to Si and nearest neighbor in Si
                c, d = self.nearest_neighbor(x, Si)
                nn_in_Si.append(c)
                dist_to_Si.append(d)
            sort_idx = np.argsort(np.array(dist_to_Si))  # sort index by ascending
            sort_dist = np.sort(np.array(dist_to_Si))
            # draw a ball centered at Si with size at least beta * |Xi|
            bound = int(np.ceil(beta * len(Xi)))
            Ci = []  # initialize the ball
            #print ('bound: ' + str(bound) + ' Si: ' + str(len(Si)))
            for i in range(bound):
                Ci.append(Xi[sort_idx[i]])
                # calculate the weights in Si
                idx = np.where(np.array(Si) == nn_in_Si[sort_idx[i]])
                Si_weights[idx] += 1
            #print (set(Si).issubset(set(Ci)))
            Xi = [e for e in Xi if e not in Ci]
            # combine Si to S
            S = S + Si
            S_weights = S_weights + Si_weights.tolist()
            
        return S, S_weights, Xi

    def aug_summary_outliers(self, k, t):
        ''' The augumented version of summary outliers


         Args:
            k -int, the number of clusters
            t -int, the number of outliers
         Returns:
            A summary with centers and weights
        '''
        S, S_weights, Xr = self.summary_outliers(k, t)
        if len(Xr) <= len(S) or len(S + Xr) >= self.size:
            print ('the number of outlier is ' + str(len(Xr)) + ', the number of local summary is ' + str(len(S)))
        else:
            num = len(Xr) - len(S)
            X =  list(range(self.size))
            Xn = [e for e in X if e not in (S + Xr)]
            S_prime = np.random.choice(Xn, num, replace = True).tolist()
            Xn1 = [e for e in X if e not in Xr]
            S_weights1 = np.zeros(len(S)) 
            for x in Xn1: # for each point in Xi, calculate its disance to Si and nearest neighbor in Si
                c, d = self.nearest_neighbor(x, S + S_prime)
                idx = np.where(np.array(S) == c)
                S_weights1[idx] += 1
            S_weights = S_weights1.tolist()
        S = S + Xr
        for j in range(len(Xr)):
            S_weights.append(1)
        return self.data[S, ], S_weights 
        


def gen_and_distribute_data(is_real = False, data_name = 'snr-1', data_dir = 'data', \
        num_of_features = 2000, num_of_samples = 1000, num_of_source = 10,
        par_method = 'uniform', num_of_pars = 5, seed = 0):
    ''' This function is used to generate the raw data for distributed computing.
    Specifically, the data is generated, partitioned, and then distributed among
    available local nodes.

    There are two kinds of datasets: synthetic datasets and real datasets. Only synthetic datasets
    are required to be generated once since it will be saved to file and can be directly obtained
    from the file next time. So, to guarantee that the datasets used by all nodes are the same, in
    the first time only the central site perform synthetic data generation and the synthetic datasets
    will be shared between all local nodes by filesystem. It will not be an issue for real datasets
    since all real datasets need to be prepared for all nodes in advance and are read from files.

    Args:
        is_real (boolean): a flag indicates which kinds of datasets to use, default: false
                        False: use synthetic datasets
                        True: use real-world datasets
        data_name (string): the name of the dataset required which provides info about the data
                        For synthetic datasets, it provides the method to generated data and
                        some parameters, e.g. 'snr-1' means to use linear model to generated data
                        and the SNR sets to be -1.
                        For real datasets, it provides the data name and ID, e.g. 'mnist1' means
                        the first dataset of mnist database.
        data_dir (string): the dir under which the data is stored
        num_of_features (int): used when is_real = False, optional, default: 2000.
        num_of_samples (int): used when is_real = False, optional, default: 1000.
        num_of_source (int): used when is_real = False, optional, default: 10
        par_method (string): the method used to partition the dataset for distributing
        num_of_pars (int): the number of partitions
        proc_id (int): the id of processors (node) who is running the method
        seed (int): the seed for random generator. Since the generation of synthetic datasets and
                    data partition will use random data generator, it is important to set the seed
                    for reproduction.
    Returns:
        par_idx (1-D numpy array): the index of partitions for each data sample indicating to which
                    partition it belongs
    Raises:
        ValueError: the synthetic dataset is expected to be generated already and has been shared
                between all nodes. If not, an error occurs
    '''
    # generate the dataset and the corresponding labels
    data_gen = DataGen(root_dir = data_dir, is_real = is_real, data_name = data_name, \
            num_of_features = num_of_features, num_of_samples = num_of_samples, \
            num_of_cls = num_of_source)

    # partition the dataset
    if not data_name.startswith('mnist'):
        data_gen.partition_data_array(num_of_pars = num_of_pars, m_name = par_method) # partition the data

    if not data_gen.existed_or_not():
        raise ValueError('Error: the synthetic data is not generated before. So please ensure \
                the data just generated has been prepared for local nodes and then run again!')

    data_mat, true_labels = data_gen.get_data_and_labels()
    par_idx = data_gen.get_partitions() # get the partition index

    return data_mat, true_labels, par_idx



def grow_ball(data_mat = None, par_idx = None, num_of_cls = 10, num_of_locs = 3, t = 256, seed = 1):
    ''' 

    Args:
        data_mat (numpy array or mat): the input data with rows being features and columns being samples
        par_idx (numpy array): a 1-D array which indicates how the data is partitioned among local nodes
        true_labels (numpy array): a 1-D array indicates the true clustering assignments of samples
        num_of_cls (int): the number of clusters in the data
        num_of_locs (int): the number of local nodes
        t_sum (int): the size of the global coreset
        seed (float): the seed for random data generator for
                    1. the local approximation solution obtained by kmeans (initializations)
                    2. the sampling among local data
    Returns:
        the coreset and the associated weights
    '''
    # construct local node manager for local computation
    local_nodes = []
    global_coreset = None
    global_weights = []
    for node_id in range(num_of_locs):
        local_data_idx = np.where(par_idx == node_id)[0] # get the index of data samples belong to local site i
        local_data = data_mat[:, local_data_idx]   # get local data
        local_node = Node(local_data.transpose(), seed)
        (points, weights) = local_node.aug_summary_outliers(num_of_cls, 2 * t / num_of_locs)
        #print (points.shape)
        if global_coreset is None: 
            global_coreset = points
            global_weights = weights
        else:
            global_coreset = np.concatenate((global_coreset, points), axis = 0)
            global_weights += weights
        local_nodes.append(local_node)

    return global_coreset, global_weights


def write_cluster_quality(file_path, res_dict):
    """ This function save the cluster quality to one csv file like
	    "Purity"    "ARI"     "ACC"    "NMI"...
	1   000        000        000     000  ...
	2    000        000        000     000  ...
	...
    The fields to be saved includes
        1. the seed number
	2. Purity
	3. ARI
	4. ACC
	5. NMI

    Args:
        file_path: an absolute file path
    Returns:
        None
    """
    field_names = ['Seed', 'Purity', 'ARI', 'ACC', 'NMI']  # fill out the field names for CSV

    with open(file_path, mode = 'w', newline = '') as csv_file:  # open the file, if not exist, create it
        writer = csv.DictWriter(csv_file, fieldnames = field_names) # create a writer which maps the dictionaries onto output rows in CSV
        writer.writeheader() # write the field names to the header
        for key in res_dict.keys():
            writer.writerow(res_dict[key])

def main():
    ''' This function starts the running
    '''
    print ('Begin----------')

    cwd = os.getcwd()   # get current directory
    parent_dir = os.path.abspath(os.path.join(cwd, os.pardir)) # get the parent directory
    data_dir = os.path.join(parent_dir, 'data')  # get the data directory where the data is stored
    num_of_pars = 100
    #dat_name = 'snr-3'
    #dat_name = 'mnist_unbal'
    dat_name = 'tcga11'
    #dat_name = 'tdt3'
    dat_seed = 0
    #par_method = 'uniform'
    #par_method = 'ubal_noniid#2'
    par_method = 'similarity'

    if dat_name.startswith('snr'):
        is_real, num_of_cls, num_features, num_data = False, 20, 2000, 10000
    elif dat_name.startswith('mnist'):
        is_real, num_of_cls, num_features, num_data = True, 10, 784, 10000
    elif dat_name.startswith('tcga'):
        is_real = True
        if dat_name == 'tcga4':
            num_of_cls, num_features, num_data = 33, 20531, 11135
        elif dat_name == 'tcga5':
            num_of_cls, num_features, num_data = 33, 2000, 11135
        elif dat_name == 'tcga6':
            num_of_cls, num_features, num_data = 10, 2000, 3086
        elif dat_name == 'tcga7':
            num_of_cls, num_features, num_data = 20, 2000, 5314
        elif dat_name == 'tcga10':
            num_of_cls, num_features, num_data = 20, 2000, 7532
        elif dat_name == 'tcga11':
            num_of_cls, num_features, num_data = 20, 5000, 5314
        else:
            num_of_cls, num_features, num_data = 10, 1290, 3353
    elif dat_name.startswith('tdt'):
        is_real = True
        if dat_name == 'tdt1':
            num_of_cls, num_features, num_data = 30, 36771, 9394
        elif dat_name == 'tdt2':
            num_of_cls, num_features, num_data = 30, 2000, 9394
        elif dat_name == 'tdt3':
            num_of_cls, num_features, num_data = 30, 5000, 9394


    data_mat, true_labels, par_idx = gen_and_distribute_data(is_real = is_real, data_name = dat_name, \
            data_dir = data_dir, num_of_features = num_features, num_of_samples = num_data, num_of_source = num_of_cls,
            par_method = par_method, num_of_pars = num_of_pars, seed = dat_seed)

    print ('Data generated and partitioned ----------')

    num_of_outliers = 256  # the number of outlier
    cls_assign = None
    #num_of_outliers = 1024
    res_dict = collections.OrderedDict() # clustering accurary
    for seed_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        global_coreset, global_weights = grow_ball(data_mat = data_mat, par_idx = par_idx, \
                num_of_cls = num_of_cls, num_of_locs = num_of_pars, t = num_of_outliers, seed = seed_num)
        #centers = weighted_clustering(data = global_coreset, weights = global_weights, \
        #        num_of_cls = num_of_cls, seed = seed_num)
        print (len(global_weights))
        # here we use kmeans to do clustering with kmeans++ initialization
        kmeans = KMeans(n_clusters = num_of_cls, init = 'k-means++', n_init = 1, \
                random_state = seed_num).fit(global_coreset, sample_weight = np.array(global_weights))
        centers = kmeans.cluster_centers_
        # get the cluster assignments
        dist = euclidean_distances(data_mat.transpose(), centers)** 2
        cls_assignments = np.argmin(dist, axis = 1)

        if seed_num == 1: cls_assign = cls_assignments

        # save clustering performance
        temp_dict = collections.OrderedDict()
        temp_dict['Seed'] = seed_num
        temp_dict['Purity'] = calculate_purity(cls_assignments, true_labels)
        temp_dict['ARI'] = adjusted_rand_idx = calculate_rand_index(cls_assignments, true_labels)
        temp_dict['ACC'] = calculate_accuracy(cls_assignments, true_labels)
        temp_dict['NMI'] = calculate_NMI(cls_assignments, true_labels)
        res_dict[seed_num] = temp_dict

    print ('Distributed clustering is done --------)')

    res_dir = os.path.join(parent_dir, 'result_icml2', 'parallel_growball', dat_name + '_#' + str(num_of_pars) + 'pars') # get the result directory where the result is stored
    f_manager = FileManager(res_dir)
    f_path = os.path.join(res_dir, par_method, 'cls_quality_#z' + str(num_of_outliers) + '.csv')
    f_manager.add_file(f_path)
    write_cluster_quality(f_path, res_dict)



    cls_assign = best_map(true_labels, cls_assign)
    #cls_assign = np.asarray(np.argmax(np.asmatrix(H), axis = 0))[0, :] # get the cluster assignments
    emb_path = os.path.join(data_dir, 'embeded2', str(dat_name) + '.csv')
    emb_data = get_embeded_data(emb_path, data_mat, 2)
    f_path = os.path.join(res_dir, str(num_of_outliers) + '#otlr' + '_seed1.pdf')
    visualize_data(dat_embeded = emb_data, dim = 2, partition_idx = cls_assign, dat_path = f_path)
    
    print ('Saving clustering quality is done --------')


if __name__ == '__main__':
    main()



