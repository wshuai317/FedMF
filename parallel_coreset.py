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
    def __init__(self, data, num_of_cls = 10, comm = None, seed = 1):
        ''' __init__ function

        Args
            data (numpy array): the local data
            num_of_cls (int): the number of clusters
            seed (int): the seed for random generator
        Returns:
            None
        '''
        #self.neighbors = neighbors
        #self.degree = degree
        self.data = data            # Holds the local data Pi
        self.centers = None         # Holds the centers Bi
        self.local_coreset = None   # To store coreset
        self.weights = None         # To store the weights of points in local coreset
        #self.message_received = {}
        #self.X = None              # To store the final centers
        self.cost_of_each_data = None
        self.local_cost = None
        self.num_of_cls = num_of_cls
        self.comm = comm
        self.seed = seed
        np.random.seed(seed)
        #self.proc_id = proc_id

    def local_clustering(self):
        ''' compute local approximation solution using k-means
        Specifically, the centers and cost of each data will be set

        Args:
            None
        Returns:
            None
        '''
        cls_num = np.minimum(self.data.shape[0], self.num_of_cls)
        kmeans = KMeans(n_clusters = cls_num, init = 'k-means++', n_init = 10, \
                random_state = self.seed).fit(self.data)
        self.centers = kmeans.cluster_centers_ # set the centers
        self.local_cost = kmeans.inertia_   # set the local cost as sum of squared distances of
                                            # data samples to their closet enters
        self.labels = kmeans.labels_  # the labels tell the closet center of each data sample
        self.cost_of_each_data = np.zeros(len(self.labels))
        for i in range(len(self.labels)): # for each data point
            self.cost_of_each_data[i] = LA.norm(self.centers[self.labels[i], :] - self.data[i, :]) ** 2


    def construct_local_coreset(self, t_num, total_cost):
        ''' The function locally compute the coreset including the following steps:
            1. compute the size of sampled data points
            2. compute the probability of each data point during sampling
            3. sampling
            4. compute weights for data points in local coreset
        Args:
           t_num (int): the total number of data samples to be sampled
           total_cost (float): the total cost of all local nodes based on Bi
        Returns:
            None
        '''
        if self.centers is None: # if the local approximation solution is not computed
            self.get_local_solution()
        print ('calculating ti, m_p, S_i, w_q, w_b')
        ti = math.floor(t_num * self.local_cost / total_cost)  # the number of to be sampled locally
        if ti > 0:
            m_p = 2 * self.cost_of_each_data
            probs = m_p / np.sum(m_p)   # the probability of each data being sampled
            #np.random.seed(self.seed)  # set the seed
            print ('samling ' + str(ti) + ' data points--')
            '''
            if ti > self.data.shape[0]:
                flag = True
            else:
                flag = False
            '''
            flag = True
            Si_idx = np.random.choice(range(len(m_p)), size = ti, replace = flag, p = probs)
            S_i = self.data[Si_idx, :]  # get the sampled data points
            w_q = 2 * total_cost / (t_num * m_p[Si_idx]) # calculate the weights
            w_b = np.zeros(self.num_of_cls)
            # calcualte the weights for the centers
            #_, Pb = np.unique(self.labels, return_counts = True) # the size of each cluster
            for i in range(self.num_of_cls):
                print ('finding points belonging to center ' + str(i))
                Pb = np.where(self.labels == i)[0]  # the index of points in cluster i
                temp = 0 # the weights of intersection of Pb and S_i
                for j in range(ti): # for each sampled data point
                    if Si_idx[j] in Pb:
                        temp += w_q[j]
                w_b[i] = len(Pb) - temp
                #print ('weights done')
            # concatenate the sampled data points and centers, and their associated weights
            # Note that S_i and self.centers must not be 1-D
            self.local_coreset = np.concatenate((S_i, self.centers), axis = 0)
            self.weights = np.concatenate((w_q, w_b), axis = None)
        else:
            self.local_coreset = None
            self.weights = None

        return self.local_coreset, self.weights


    def get_local_coreset(self):
        ''' The function returns the computed local coreset and the associated weights

        Args:
            None
        Returns:
            None
        '''
        return self.local_coreset, self.weights

    def get_local_cost(self):
        ''' The function returns the local cost based on local solution Bi

        Args:
            None
        Returns:
            None
        '''
        return self.local_cost


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

    q, counts = np.unique(par_idx, return_counts = True)
    print (counts)

    return data_mat, true_labels, par_idx


def construct_coreset(data_mat = None, par_idx = None, true_labels = None, \
        num_of_cls = 10, num_of_locs = 3, t_sum = None, seed = 1):
    ''' The function performs construction of global coreset for the input data by combining
    local coresets. These local coresets are sampled locally by the local nodes. Specifically,
        1. Compute local approximation to Bi for Pi (local data)
        2. Communicate Cost(Pi, Bi) to the central site which get the total cost and distribute
        3. Given the total cost, each local site samples the data sampling based on a certain
            probability which is computed from the total cost and Bi.
        4. The local coreset is the union of weighted sampled data points and Bi.
        5. The gloal coreset is the union of all local coresets.

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
    total_cost = 0   # save the total cost by summing up all local costs
    for node_id in range(num_of_locs):
        local_data_idx = np.where(par_idx == node_id)[0] # get the index of data samples belong to local site i
        local_data = data_mat[:, local_data_idx]   # get local data
        local_node = Node(local_data.transpose(), num_of_cls = num_of_cls, seed = seed)
        local_node.local_clustering()  # perform clustering to obtain local approximation soluition
        total_cost += local_node.get_local_cost()
        local_nodes.append(local_node)

    #print ('totla_cost: ' + str(total_cost))

    # construct local corest by each node and get the global corest by union
    global_coreset = None
    global_weights = None
    for node_id in range(num_of_locs):
        local_coreset, local_weights = local_nodes[node_id].construct_local_coreset(t_sum, total_cost)
        if global_coreset is None:
            global_coreset = local_coreset
            global_weights = local_weights
        else:
            # concatenate the sampled data points and centers, and their associated weights
            # Note that they must not be 1-D
            if not local_coreset is None:
                global_coreset = np.concatenate((global_coreset, local_coreset), axis = 0)
                global_weights = np.concatenate((global_weights, local_weights), axis = None)

    return global_coreset, global_weights


def weighted_clustering(data = None, weights = None, num_of_cls = 10, seed = 1):
    ''' The function performs clustering (kmeans++) on weighted data samples

    Args:
        data (numpy array or mat): the input data with rows being samples
        weights (numpy array): the weights correspongding to the data samples
        num_of_cls (int): the number of clusters
        seed (float): the seed used for kmeans++ initialization
    Returns:
        K centers
    '''
    # replicate the data points according to their weights
    new_data = np.zeros((np.sum(weights), data.shape[1]))
    start_idx = 0
    end_idx = 0
    for row_idx in range(data.shape[0]): # for each center, replicate it based on its weights
        end_idx = start_idx + weights[row_idx]
        new_data[start_idx:end_idx, :] = data[row_idx, :]
        start_idx = end_idx

    # here we use kmeans to do clustering with kmeans++ initialization
    kmeans = KMeans(n_clusters = num_of_cls, init = 'k-means++', n_init = 1, random_state = seed).fit(new_data)
    final_centers = kmeans.cluster_centers_

    return final_centers

def write_cluster_quality(file_path, res_dict):
    """ This function save the cluster quality to one csv file like
	    "seed_num" "Purity"    "ARI"     "ACC"    "NMI"...
	1      1       000        000        000     000  ...
	2      2       000        000        000     000  ...
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
    field_names = ['seed', 'Purity', 'ARI', 'ACC', 'NMI']  # fill out the field names for CSV

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
    #num_of_cls = 20
    num_of_pars = 100
    #dat_name = 'snr-3'
    #dat_name = 'mnist_unbal'
    dat_name = 'tcga11'
    #dat_name = 'tdt3'
    dat_seed = 0
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
    elif dat_name.startswith('coil'):
        is_real, num_of_cls, num_features, num_data = True, 100, 1024, 7200
    elif dat_name.startswith('ypmsd'):
        is_real, num_of_cls, num_features, num_data = True, 78, 90, 51630
    elif dat_name.startswith('tdt'):
        if dat_name == 'tdt1':
            is_real, num_of_cls, num_features, num_data, tol = True, 30, 36771, 9394, 2e-5
        elif dat_name == 'tdt2':
            is_real, num_of_cls, num_features, num_data, tol = True, 30, 2000, 9394, 2e-5
        elif dat_name == 'tdt3':
            is_real, num_of_cls, num_features, num_data, tol = True, 30, 5000, 9394, 2e-5


    data_mat, true_labels, par_idx = gen_and_distribute_data(is_real = is_real, data_name = dat_name, \
            data_dir = data_dir, num_of_features = num_features, num_of_samples = num_data, num_of_source = num_of_cls,
            par_method = par_method, num_of_pars = num_of_pars, seed = dat_seed)

    print ('Data generated and partitioned ----------')

    t_factor = 0.5   # set the total size of coreset = t_factor * num_of_samples
    t_num = t_factor * data_mat.shape[1]
    res_dict = collections.OrderedDict() # clustering accurary

    cls_assign = None
    # we perform distributed clustering multiple times with different seeds
    #for seed_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
    for seed_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        global_coreset, global_weights = construct_coreset(data_mat = data_mat, par_idx = par_idx, \
                true_labels = true_labels, num_of_cls = num_of_cls, num_of_locs = num_of_pars, \
                t_sum = t_num, seed = seed_num)
        #centers = weighted_clustering(data = global_coreset, weights = global_weights, \
        #        num_of_cls = num_of_cls, seed = seed_num)
        print (global_weights.shape)
        # here we use kmeans to do clustering with kmeans++ initialization
        kmeans = KMeans(n_clusters = num_of_cls, init = 'k-means++', n_init = 1, \
                random_state = seed_num).fit(global_coreset, sample_weight = global_weights)
        centers = kmeans.cluster_centers_
        # get the cluster assignments
        dist = euclidean_distances(data_mat.transpose(), centers)** 2
        cls_assignments = np.argmin(dist, axis = 1)

        if seed_num == 1: cls_assign = cls_assignments

	# save clustering performance
        temp_dict = collections.OrderedDict()
        temp_dict['seed'] = seed_num
        temp_dict['Purity'] = calculate_purity(cls_assignments, true_labels)
        temp_dict['ARI'] = adjusted_rand_idx = calculate_rand_index(cls_assignments, true_labels)
        temp_dict['ACC'] = calculate_accuracy(cls_assignments, true_labels)
        temp_dict['NMI'] = calculate_NMI(cls_assignments, true_labels)
        res_dict[seed_num] = temp_dict

    print ('Distributed clustering is done --------)')

    res_dir = os.path.join(parent_dir, 'result_icml2', 'parallel_coreset', dat_name + '_#' + str(num_of_pars) + 'pars') # get the result directory where the result is stored
    f_manager = FileManager(res_dir)
    f_path = os.path.join(res_dir, 'cls_quality_#' + '_' + par_method + str(t_factor) + 'N.csv')
    f_manager.add_file(f_path)
    write_cluster_quality(f_path, res_dict)

    
    cls_assign = best_map(true_labels, cls_assign)
    #cls_assign = np.asarray(np.argmax(np.asmatrix(H), axis = 0))[0, :] # get the cluster assignments
    emb_path = os.path.join(data_dir, 'embeded2', str(dat_name) + '.csv')
    emb_data = get_embeded_data(emb_path, data_mat, 2)
    f_path = os.path.join(res_dir, str(t_factor) + 'seed1.pdf')
    visualize_data(dat_embeded = emb_data, dim = 2, partition_idx = cls_assign, dat_path = f_path)
    
    print ('Saving clustering quality is done --------')


if __name__ == '__main__':
    main()



