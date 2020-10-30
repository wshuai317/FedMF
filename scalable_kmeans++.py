#######################################################################################
# This script does distributed clustering based on K-means++ and K-means
#
# @author Wang Shuai
# @date 2019.07.15
#######################################################################################

from __future__ import division
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import os 
from numpy import linalg as LA
# import two helper class
from data_generator import DataGenerator as DataGen # manage the data generation
#from cluster_onmf_manager import ClusterONMFManager # manage the result generated
from sklearn.cluster import KMeans
from file_manager import FileManager
from cluster_metrics import *
import csv
import collections
from cluster_tsne import *

def distance(data, centroids):
    ''' The function calculate the distance from each data point to each center

    Args:
        data (numpy array or mat): data array with rows being samples and columns being features
        center (numpy array or mat): each row is a center
    Returns:
        distance matrix
    '''
    #print (data[:, np.newaxis, :])
    #dist = np.sum((data - centroids) ** 2, axis = 2)
    return euclidean_distances(data, centroids)** 2

def cost(dist):
    ''' The function calculate the cost of data with respect to the current centroids

    Args:
        dist (numpy array or mat): distance matrix between data and current centroids
    Returns:
        the normalized constant in the distribution
    '''
    return np.sum(np.min(dist, axis = 1))

def distribution(dist, cost):
    ''' The function calculates the distribution to sample new centers

    Args:
        dist (numpy array or mat): distance matrix between data and current centroids
        cost (float): the total cost of data with respect to the current centroids
    Returns:
        distribution
    '''
    return np.min(dist, axis = 1) / cost

def sample_new(data, distribution, l):
    ''' The function sample new centers based the distribution

    Args:
        data (numpy array or mat): rows being samples and columns being features
        distribution (1-D numpy array): the probability of each point being selected
        l (int): the number of new centers to sample
    Returns:
        new centers
    '''
    print (distribution)
    idx = np.random.choice(range(len(distribution)), l, p = distribution, replace = False)
    return data[idx, :]

def get_weight(dist, centroids):
    ''' The function calculate the weights of the centroids for the data points
    Specifically, the weight of a centroid is the number of data points which
    are closet to the centroid

    Args:
        dist
    '''
    min_dist = np.zeros(dist.shape)
    min_dist[range(dist.shape[0]), np.argmin(dist, axis = 1)] = 1
    count = np.array([np.count_nonzero(min_dist[:, i]) for i in range(centroids.shape[0])])
    return count

def scalable_kmeans_pp(data = None, k = 10, l = None, iter_num = 5, seed_num = 1):
    ''' The function apply the kmeans|| method to get initial centroids for kmeans

    Args:
        data (numpy array or mat): data matrix with rows being samples
        k (int): the number of clusters
        l (int): the number of point sampled in each iteration
        iter_num (int): the max number of iterations for kmeans||
        seed_num (int): the seed for random data generator used for
                        1. the first center in kmeans||
                        2. the initialization for kmeans++ for weighted clustering
    Returns:
        the final centroids found by kmeans|| + kmeans
        the cluster assignments for data points
    '''
    if l is None: # we set l as 2 * K defaultly
        l = 10 * k
    np.random.seed(seed_num)
    # randomly sample a data point as the center
    centroids = data[np.random.choice(range(data.shape[0]), 1), :]
    for i in range(iter_num):
        #print (centroids[:, 0:5])
        # get the distance between data and centroids
        dist = distance(data, centroids)
        # calculate the cost of data with respect to the centroids
        total_cost = cost(dist)
        # calculate the distribution for sampling l new centers
        p = distribution(dist, total_cost)
        # sample l new centers and append them to the original ones
        centroids = np.r_[centroids, sample_new(data, p, l)]

    # get the weights for all the centroids
    dist = distance(data, centroids)
    weights = get_weight(dist, centroids)

    num_of_centers = centroids.shape[0]
    new_data = np.zeros((np.sum(weights), data.shape[1]))
    start_idx = 0
    end_idx = 0
    for row_idx in range(num_of_centers): # for each center, replicate it based on its weights
        end_idx = start_idx + weights[row_idx]
        new_data[start_idx:end_idx, :] = centroids[row_idx, :]
        start_idx = end_idx

    # here we use kmeans to do clustering with kmeans++ initialization
    #kmeans = KMeans(n_clusters = k, init = 'k-means++', n_init = 1, random_state = seed_num).fit(data)
    kmeans = KMeans(n_clusters = k, init = 'k-means++', n_init = 1, random_state = seed_num).fit(centroids, sample_weight = weights)
    final_centers = kmeans.cluster_centers_
    # get the cluster assignments
    dist = distance(data, final_centers)
    print (dist.shape)
    labels = np.argmin(dist, axis = 1)
    print (len(labels))

    return final_centers, labels

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
            num_of_cls = num_of_source, seed = seed)

    # partition the dataset
    if not data_name.startswith('mnist'):
        data_gen.partition_data_array(num_of_pars = num_of_pars, m_name = par_method) # partition the data

    if not data_gen.existed_or_not():
        raise ValueError('Error: the synthetic data is not generated before. So please ensure \
                the data just generated has been prepared for local nodes and then run again!')

    data_mat, true_labels = data_gen.get_data_and_labels()
    par_idx = data_gen.get_partitions() # get the partition index

    return data_mat, true_labels, par_idx


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
    dat_seed = 0
    #dat_name = 'mnist_unbal'
    #dat_name = 'tdt3'
    #dat_name = 'snr-3'
    #par_method = 'similarity'
    dat_name = 'tcga11'
    par_method = 'similarity'
    #par_method = 'uniform'
    num_pars = 100
    #dat_name = 'mnist1'
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
            par_method = par_method, num_of_pars = num_pars, seed = dat_seed)

    print ('Data generated and partitioned ----------')

    s_iter = 10 # the number of iters for sampling
    s_num = 20 # #K sampled for each iter

    cls_assign = None

    res_dict = collections.OrderedDict() # clustering accurary
    #for seed_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
    for seed_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        # kmeans|| initialization + kmeans++ weighted clustering
        f_centers, cls_assignments = scalable_kmeans_pp(data = data_mat.transpose(), \
                k = num_of_cls, l = s_num * num_of_cls, iter_num = s_iter, seed_num = seed_num)
        # save clustering performance
        temp_dict = collections.OrderedDict()
        temp_dict['seed'] = seed_num
        temp_dict['Purity'] = calculate_purity(cls_assignments, true_labels)
        temp_dict['ARI'] = adjusted_rand_idx = calculate_rand_index(cls_assignments, true_labels)
        temp_dict['ACC'] = calculate_accuracy(cls_assignments, true_labels)
        temp_dict['NMI'] = calculate_NMI(cls_assignments, true_labels)
        res_dict[seed_num] = temp_dict

        if seed_num == 1: cls_assign = cls_assignments

    print ('Clustering by kmeans|| is done ----------')

    res_dir = os.path.join(parent_dir, 'result_icml2', 'kmeans||', dat_name, par_method) # get the result directory where the result is stored
    f_manager = FileManager(res_dir)
    f_path = os.path.join(res_dir, 'cls_quality_#' + str(s_iter) + 'iter_#' + str(s_num) + 'k.csv')
    f_manager.add_file(f_path)
    write_cluster_quality(f_path, res_dict)
    
    cls_assign = best_map(true_labels, cls_assign)
    #cls_assign = np.asarray(np.argmax(np.asmatrix(H), axis = 0))[0, :] # get the cluster assignments
    emb_path = os.path.join(data_dir, 'embeded2', str(dat_name) + '.csv')
    emb_data = get_embeded_data(emb_path, data_mat, 2)
    f_path = os.path.join(res_dir, str(s_iter) + '#iter_' + str(s_num) + '#num' +  '_seed1.pdf')
    visualize_data(dat_embeded = emb_data, dim = 2, partition_idx = cls_assign, dat_path = f_path)
    
    print ('Saving clustering quality is done --------')

if __name__ == '__main__':
     main()

