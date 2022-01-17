###########################################################################################
# The script is used to reproduce the result for the paper
#
# @author Wang Shuai
# @date 2020.02.18
###########################################################################################

from __future__ import division

import numpy as np
import pandas as pd
import os
import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn import manifold
from file_manager import FileManager
import time
from numpy import linalg as LA
# import two helper class
from data_generator import DataGenerator as DataGen  # manage the data generation
from cluster_metrics import *
from sklearn import manifold


def read_data_from_csvfile(f_path):
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

    print(f_path)
    # read csv file into a pandas dataframe
    df = pd.read_csv(f_path, header=None)

    # convet the dataframe into a numpy mat
    data_arr = df.as_matrix()

    print(data_arr.shape)

    return data_arr


def get_embeded_data(embeded_path, data, dim):
    # embeded_path = os.path.join(res_dir, 'real_data', self.data_kind, 'data_' + bal_str + '_#' + str(
    # num_of_samples) + '_seed' + str(seed) + '.csv')
    print(os.path.exists(embeded_path))
    if os.path.exists(embeded_path):  # the data file exists, just read it
        embeded_data = read_data_from_csvfile(embeded_path)
        return embeded_data
    else:
        if data.shape[0] > dim:
            print('transform---------------')
            tsne = manifold.TSNE(n_components=dim, random_state=0)
            print('data_mat:' + str(data.shape))
            embeded_data = tsne.fit_transform(data.transpose())
        else:
            embeded_data = np.asmatrix(np.copy(data.transpose()))
        f_manager = FileManager(os.path.dirname(embeded_path))
        f_manager.add_file(embeded_path)
        np.savetxt(embeded_path, np.asmatrix(embeded_data), delimiter=',')
        return embeded_data


def visualize_data(dat_embeded=None, dim=2, partition_idx=None, dat_path=None, \
                   xlabel='', ylabel='', title='', data_points=None):
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

    if dat_embeded is None or partition_idx is None:
        raise ValueError('Error: the data embeded is none!')

    colormap = np.array(['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
                         '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324',
                         '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
                         '#000000'])

    print('data_embeded: ' + str(dat_embeded.shape))
    print(len(partition_idx))
    print(partition_idx)
    # map cluster indexs to 0-20
    pars = list(map(int, partition_idx))
    pars = np.asarray(pars)
    print(pars)
    # pars = artition_idx
    # pars = map(int, partition_idx)
    # print (list(pars))
    u_elms = np.unique(partition_idx)
    ind = -1
    num = len(list(u_elms))
    print('num : ' + str(num))
    if num > 20:
        raise ValueError('The number of pars is greater than 20!!')

    step = int(20 / num)
    p_list = []
    for i in range(num):
        if not u_elms[i] in p_list:
            ind = ind + step
            pars[pars == u_elms[i]] = ind
            p_list.append(u_elms[i])

    '''            
    print (pars)
    if len(list(u_elms)) > self.num_of_cls or len(p_list) > self.num_of_cls:
        print (len(list(u_elms)), len(p_list), self.num_of_cls)
        raise ValueError('Error: the number of clusters are not consistent')
    '''
    print(list(pars))
    color_used = colormap[list(pars)]
    dat_embeded = np.array(dat_embeded)
    print(dat_embeded[0:5, 0:2])
    print(dat_embeded.shape)
    plt.figure()
    plt.scatter(dat_embeded[:, 0], dat_embeded[:, 1], c=color_used)
    if not data_points is None:
        print(data_points.shape)
        plt.scatter(data_points[:, 0], data_points[:, 1], c='blue', s=40)
    # plt.ylim(0, 20)
    # plt.xlim(0, 20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(dat_path, bbox_inches='tight')


def gen_and_distribute_data(is_real=False, data_name='snr-1', data_dir='data',
                            num_of_features=2000, num_of_samples=1000, num_of_source=10,
                            par_method='uniform', num_of_pars=100, seed=10):
    """ This function is used to generate the raw data for distributed computing.
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
    """
    # generate the dataset and the corresponding labels
    data_gen = DataGen(root_dir=data_dir, is_real=is_real, data_name=data_name,
                       num_of_features=num_of_features, num_of_samples=num_of_samples,
                       num_of_cls=num_of_source)

    # partition the dataset
    data_gen.partition_data_array(num_of_pars=num_of_pars, m_name=par_method)  # partition the data

    if not data_gen.existed_or_not():
        raise ValueError('Error: the synthetic data is not generated before. So please ensure \
                the data just generated has been prepared for local nodes and then run again!')

    return data_gen


def main():
    """ This function starts the running
    """
    print('Begin----------')

    cwd = os.getcwd()  # get current directory
    parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))  # get the parent directory
    data_dir = os.path.join(parent_dir, 'data')  # get the data directory where the data is stored
    dat_seed = 0
    nu = 1e-10
    rho = 1e-8
    dim = 2

    for dat_name in ['tcga11']:
        # for dat_name in ['mnist_unbal', 'tdt3', 'tcga11']:
        for par_m in ['similarity']:
            if dat_name.startswith('snr'):
                is_real, num_of_cls, num_features, num_data, tol = False, 20, 2000, 10000, 5e-5
            elif dat_name.startswith('mnist'):
                is_real, num_of_cls, num_features, num_data, tol = True, 10, 784, 10000, 1e-4
            elif dat_name.startswith('tcga'):
                if dat_name == 'tcga5':
                    is_real, num_of_cls, num_features, num_data, tol = True, 33, 2000, 11135, 2e-5
                elif dat_name == 'tcga6':
                    is_real, num_of_cls, num_features, num_data, tol = True, 10, 2000, 3086, 2e-5
                elif dat_name == 'tcga7':
                    is_real, num_of_cls, num_features, num_data, tol = True, 20, 2000, 5314, 8e-5
                elif dat_name == 'tcga8':
                    is_real, num_of_cls, num_features, num_data, tol = True, 20, 2000, 5314, 2e-5
                elif dat_name == 'tcga10':
                    is_real, num_of_cls, num_features, num_data, tol = True, 20, 2000, 7532, 1e-5
                elif dat_name == 'tcga11':
                    is_real, num_of_cls, num_features, num_data, tol = True, 20, 5000, 5314, 1e-4
                else:
                    is_real, num_of_cls, num_features, num_data, tol = True, 10, 1290, 3353, 2e-5
            elif dat_name.startswith('coil'):
                is_real, num_of_cls, num_features, num_data, tol = True, 100, 1024, 7200, 2e-5
            elif dat_name.startswith('tdt'):
                if dat_name == 'tdt1':
                    is_real, num_of_cls, num_features, num_data, tol = True, 30, 36771, 9394, 2e-5
                elif dat_name == 'tdt2':
                    is_real, num_of_cls, num_features, num_data, tol = True, 30, 2000, 9394, 1e-6
                elif dat_name == 'tdt3':
                    is_real, num_of_cls, num_features, num_data, tol = True, 30, 5000, 9394, 1e-4
                elif dat_name == 'tdt4':
                    is_real, num_of_cls, num_features, num_data, tol = True, 30, 8000, 9394, 1e-5

        # generate the data and partition it
        data_gen = gen_and_distribute_data(is_real=is_real, data_name=dat_name,
                                           data_dir=data_dir, num_of_features=num_features, num_of_samples=num_data,
                                           num_of_source=num_of_cls, par_method=par_m, num_of_pars=100, seed=dat_seed)
        print('Data generated and partitioned ----------')
        # get the data, labels
        data_mat, true_labels = data_gen.get_data_and_labels()

        emb_path = os.path.join(data_dir, 'embeded2', str(dat_name) + '.csv')
        emb_data = get_embeded_data(emb_path, data_mat, dim)

        f_path = os.path.join(data_dir, 'embeded2', str(dat_name) + '.pdf')

        visualize_data(dat_embeded=emb_data, dim=dim, partition_idx=true_labels, dat_path=f_path)

        # get cluster assign
        H_dir = os.path.join(parent_dir,
                             'result_acc/parallel_fedcgds1.5/tcga11/similarity#100/uniform#100/H#10_W#100_rho#1e-08_nu#1e-10_tol#5e-05/seed#1/prim_vars/H/')
        H_path = os.path.join(H_dir, '1001.csv')
        H = read_data_from_csvfile(H_path)
        cls_assign = np.asarray(np.argmax(np.asmatrix(H), axis=0))[0, :]  # get the cluster assignments
        print(len(cls_assign))
        print(len(true_labels))
        cls_assign = best_map(true_labels, cls_assign)
        f_path = os.path.join(parent_dir, 'result_icml2', 'fedcgds', 'uniform100', str(dat_name) + '.pdf')
        f_manager = FileManager(os.path.dirname(f_path))
        visualize_data(dat_embeded=emb_data, dim=dim, partition_idx=cls_assign, dat_path=f_path)


if __name__ == '__main__':
    main()
