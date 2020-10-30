###########################################################################################
# The main entry script uses SNCP method on ONMF model to perform clustering
#
# @author Wang Shuai
# @date 2019.10.18
###########################################################################################

from __future__ import division

import numpy as np
import os
import time
from numpy import linalg as LA
# import two helper class
from data_generator import DataGenerator as DataGen # manage the data generation
from cluster_onmf_manager import ClusterONMFManager # manage the result generated


def get_onmf_cost(data, n_factor, W, H, nu = 1e-4, mul = 0):
    ''' This function returns the approximation error of ONMF based on current W and H

    Args:
        data (numpy array or mat): the input data
        W (numpy array or mat): the factor W
        H (numpy array or mat): the factor H
        nu (float): the penalty parameter
    Returns:
        the cost
    '''
    (ha, hb) = H.shape
    res = LA.norm(data - W * H, 'fro')**2 / hb \
		+ 0.5 * nu * LA.norm(H, 'fro')** 2 \
                + 0.5 * mul * LA.norm(W, 'fro') ** 2
    return res * hb / n_factor

def get_sncp_cost(data, n_factor, W, H, nu, mul, rho):
    ''' This function returns the cost of the penalized subproblem when using SNCP

    Args:
        data (numpy array or mat): the input data
        n_factor (float): the normalization fator
        W (numpy array or mat): the factor W
        H (numpy array or mat): the factor H
        nu (float): the parameter nu * ||H||_F^2
        mul (float): the parameter mul * ||W||_F^2
        rho (float): the penalty parameter rho * \sum_j (||hj||_1^2 - ||hj||_2^2)
    Returns:
        the cost
    '''
    (ha, hb) = H.shape
    all_1_mat = np.asmatrix(np.ones((ha, ha)))
    p_cost = LA.norm(data - W * H, 'fro') ** 2 / hb \
            + 0.5 * nu * LA.norm(H, 'fro')** 2  \
            + 0.5 * mul * LA.norm(W, 'fro') ** 2 \
            + 0.5 * rho * (np.trace(H.transpose() * all_1_mat * H) \
            - LA.norm(H, 'fro') ** 2)
    return p_cost * hb / n_factor

def gen_and_distribute_data(is_real = False, data_name = 'snr-1', data_dir = 'data', \
        num_of_features = 2000, num_of_samples = 1000, num_of_source = 10,
        par_method = 'uniform', num_of_pars = 100, seed = 10):
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

    return data_gen

def sncp(dat_gen = None, res_dir = 'res', num_of_cls = 10, seed = 1, nu = 1e-4, \
        rho = 1e-8, tol = 2e-3, DEBUG = False):
    ''' This function peforms the clustering using the SNCP method on the ONMF model

    Args:
        data_gen (object): the data generator
        res_dir (string): the res_dir where the generated results are stored
        seed (int): the seed for random generator when
                    1. init for W and H
                    2. sample data points when update W
        nu (float): the setting for parameter \nu in sncp
        rho (float): the initialization for penalty rho
        tol (float): the optimization accuracy for each penalized subproblem
        DEBUG (boolean): the flag to indicate whether print the debug info
    Returns:
        None
    '''
    # get the data, labels, and partitions
    data_mat, true_labels = dat_gen.get_data_and_labels()
    """
    par_idx = dat_gen.get_partitions() # get the partition index
    arr = None
    for j in range(len(np.unique(par_idx))):
        tmp = np.where(par_idx == j)[0]
        if arr is None:
            arr = tmp
        else:
            arr = np.concatenate((arr, tmp))
    data_mat = data_mat[:, arr]
    true_labels = true_labels[arr]
    """
    # init W and H
    W_init, H_init = dat_gen.gen_inits_WH(seed = seed)

    n_factor = LA.norm(data_mat, 'fro') ** 2 # we normalize the ||X - WH||_F^2 so that the initial
                                             # value of rho (penalty) is suitable for all datasets
    data_num = data_mat.shape[1]

    print ('n_factor: ' + str(n_factor))

    # the setting for the penalty 
    rho = rho * n_factor / data_num 
    nu = nu * n_factor / data_num
    mul = 0

    max_val, min_val = np.max(data_mat), np.min(data_mat)

    # we construct a result manager to manage and save the result
    res_manager = ClusterONMFManager(root_dir = res_dir, save_pdv = False) # get an instance of ClusterONMFManager to manage the generated result
    res_manager.push_W(W_init)  # store W
    res_manager.push_H(H_init)  # store H
    res_manager.push_H_norm_ortho()  # store feasibility
    res_manager.add_cost_value('onmf_cost', get_onmf_cost(data_mat, n_factor, W_init, H_init, nu, 0)) # store obj val
    res_manager.add_cost_value('sncp_cost', get_sncp_cost(data_mat, n_factor, W_init, H_init, nu, 0, rho))
    res_manager.calculate_cluster_quality(true_labels) # calculate and store clustering quality

    W_cur, W_pre = np.asmatrix(np.copy(W_init)), np.asmatrix(np.copy(W_init))
    H_cur, H_pre = np.asmatrix(np.copy(H_init)), np.asmatrix(np.copy(H_init))

    # some intermediate values during computation
    (ha, hb) = H_pre.shape
    I_ha = np.asmatrix(np.eye(ha))
    all_1_mat = np.asmatrix(np.ones((ha, ha)))

    p_max_iter = 500 # the max number of iterations for each subproblem
    temp_iter = 0 # used to calculate iters used for each subproblem
    beta_rho = 1.5 # the increase factor of penalty rho

    sncp_iter = 1
    while True:
        # firstly, we update H using PGD on 
        #  \min_{H \in \mathcal{H}} F(W, H)
        #  where F(W, H) = ||X - W * H||_F^2 / N + (0.5 * nu * ||H_p||_F^2 
        #                    + 0.5 * rho * sum_{j=1}^{N_p}((1^Thj)^2 - ||hj||_F^2)) 
        Hessian = 2 * W_pre.transpose() * W_pre / data_num + rho * all_1_mat + (nu - rho) * I_ha
        egenvals, _ = LA.eigh(Hessian)
        t = 0.51 * np.max(egenvals)
        grad_H_pre = Hessian * H_pre - 2 * W_pre.transpose() * data_mat / data_num
        H_cur = np.maximum(0, H_pre - grad_H_pre / t)

        if DEBUG:
            #onmf_cost = master_node.get_onmf_cost()
            sncp_cost = get_sncp_cost(data_mat, n_factor, W_pre, H_cur, nu, mul, rho)
            print ('SNCP_Iter(' + str(sncp_iter) + '), after update H, sncp_cost = ' + str(sncp_cost))

        # Then, we update W using PGD on 
        #   \min_{W \in \mathcal{W}} F(W, H)
        #  where F(W, H) = ||X - W * H||_F^2 / N
        Hessian = 2 * H_cur * H_cur.transpose() / data_num
        egenvals, _ = LA.eigh(Hessian)
        c = 0.51 * np.max(egenvals)
        grad_W_pre = W_pre * Hessian - 2 * data_mat * H_cur.transpose() / data_num
        W_cur = np.maximum(0, np.minimum(max_val, W_pre - grad_W_pre / c))

        if DEBUG:
            #onmf_cost = master_node.get_onmf_cost()
            sncp_cost = get_sncp_cost(data_mat, n_factor, W_cur, H_cur, nu, mul, rho)
            print ('SNCP_Iter(' + str(sncp_iter) + '), after update W, sncp_cost = ' + str(sncp_cost))

        # save results
        res_manager.push_W(W_cur)  # store W
        res_manager.push_H(H_cur)  # store H
        res_manager.push_H_norm_ortho()  # store feasibility
        res_manager.add_cost_value('onmf_cost', get_onmf_cost(data_mat, n_factor, W_cur, H_cur, nu, 0)) # store obj val
        res_manager.add_cost_value('sncp_cost', get_sncp_cost(data_mat, n_factor, W_cur, H_cur, nu, 0, rho))
        res_manager.calculate_cluster_quality(true_labels) # calculate and store clustering quality
        res_manager.push_W_norm_residual()
        res_manager.push_H_norm_residual()

        sncp_cost_nr = res_manager.peek_sncp_cost_residual()
        W_change = LA.norm(W_cur - W_pre, 'fro') / LA.norm(W_pre, 'fro')
        print ('W_change: ' + str(W_change) + ' sncpc: ' + str(sncp_cost_nr) + ' ortho: ' + str(res_manager.peek_H_norm_residual()))

        if sncp_cost_nr < 1e-8 or sncp_iter > 2000: # converges
            print ('Converges------')
            res_manager.write_to_csv()  # store the generated results to csv files
            break
        else: 
            # if not converge, increase rho when subproblem is solved to certain accuracy or
            # the subproblem is solved with p_max_iter iterations
            if sncp_cost_nr < tol or sncp_iter % p_max_iter == 0:
                print ('Increase penalty parameter rho ------')
                print ('Push iters_used ' + str(sncp_iter - temp_iter))
                res_manager.push_iters(rho, sncp_iter - temp_iter)
                temp_iter = sncp_iter
                rho =  np.minimum(beta_rho * rho, 100 * n_factor / data_num)

        # update
        W_pre = np.asmatrix(np.copy(W_cur))
        H_pre = np.asmatrix(np.copy(H_cur))
        sncp_iter += 1
        print ('rho: ' + str(rho * data_num / n_factor))

def main():
    ''' This function starts the running
    '''
    print ('Begin----------')

    cwd = os.getcwd()   # get current directory
    parent_dir = os.path.abspath(os.path.join(cwd, os.pardir)) # get the parent directory
    data_dir = os.path.join(parent_dir, 'data')  # get the data directory where the data is stored
    dat_seed = 0
    nu = 1e-10
    rho = 1e-8

    #for dat_name in ['snr-3', 'tcga11', 'tdt3']:
    for dat_name in ['mnist_bal', 'mnist_unbal']:
        if dat_name.startswith('snr'):
            is_real, num_of_cls, num_features, num_data, tol = False, 20, 2000, 10000, 1e-5
        elif dat_name.startswith('mnist'):
            is_real, num_of_cls, num_features, num_data, tol = True, 10, 784, 10000, 1e-5
        elif dat_name.startswith('tcga'):
            if dat_name == 'tcga5':
                is_real, num_of_cls, num_features, num_data, tol = True, 33, 2000, 11135, 1e-5
            elif dat_name == 'tcga6':
                is_real, num_of_cls, num_features, num_data, tol = True, 10, 2000, 3086, 1e-5
            elif dat_name == 'tcga7':
                is_real, num_of_cls, num_features, num_data, tol = True, 20, 2000, 5314, 1e-5
            elif dat_name == 'tcga8':
                is_real, num_of_cls, num_features, num_data, tol = True, 20, 2000, 5314, 1e-5
            elif dat_name == 'tcga9':
                is_real, num_of_cls, num_features, num_data, tol = True, 20, 5000, 5314, 1e-5
            elif dat_name == 'tcga11':
                is_real, num_of_cls, num_features, num_data, tol = True, 20, 5000, 5314, 1e-5
            else:
                is_real, num_of_cls, num_features, num_data, tol = True, 10, 1290, 3353, 1e-5
        elif dat_name.startswith('tdt'):
            if dat_name == 'tdt1':
                is_real, num_of_cls, num_features, num_data, tol = True, 30, 36771, 9394, 1e-5
            elif dat_name == 'tdt2':
                is_real, num_of_cls, num_features, num_data, tol = True, 30, 2000, 9394, 1e-5
            elif dat_name == 'tdt3':
                is_real, num_of_cls, num_features, num_data, tol = True, 30, 5000, 9394, 1e-5

        # generate the data and partition it
        data_gen = gen_and_distribute_data(is_real = is_real, data_name = dat_name, \
            data_dir = data_dir, num_of_features = num_features, num_of_samples = num_data, \
            num_of_source = num_of_cls, seed = dat_seed)
        print ('Data generated and partitioned ----------')

        # get the result directory where the result is stored
        res_dir = os.path.join(parent_dir, 'result', 'sncp', dat_name, \
                '_rho#' + str(rho) + '_nu#' + str(nu) + '_tol#' + str(tol)) 
        # we perform distributed clustering multiple times with different initiliations
        for seed_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        #for seed_num in [1]:
            seed_dir = os.path.join(res_dir, 'seed#' + str(seed_num))
            sncp(dat_gen = data_gen, res_dir = seed_dir, num_of_cls = num_of_cls, \
                    seed = seed_num, nu = nu, rho = rho, tol = tol, DEBUG = True)
    print ('Parallel FedCGds for clustering is done --------)')

if __name__ == '__main__':
    main()



