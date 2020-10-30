###########################################################################################
# The main entry script for parallel clustering which collaborates with several local
# nodes to perform clustering
#
# It is a couterpart of distributed_sncp.py but it just gives a sequential implementation
# of distributed sncp to simulate the process for testing
#
# @author Wang Shuai
# @date 2019.07.18
###########################################################################################

from __future__ import division

import numpy as np
import os
import time
from numpy import linalg as LA
# import two helper class
from data_generator import DataGenerator as DataGen # manage the data generation
from cluster_onmf_manager import ClusterONMFManager # manage the result generated

class Node(object):
    ''' The class is responsible for distributed computing in a local site (node). In particular,
    it provides functionality to H following SNCP method for the ONMF problem and
    communicate with the master node.

    Attributes:
        local_data (numpy array or mat): the data owned by a local site
        data_scale (float): normalized factor used to normalize the dataset
        #W (numpy array or mat): the primal variable W in SNCP
        H (numpy array or mat): the primal variable H in SNCP
        rho (float): the positive penalty parameter used in SNCP
    '''

    def __init__(self, data, data_num = 10000, rank = 4, init_W = None, init_H = None, \
            init_Z = None, init_rho = 1e-8, nu = 1e-4, mul = 0, para = 1e-6, \
            max_val = 100, seed = 1):
        ''' __init__ method to initialize all the fields

        Args:
            data (numpy array or mat): the input data array
            data_scale (float): the scale used to normalize the data
            rank (int): the rank of W and H for decomposition
	    init_rho (float): the initialization for the penalty parameter rho
	    nu (float): the setting of nu for (nu * ||H||_F^2)
            mul (float): the setting of mul for (mul * ||W||_F^2)
            sample (string): the method used to sample points to update W and the sample number
                        "XXX#XXX"
                    the sample method include:
                        1. 'full': all data points
                        2. 'uniform': uniformly sample data points
                        3 ...
            #par_idx (int): the partition of data sampling belonging to each local site
            comm (object): the communicator
            seed (int): the seed for random generator
        Returns:
            None
        '''
        self.local_data = data
        self.data_num = data_num
        self.rho = init_rho
        self.nu = nu
        self.para = para # the penalty parameter used in AL-like methods
        self.max_val  = max_val # the maximum value of the data
        self.seed = seed
        np.random.seed(seed)  # set the seed
        (da, db) = self.local_data.shape
        self.W, self.H = init_W, init_H

        # some intermediate values during computation
        (ha, hb) = self.H.shape
        self.I_ha = np.asmatrix(np.eye(ha))
        self.all_1_mat = np.asmatrix(np.ones((ha, ha)))

        # keep the gradient w.r.t W for update W with partial nodes
        self.HHT = self.H * self.H.transpose()
        self.XHT = self.local_data * self.H.transpose()
        

    def update_H_by_bcd(self, max_iter):
        ''' This function performs the update of H_p by solving
            H_p = \arg\min_{H_p \in \mathcal{H}_p} F_p(W_p, H_p)
        where F_p(W_p, H_p) = ||X_p - W_p * H_p||_F^2 / N 
                            + (0.5 * nu * ||H_p||_F^2 + 0.5 * rho * sum_{j=1}^{N_p}((1^Thj)^2 - ||hj||_F^2)) * (N / N_p)

        Args: 
            max_iter (int): the number of PGDs to be performed
        Returns
            H_p
        '''
        N_p = self.H.shape[1]
        Hessian = 2 * self.W.transpose() * self.W / N_p + \
                (self.rho * self.all_1_mat + (self.nu - self.rho) * self.I_ha) * self.data_num / N_p
        egenvals, _ = LA.eigh(Hessian)
        t = 0.51 * np.max(egenvals)
        H_pre = np.asmatrix(np.copy(self.H))
        for j in range(max_iter):  
            grad_H_pre = Hessian * H_pre - 2 * self.W.transpose() * self.local_data / N_p
            self.H = np.maximum(0, H_pre - grad_H_pre / t)
            H_pre = np.asmatrix(np.copy(self.H))
        print ('H update converges #' + str(j))

        end_time = time.time()
        return self.H


    def receive_W(self, W = None):
        ''' This function receives the current value of W from the master node

        Args:
            W (numpy array or mat): the value of W
        Returns:
            None
        '''
        self.W = np.asmatrix(np.copy(W))


    def send_W_grad_diff(self):
        ''' This function sends the difference of gradient w.r.t W over two consenctive iterations
        to the master node

        Args:
            None
        Returns:
            the gradient
        '''
        HHT = self.H * self.H.transpose()
        XHT = self.local_data * self.H.transpose()
        diff_HHT, diff_XHT = HHT - self.HHT, XHT - self.XHT
        self.HHT, self.XHT = HHT, XHT

        return diff_HHT, diff_XHT


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
        par_method = 'uniform', num_of_pars = 5, seed = 10):
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
    if data_name.startswith('mnist') and par_method == 'similarity': 
        pass
    else:
        data_gen.partition_data_array(num_of_pars = num_of_pars, m_name = par_method) # partition the data

    if not data_gen.existed_or_not():
        raise ValueError('Error: the synthetic data is not generated before. So please ensure \
                the data just generated has been prepared for local nodes and then run again!')

    return data_gen

def fedcgds_with_partial_nodes(dat_gen = None, res_dir = 'res', num_of_cls = 10, \
        num_of_locs = 100, seed = 1, nu = 1e-4, rho = 1e-8, selection = 'uniform', \
        num_of_active_nodes = 100, H_max_iter = 100, W_max_iter = 100, DEBUG = False):
    ''' This function peforms the distributed clustering using the SNCP method on the ONMF model
    Note that its computation is not done in distriubted setting. The function just simulate the
    distributed clustering in one node.

    Specifically, it first update each local H one node by one node through proximal gradient
    Then, the update of W is performed by proximal gradient with averaged gradient obtained from
    each local computation.

    Args:
        data_gen (object): the data generator
        res_dir (string): the res_dir where the generated results are stored
        num_of_cls (int): the number of clusters
        num_of_locs (int): the number of local nodes
        seed (int): the seed for random generator when
                    1. init for W and H
                    2. sample data points when update W
        nu (float): the setting for parameter \nu in sncp
        update (string): the method used to update W and H
                        1. 'palm'
                        2. 'bcd'
        sample (string): the method to sample points for updating W and the number of data points to be sampled
                    Note that the string follows the format "XXX#XXX": the sample method plus the sample number
        DEBUG (boolean): the flag to indicate whether print the debug info
    Returns:
        None
    '''
    # get the data, labels, and partitions
    data_mat, true_labels = dat_gen.get_data_and_labels()
    par_idx = dat_gen.get_partitions() # get the partition index

    # init W and H
    W_init, H_init = dat_gen.gen_inits_WH(seed = seed)
    Z_init = np.asmatrix(np.zeros(W_init.shape)) # the init for the dual variable in each local node

    n_factor = LA.norm(data_mat, 'fro') ** 2 # we normalize the ||X - WH||_F^2 so that the initial
                                             # value of rho (penalty) is suitable for all datasets
    data_num = data_mat.shape[1]

    print ('n_factor: ' + str(n_factor))

    # the setting for the penalty 
    rho = rho * n_factor / data_num 
    nu = nu * n_factor / data_num
    mul = 0

    max_val, min_val = np.max(data_mat), np.min(data_mat)


    # for each node, we create a node manager which is responsible for local computation
    local_nodes = []
    for node_id in range(num_of_locs):
        local_data_idx = np.where(par_idx == node_id)[0] # get the index of data samples belong to local site i
        local_data = data_mat[:, local_data_idx]
        local_node = Node(data = local_data, data_num = data_num, rank = num_of_cls, \
                init_rho = rho, init_W = W_init, init_H = H_init[:, local_data_idx], init_Z = Z_init, \
                nu = nu, mul = mul, max_val = max_val, seed = seed)
        local_nodes.append(local_node)

    # we construct a result manager to manage and save the result
    res_manager = ClusterONMFManager(root_dir = res_dir, save_pdv = False) # get an instance of ClusterONMFManager to manage the generated result
    res_manager.push_W(W_init)  # store W
    res_manager.push_H(H_init)  # store H
    res_manager.push_H_norm_ortho()  # store feasibility
    res_manager.add_cost_value('onmf_cost', get_onmf_cost(data_mat, n_factor, W_init, H_init, nu, 0)) # store obj val
    res_manager.add_cost_value('sncp_cost', get_sncp_cost(data_mat, n_factor, W_init, H_init, nu, 0, rho))
    res_manager.calculate_cluster_quality(true_labels) # calculate and store clustering quality
    
    '''
    Then we simulate the distributed computation which collaborates with
    local nodes to perform SNCP. Specifically, iteratively, we do
        1. the local node receive H and do its local computation
        2. the local node sample data points to caluclate the partial gradient w.r.t W
        3. the local node sends the gradient and the master node update W
    '''
    full_W_grad = np.asmatrix(np.zeros((data_mat.shape[0], num_of_cls)))
    W_cur, W_pre = np.asmatrix(np.copy(W_init)), np.asmatrix(np.copy(W_init))

    # In distributed setting, the master node actually cannot access the value of H.
    # But we also get the value of H from local nodes in the implementation
    # since we want to test whether the implementation is correct or not.
    # We will delete it in distributed setting.
    H_cur, H_pre = np.asmatrix(np.copy(H_init)), np.asmatrix(np.copy(H_init))
    local_idx_list = []
    for node_id in range(num_of_locs):
        local_idx_list.append(np.where(par_idx == node_id)[0])

    # keep two intermediate variables used to obtain full_W_grad later
    # However, in distributed setting, we cannot get the total dataset. So
    # we need one round of a communication to obtain themn
    HHT = H_pre * H_pre.transpose()
    XHT = data_mat * H_pre.transpose()

    sncp_iter = 1
    while True:
        # firstly, we need to select a set of m nodes used for update
        # There are two kinds of selection strategy currently
        #   1. random selection with a given probability
        #   2. select m nodes with best cost improvement

        if selection == 'uniform':
            select_idx = np.random.choice(num_of_locs, size = int(num_of_active_nodes), replace = False)
            # update H locally by all active nodes
            for node_id in select_idx:
                node = local_nodes[node_id]
                node.receive_W(W_pre)
                H_tmp = node.update_H_by_bcd(H_max_iter)
                H_cur[:, local_idx_list[node_id]] = H_tmp
                HHT_diff, XHT_diff = node.send_W_grad_diff()
                HHT += HHT_diff
                XHT += XHT_diff
        else:
            raise ValueError('no other selection strategy is supported!')
        
        if DEBUG:
            #onmf_cost = master_node.get_onmf_cost()
            sncp_cost = get_sncp_cost(data_mat, n_factor, W_pre, H_cur, nu, mul, rho)
            print ('SNCP_Iter(' + str(sncp_iter) + '), after update H, sncp_cost = ' + str(sncp_cost))

        # update W
        egenvals, _ = LA.eigh(HHT)
        full_max_eval = np.max(egenvals)
        t = 0.501 * full_max_eval
        W_tmp = np.asmatrix(np.copy(W_pre))
        for j in range(W_max_iter):
            full_W_grad = W_pre * HHT - XHT
            W_cur = np.minimum(max_val, np.maximum(0, W_pre - full_W_grad / t))
            W_pre = np.asmatrix(np.copy(W_cur))
        print ('W converge #' + str(j))

        W_change = LA.norm(W_cur - W_tmp, 'fro') / LA.norm(W_tmp, 'fro')

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

        print ('W_change: ' + str(W_change))

        if W_change < 1e-8 or sncp_iter > 500: # converges
            print ('Converges------')
            res_manager.write_to_csv()  # store the generated results to csv files
            break

        # update
        W_pre = np.asmatrix(np.copy(W_cur))
        H_pre = np.asmatrix(np.copy(H_cur))
        sncp_iter += 1
        print ('rho: ' + str(rho))

def main():
    ''' This function starts the running
    '''
    print ('Begin----------')

    cwd = os.getcwd()   # get current directory
    parent_dir = os.path.abspath(os.path.join(cwd, os.pardir)) # get the parent directory
    data_dir = os.path.join(parent_dir, 'data')  # get the data directory where the data is stored
    num_of_pars = 100
    dat_seed = 0
    nu = 1e-10
    rho = 1e-8
    #para = 1e-6
    selection = 'uniform'

    #for dat_name in ['snr-3', 'minist_bal', 'minist_unbal']:
    for dat_name in ['snr-3']:
        for par_m in ['similarity']:
        #for par_m in ['uniform']:
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
                elif dat_name == 'tcga9':
                    num_of_cls, num_features, num_data = 20, 2000, 5314
                elif dat_name == 'tcga11':
                    num_of_cls, num_features, num_data = 20, 5000, 5314
                else:
                    num_of_cls, num_features, num_data = 10, 1290, 3353
            elif dat_name.startswith('tdt'):
                is_real = True
                if dat_name == 'tdt1':
                    num_of_cls, num_features, num_data, tol = 30, 36771, 9394, 2e-5
                elif dat_name == 'tdt2':
                    num_of_cls, num_features, num_data, tol = 30, 2000, 9394, 1e-6
                elif dat_name == 'tdt3':
                    num_of_cls, num_features, num_data, tol = 30, 5000, 9394, 1e-5
                elif dat_name == 'tdt4':
                    num_of_cls, num_features, num_data, tol = 30, 8000, 9394, 1e-5


            # generate the data and partition it
            data_gen = gen_and_distribute_data(is_real = is_real, data_name = dat_name, \
                    data_dir = data_dir, num_of_features = num_features, num_of_samples = num_data, \
                    num_of_source = num_of_cls, par_method = par_m, num_of_pars = num_of_pars, seed = dat_seed)
            print ('Data generated and partitioned ----------')

            for s_num in [10]: # the number of active clients
            #for s_num in [100]:
                #for H_iter in [10, 100]: # the number of epochs for W update
                for H_iter in [100]:
                    for W_iter in [1, 20, 50, 100]: # the number of epochs for H update
                    #for W_iter in [100]:
                        # get the result directory where the result is stored
                        res_dir = os.path.join(parent_dir, 'result', 'fedcgds', dat_name,  par_m + '#' + \
                                str(num_of_pars), selection + '#' + str(s_num), 'H#' + str(H_iter) + '_W#' + \
                                str(W_iter) + '_rho#' + str(rho) + '_nu#' + str(nu)) 
                        # we perform distributed clustering multiple times with different initiliations
                        for seed_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                        #for seed_num in [1]:
                            seed_dir = os.path.join(res_dir, 'seed#' + str(seed_num))
                            fedcgds_with_partial_nodes(dat_gen = data_gen, res_dir = seed_dir, num_of_cls = num_of_cls, \
                                    num_of_locs = num_of_pars, seed = seed_num, nu = nu, rho = rho, \
                                    selection = selection, num_of_active_nodes = s_num, H_max_iter = H_iter, \
                                    W_max_iter = W_iter, DEBUG = True)
    print ('Parallel FedCGds for clustering is done --------)')

if __name__ == '__main__':
    main()



