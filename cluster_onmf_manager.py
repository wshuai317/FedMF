################################################################################
# This script aims to design a class to manage the result generated by algorithms
# for NMF or ONMF model.
#
# @date 2019.08.10
# @author Wang Shuai
#################################################################################

from __future__ import division
import collections
import numpy as np
from numpy import linalg as LA
from cluster_metrics import *
import csv
import os
from result_manager import ResultManager

class ClusterONMFManager(ResultManager):
    """ This class is a child class of Result_manager which is explicitly designed
    for NMF of ONMF model for clustering. Specifically, in addition to primal variable
    dual variable, the change of variables, and feasbile conditions, this class
    also manipulate the cluster metrics and how to save these results.

    Attributes:
        root_dir (string): the root directory to store results
        ACC (dict): the dict of iteration-accurary pair
        Purity (dict): the dict of iteration-purity pair
        NMI (dict): the dict of iteration-nmi pair
        ARI (dict): the dict of iteration-ari pair
    """
    def __init__(self, root_dir, save_pdv = False):
        """ __init__ method to initialize all fields

        Args:
            root_dir (string): a given directory containing all results
            save_pdv (boolean): the flag indicating whether to save primal and dual varialbes to csv files
        Returns:
            None
        """
        self.root_dir = root_dir
        #self.pdv_to_csv = pdv_to_csv
        super(ClusterONMFManager, self).__init__(self.root_dir, save_pdv)

        self.ACC = collections.OrderedDict() # the clustering accurary
        self.Purity = collections.OrderedDict() # the purity
        self.NMI = collections.OrderedDict() # normalized mutual information
        self.ARI = collections.OrderedDict() # adjusted rand index
        self.iters_used = collections.OrderedDict() # the iters used for each penalty
        self.time_all = collections.OrderedDict() # the time lasts after each iter
        self.time_comm = collections.OrderedDict() # the time used for communication in each iter


    def push_W(self, val = None):
        """ This function save the value of W to the last position of self.prim_var['W']
        Note that the reason why the function name is 'push' is that we only get and set
        W at the end of self.prim_var['W'], which is the same as the push and pop
        operations on the stack.

        Args:
            val (numpy array or mat): the value of W
        Returns:
            None
        """
        if val is None:
            raise ValueError('Error: the input W is None!')
        else:
            self.add_prim_value('W', val)


    def push_H(self, val = None):
        """ This function save the value of H to the last position of self.prim_var['H']
        Note that the reason why the function name is 'push' is that we only get and set
        H at the end of self.prim_var['H'], which is the same as the push and pop
        operations on the stack.

        Args:
            val (numpy array or mat): the value of H
        Returns:
            None
        """
        if val is None:
            raise ValueError('Error: the input H is None!')
        else:
            self.add_prim_value('H', val)

    def push_iters(self, penalty = None, iters = None):
        """ This function save the pair of penalty-iters to the last position of self.iters_used

        Args:
            val (integer): the number of iters used
        Returns:
            None
        """
        if penalty is None or iters is None:
            raise ValueError('Error: the input is None!')
        else:
            self.iters_used[penalty] = iters

    def push_time(self, time_all = 0, time_comm = 0):
        """ This function saves the pair of iter-time to the last position of self.time_used

	Args:
	    time_used (float): time used after each iter
	Returns:
	    None
	"""
        iter_num = self.get_iter_num()
        self.time_all[iter_num] = time_all
        self.time_comm[iter_num] = time_comm


    def push_H_norm_ortho(self, val = None):
        """ This function save the value of ||HDD^{-1}H^T - I||_F where D is a diagonal
        matrix used to normalized each row of H.

        Args:
            val (float): the input value
        Returns:
            None
        """
        if val is None: # if no input, just compute it using the last H
            H = self.get_last_prim_val('H')
            H = np.asmatrix(H)
            (ha, hb) = H.shape
            norm_H = np.asmatrix(np.diag(np.diag(H * H.transpose()) ** (-0.5))) * H
            val = LA.norm(norm_H * norm_H.transpose() - np.eye(ha), 'fro') / (ha * hb)

        self.add_fea_condition_value('H_no', val)


    def peek_H_norm_ortho(self):
        """ This function return the computed normalized orthogonality of H at position
        len - 1

        Args:
            None
        Returns:
            float, the last value of self.fea_conditions['H_no']
        """
        return self.get_last_fea_condition_value('H_no')


    def push_W_norm_residual(self, val = None):
        """ This function saves the value of normalized residual of W:
                ||W^r - W^{r-1}||_F / ||W^{r-1}||_F
        at position len - 1

        Args:
            val (float): the input value
        Returns:
            None
        """
        if val is None: # if no input, just compute it using the last W
            W = self.get_last_prim_val('W')
            W_last = self.get_last_last_prim_val('W')
            val = LA.norm(W - W_last, 'fro') / LA.norm(W_last, 'fro')

        self.add_prim_change_value('W_nr', val)


    def push_H_norm_residual(self, val = None):
        """ This function saves the value of normalized residual of H:
                ||H^r - H^{r-1}||_F / ||H^{r-1}||_F
        at position len - 1

        Args:
            val (float): the input value
        Returns:
            None
        """
        if val is None: # if no input, just compute it using the last H
            H = self.get_last_prim_val('H')
            H_last = self.get_last_last_prim_val('H')
            val = LA.norm(H - H_last, 'fro') / LA.norm(H_last, 'fro')

        self.add_prim_change_value('H_nr', val)

    def peek_sncp_cost_residual(self):

        return self.get_last_cost_change('sncp_cost_change')

    def peek_onmf_cost_residual(self):
        return self.get_last_cost_change('onmf_cost_change')


    def peek_W_norm_residual(self):
        """ This function returns the computed normalized residual of W at pos
        len - 1

        Args:
            None
        Returns:
            float, the last value of self.prim_var_change['W_nr']
        """
        return self.get_last_prim_change('W_nr')

    def peek_H_norm_residual(self):
        """ This function returns the computed normalized residual of H at pos
        len -1

        Args:
            None
        Returns:
            float, the last value of self.prim_var_change['H_nr']
        """
        return self.get_last_prim_change('H_nr')


    def calculate_cluster_quality(self, true_labels = None):
        """ This function calculates the clustering performance based on
        the H and the following three metrics:
            1. Purity
            2. ARI
            3. Clustering Accuracy
            4. NMI

        Args:
            H (numpy array or mat): the input cluster indicator matrix
            true_labels (1-D array): the label of each data point
        Returns:
            None
        """
        if true_labels is None:
            raise ValueError('Error: no input true labels for comparison!')
        #if H is None: # if no input, just use the last H saved
        H = np.asmatrix(self.get_last_prim_val('H'))
        cls_assignments = np.asarray(np.argmax(H, axis = 0))[0, :] # get the cluster assignments
        iter_num = self.get_iter_num()

        #print (cls_assignments)
        # compute the clustering performance and store them
        self.Purity[iter_num] = calculate_purity(cls_assignments, true_labels)
        self.ARI[iter_num] = calculate_rand_index(cls_assignments, true_labels)
        self.ACC[iter_num] = calculate_accuracy(cls_assignments, true_labels)
        self.NMI[iter_num] = calculate_NMI(cls_assignments, true_labels)

        return self.ACC[iter_num]

    def peek_cluster_ACC(self):
        """ This function returns the recently-inserted items in self.ACC

        Args:
            None
        Returns:
            the recent inserted iter-ACC pair
        """
        return next(reversed(self.ACC.items()))

    def write_cluster_quality(self, file_path):
        """ This function saves time used and the clustering performance computed to a csv file
        with the following form
            iter_num    Time Purity  ARI ACC NMI
            XXXX        XXX     XXX   XXX XXX XXX
            ...
            ...
        Args:
            name_suffix (string): in case sometimes the name of the csv file should be added a suffix
        Returns:
            None
        """

        field_names = ['iter_num', 'Time_all', 'Time_comm', 'Purity', 'ARI', 'ACC', 'NMI']  # fill out the field names for CSV

        with open(file_path, mode = 'w', newline = '') as csv_file:  # open the file, if not exist, create it
            writer = csv.DictWriter(csv_file, fieldnames = field_names) # create a writer which maps the dictionaries onto output rows in CSV
            writer.writeheader() # write the field names to the header
            # save clustering performance
            for key in self.ACC.keys():
                temp_dict = collections.OrderedDict()
                temp_dict['iter_num'] = key
                if key in self.time_all.keys():
                    temp_dict['Time_all'] = self.time_all[key]
                else:
                    temp_dict['Time_all'] = 0  # no record
                if key in self.time_comm.keys():
                    temp_dict['Time_comm'] = self.time_comm[key]
                else:
                    temp_dict['Time_comm'] = 0  # no record
                temp_dict['Purity'] = self.Purity[key]
                temp_dict['ARI'] = self.ARI[key]
                temp_dict['ACC'] = self.ACC[key]
                temp_dict['NMI'] = self.NMI[key]
                writer.writerow(temp_dict)

    def write_iters(self, file_path):
        """ This function saves the penalty-iters pairs to a csv file
        with the following form
            penalty:rho  iters_used
            XXX          XXX
            ...          ...
        Args:
            f_path (string): the absolute path of the csv file
        Returns:
            None
        """
        field_names = ['penalty:rho', 'iters_used']  # fill out the field names for CSV

        with open(file_path, mode = 'w', newline = '') as csv_file:  # open the file, if not exist, create it
            writer = csv.DictWriter(csv_file, fieldnames = field_names) # create a writer which maps the dictionaries onto output rows in CSV
            writer.writeheader() # write the field names to the header
                                 # save clustering performance
            for key in self.iters_used.keys():
                temp_dict = collections.OrderedDict()
                temp_dict['penalty:rho'] = key
                temp_dict['iters_used'] = self.iters_used[key]
                writer.writerow(temp_dict)


    def write_to_csv(self, name_suffix = ''):
        """ This function saves all of the generated results to csv files

        Args:
            name_suffix (string): in case sometimes the name of the csv file should be added a suffix
        Returns:
            None
        """
        # At first, we call the super class method to save results such as primal and dual variables
        super(ClusterONMFManager, self).write_to_csv()
        file_path = os.path.join(self.root_dir, 'cls_quality' + str(name_suffix) + '.csv')
        self.write_cluster_quality(file_path)

        f_path = os.path.join(self.root_dir, 'iters_used' + str(name_suffix) + '.csv')
        self.write_iters(f_path)

