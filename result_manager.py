##########################################################################
# This script is to design a class which can manage the result of an
# optimization algorithm
#
# @author Wang Shuai
# @date 2019.05.29
##########################################################################
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os.path
import pandas as pd
from numpy import linalg as LA
import csv
import collections
from file_manager import FileManager
from numpy import linalg as LA

class ResultManager(FileManager):
    """ Provide convient functionality to generate, save and present results

    Attributes:
        obj (float): the objective values of the target problem model
	prim_var (dict): the values of primal varibles during execution of the algorithm
	dual_var (dict): the values of dual variables if any (when solving the dual problem)
	prim_var_change (dict): the values indicating how primal variables change
	dual_var_change (dict): the values indicating how dual variables change if any
	fea_conditions (dict): the values indicating the changes of feasiblity level

        pdv_to_csv (boolean): the flag indicating whether the primal (dual) variables will be saved to csv files,
                            especially in case that the primal (dual) varialbe is a large array or matrix

                            The reason why we want to save the primal or dual variables to individual csv files
                            is that the primal (dual ) variables may be large array or matrices which consume
                            lots of memory space
    """
    def __init__(self, root_dir, pdv_to_csv = False):
        """ __init__ method to initialize all fields

        Args:
            root_dir (string): a given directory containing all results
            pdv_to_csv (boolean): the flag indicating whether to save primal and dual varialbes to csv files
        Returns:
	    None
        """
        self.root_dir = root_dir
        self.pdv_to_csv = pdv_to_csv
        super(ResultManager, self).__init__(self.root_dir)
        self.clear() # clear the root _dir
        self.reset()

    def reset(self):
        """ This function is used to make all the variables empty for storing results
        There are the variables used to evaluate the optimization algorithms

        Args:
            None
        Returns:
            None
        """

        self.cost = {}   # record cost value at each iteration
        self.cost_change = {} # record the change of cost items
        self.prim_var = {}   # record primal variable values for each iteration
        self.prim_var_change = {}  # record the change of primal variable between two consective iterations
        self.dual_var = {}   # record dual variable values for each iteration
        self.dual_var_change = {} # record the change of dual variable between any two consective iterations
        self.fea_conditions = {} # record the satisfication of feasiblity conditions at each iteration


    def get_iter_num(self):
        """ This function returns the iteration number of the algorithm.
	Specifically, the iteration number is calculated based on the number
	of cost values stored. Since the first cost value is usually
	compuated based on the initializations for the algorithms, we treat
	the iteration number is only less than the number of cost values
	by 1.

        Args:
	    None
        Returns:
            iteration number (int)
        """
        first_key = list(self.cost.keys())[0]
        return len(self.cost[first_key]) - 1

    def add_cost_value(self, var_name, val):
        """ This function receives and add a cost value
        Args:
            val (float): cost value
        Returns:
            None
        """
        self.add_other_value(self.cost, var_name, val)

        # add the change of cost automatically
        if len(self.cost[var_name]) > 1:  # if the cost is not empty
             last_val = self.get_last_last_cost_val(var_name)
             cost_change = abs(val - last_val) / last_val
             self.add_other_value(self.cost_change, var_name + '_change', cost_change)


    def add_prim_value(self, var_name, val):
        """ This function adds a variable-value pair for primal variables
        Args:
	    var_name (string): the name of a primal variable
            val (??): the value of the primal variable
        Returns:
            None
        """
        if not self.pdv_to_csv: # if not save to csv file, just store the value
            self.add_other_value(self.prim_var, var_name, val)
        else: # otherwise, we save the value to a csv file and add the file pos to self.prim_var[var_name]
            if not var_name in self.prim_var.keys(): # if the var is not established in the dictionary
                self.prim_var[var_name] = []
            pos = len(self.prim_var[var_name])
            f_path = os.path.join(self.root_dir, 'prim_vars', var_name, str(pos) + '.csv')
            self.add_file(f_path)   # we should create the file at first
            np.savetxt(f_path, np.asmatrix(val), delimiter = ',')
            self.prim_var[var_name].append(pos)

    def add_prim_change_value(self, var_name, val):
        """ This function adds the change of a primal variable, eg. 'H': ||H - H'||/||H'||
        Args:
	    var_name (string): the name of the primal variable of the change
            val (??): the value of change
        Returns:
	    None
        """
        self.add_other_value(self.prim_var_change, var_name, val)

    def add_dual_value(self, var_name, val):
        """ This function adds a variable-value pair for dual variables
        Args:
	    var_name (string): the name of a dual variable
            val (??): the value of the dual variable
        Returns:
	    None
        """
        if not self.pdv_to_csv: # if not save to csv file, just store the value
            self.add_other_value(self.dual_var, var_name, val)  # add the pair to self.dual_var
        else: # otherwise, we save the value to a csv file and add file pos to self.dual_var[var_name]
            if not var_name in self.dual_var.keys(): # if the var is not in the dict
                self.dual_var[var_name] = []
            pos = len(self.dual_var[var_name])
            f_path = os.path.join(self.root_dir, 'prim_vars', var_name, str(pos) + '.csv')
            self.add_file(f_path)  # we should create the file at first
            np.savetxt(f_path, np.asmatrix(val), delimiter = ',')
            self.dual_var[var_name].append(pos)

    def add_dual_change_value(self, var_name, val):
        """ This function adds the change of a dual variable, eg. 'Z': ||Z - Z'||/||Z'||
	Args:
	    var_name (string): the name of the dual variable of the change
            val (??): the value of change
        Returns:
	    None
        """
        self.add_other_value(self.dual_var_change, var_name, val)

    def add_fea_condition_value(self, var_name, val):
        """ This function adds the value indicating feasiblility of a condition,
	    Eg. H - S = 0,   ||H-S||/ (the number of elements in H)
        Args:
	    var_name (string): the name of a feasible condition
	    val (float): the value
        Returns:
	    None
        """
        self.add_other_value(self.fea_conditions, var_name, val)

    def add_other_value(self, otherObj, var_name, val):
        """ This function adds the varialbe-value pair to a given result variable
	Args:
	    otherObj (dict or list): the instance of a given result variable
            var_name (string): the name of a variable
	    val (??): the value of the variable
	Returns:
	    the position of the list where the val is stored
        """
        if not var_name in otherObj.keys(): # if the var is not established in the dictionary
            otherObj[var_name] = []

        otherObj[var_name].append(np.copy(val))  # very important to use np.copy !!!!!
        return len(otherObj[var_name]) - 1

    def get_cost_val(self, var_name, pos):
        """ This function get and return the cost value stored at a given position
	Args:
	    pos (int): the position (0~ len(obj) -1)
        Returns:
	    float
        """
        return self.get_other_value(self.cost, var_name, pos)


    def get_prim_val(self, var_name, pos):
        """ This function get and return the value of a primal variable at a given position
	Args:
	    var_name (string): the name of the primal varialbe
	    pos (int): a given position
	Returns:
	    a value
        """
        val = self.get_other_value(self.prim_var, var_name, pos)
        if not self.pdv_to_csv: # if not saved to csv file
            return val
        else: # otherwise, we should get the file path and read through the file
            f_path = os.path.join(self.root_dir, 'prim_vars', var_name, str(val) + '.csv')
            df = pd.read_csv(f_path, header = None) # first read csv data file into a pandas dataframe
                                                       # since it is conventient for us to use
            data_mat = np.asmatrix(df.values)
            return data_mat

    def get_dual_val(self, var_name, pos):
        """ This function get and return the value of a dual variable at a given position
        Args:
            var_name (string): the name of the primal varialbe
            pos (int): a given position
        Returns:
            a value
        """
        val = self.get_other_value(self.dual_var, var_name, pos)
        if not self.pdv_to_csv: # if not saved to csv file
            return val
        else: # otherwise, we should get the file path and read from the file to array or mat
            f_path = os.path.join(self.root_dir, 'dual_vars', var_name, str(val) + '.csv')
            df = pd.read_csv(f_path, header = None) # first read csv file into a pandas data frame and then transform
            return np.asmatrix(df.values)

    def get_last_cost_val(self, var_name):
        """ This function get and return the last cost value
	Args:
	    None
	Returns:
	    float
        """
        if not var_name in self.cost.keys():
            raise ValueError('Error:' + var_name + ' are not stored in the keys')
        pos = len(self.cost[var_name]) - 1
        return self.get_cost_val(var_name, pos)

    def get_last_last_cost_val(self, var_name):
        """ This function get and return the cost at position of len -2
        Args:
            var_name (string): the name of the cost
        Returns:
            float
        """
        if not var_name in self.cost.keys():
            raise ValueError('Error:' + var_name + ' are not stored in the keys')
        l = len(self.cost[var_name])
        if l < 2: raise ValueError('Error: out of length!')
        return self.get_cost_val(var_name, l - 2)

    def get_last_prim_val(self, var_name):
        """ This function get and return the last value of a primal variable
	Args:
	    var_name (string): the name of the primal variable
	Returns:
	    a value
        """
        if not var_name in self.prim_var.keys():
            raise ValueError('Error:' + var_name + ' are not stored in the keys')
        pos = len(self.prim_var[var_name]) - 1
        return self.get_prim_val(var_name, pos)

    def get_last_last_prim_val(self, var_name):
        """ This function get and return the value of a primal variable at position
	 of len - 2
	Args:
	    var_name (string): the name of the primal variable
	Returns:
	    a value
	"""
        if not var_name in self.prim_var.keys():
            raise ValueError('Error:' + var_name + ' are not stored in the keys')
        l = len(self.prim_var[var_name])
        if l < 2: raise ValueError('Error: out of length')
        return self.get_prim_val(var_name, l - 2)

    def get_last_dual_val(self, var_name):
        """ This function get and return the last value of a dual variable
	Args:
	    var_name (string): the name of the dual variable
	Returns:
	    a value
        """
        pos = len(self.dual_var[var_name]) - 1
        return self.get_dual_val(var_name, pos)

    def get_last_cost_change(self, var_name):
        """ This function get and return the last value of the change of objective value

        Args:
            var_name (string): the name of the change of cost 
        Returns:
            float
        """
        pos = len(self.cost_change[var_name]) - 1
        return self.get_cost_change_value(var_name, pos)

    def get_last_prim_change(self, var_name):
        """ This function get and return the last value of the change of a primal variable
	Args:
	    var_name (string): the name of the change of a primal variable
	Returns:
	    float
        """
        pos = len(self.prim_var_change[var_name]) - 1
        return self.get_prim_change_value(var_name, pos)

    def get_last_dual_change(self, var_name):
        """ This function get and return the last value of the change of a dual variable
        Args:
	    var_name (string): the name of the change of a dual variable
	Returns:
	    float
        """
        pos = len(self.dual_var_change[var_name]) - 1
        return self.get_dual_change_value(var_name, pos)

    def get_last_fea_condition_value(self, var_name):
        """ This function get and return the last value of a feasible condition
        Args:
	    var_name (string): the name of a feasible condition
	Returns:
	    float
        """
        pos = len(self.fea_conditions[var_name]) - 1
        return self.get_fea_condition_value(var_name, pos)


    def get_cost_change_value(self, var_change_name, pos):
        """ This function get and return the value of a cost item at a given position
        Args:
            var_change_name (string): the name of the change of a cost item
            pos (int): the position
        Returns:
            float
        """
        return self.get_other_value(self.cost_change, var_change_name, pos)

    def get_prim_change_value(self, var_change_name, pos):
        """ This function get and return the value of the change of a primal variable
	at a given position
        Args:
	    var_change_name (string): the name of the change of a primal variable
	    pos (int): the position
	Returns:
	    float
        """
        return self.get_other_value(self.prim_var_change, var_change_name, pos)

    def get_dual_change_value(self, var_change_name, pos):
        """ This function get and return the value of the change of a dual variable
	at a given position
        Args:
	    var_change_name (string): the name of the change of a dual variable
	    pos (int): the position
	Returns:
	    float
        """
        return self.get_other_value(self.dual_var_change, var_change_name, pos)

    def get_fea_condition_value(self, fea_name, pos):
        """ This function get and return the value of a feasible condition
	at a given position
        Args:
	    fea_name (string): the name of the feasible condition
	    pos (int): the position
	Returns:
	    float
        """
        return self.get_other_value(self.fea_conditions, fea_name, pos)

    def get_other_value(self, otherObj, var_name, pos):
        """ This function get and return the value of a given result variable
	at a given position
        Args:
	    var_name (string): the name of the result variable
	    pos (int): the position
	Returns:
	    a value
        """
        if not var_name in otherObj.keys():
            raise ValueError('Error:' + var_name + ' are not stored in the keys')
        if pos < 0 or pos > len(otherObj[var_name]) - 1:
            raise ValueError('Error: the index out of range when accessing ' + var_name)
        return otherObj[var_name][pos]

    def save_last_prims(self):
        """ This function save the last primal variables to csv files
        Args:
            None
        Returns:
            None
        """
        for var_name in self.prim_var.keys(): # for each var
            pos = len(self.prim_var[var_name]) - 1
            var_val = self.get_prim_val(var_name, pos)
            f_path = os.path.join(self.root_dir, 'prim_vars', var_name, str(pos) + '.csv')
            self.add_file(f_path)   # we should create the file at first
            np.savetxt(f_path, np.asmatrix(var_val), delimiter = ',')



    def write_to_csv(self, name_suffix = ''):
        """ This function save the following numerical results to one csv file like
		"XX"    "XX"     "XX"    "XX" ...
	     1   000    000      000     000  ...
	     2   000    000      000     000  ...
	     ...
	The fields to be saved includes
		1. the objective values,
		2. the change of primal variables if any
	        3. the change of dual variables if any
	        4. the feasible conditions

        Args:
            name_suffix (string): in case sometimes the name of the csv file should be added a suffix
        Returns:
            None
        """
        f_path = os.path.join(self.root_dir, 'res' + name_suffix + '.csv')
        field_names = []  # the first field in CSV is 'obj_val'

        # put the keys in the cost, prim_var_change, dual_var_change and fea_conditions as field names if any
        for key in self.cost.keys():
            field_names.append(key)
        for key in self.cost_change.keys():
            field_names.append(key)
        for key in self.prim_var_change.keys():
            field_names.append(key)
        for key in self.dual_var_change.keys():
            field_names.append(key)
        for key in self.fea_conditions.keys():
            field_names.append(key)

        with open(f_path, mode = 'w', newline = '') as csv_file:  # open the file, if not exist, create it
            writer = csv.DictWriter(csv_file, fieldnames = field_names) # create a writer which maps the dictionaries onto output rows in CSV
            writer.writeheader() # write the field names to the header
            temp_dict = {}   # create a temporary dict used to output rows
            row_max = self.get_iter_num()  # get the max iters which indicates the number of rows in CSV
            print ('number of rows: ' + str(row_max))
            #print (field_names)
            for row in range(row_max + 1):
                temp_dict.clear()   # clear all items
                start_idx = 0
                for i in range(len(self.cost)):
                    field = field_names[start_idx + i]
                    temp_dict[field] = self.get_cost_val(field, row)

                start_idx = start_idx + len(self.cost)  # the start pos of fields in field_names for prim_var_change
                for i in range(len(self.cost_change)): # for each cost_change
                    field = field_names[start_idx + i]
                    if row == 0:  # for row 0 (iter 0), we will set '/' to the change of primal variables
                        temp_dict[field] = '/'
                    else:
                        temp_dict[field] = self.get_cost_change_value(field, row - 1)


                start_idx = start_idx + len(self.cost_change)
                for i in range(len(self.prim_var_change)): # for each prim_var_change
                    field = field_names[start_idx + i]
                    if row == 0:  # for row 0 (iter 0), we will set '/' to the change of primal variables
                        temp_dict[field] = '/'
                    else:
                        temp_dict[field] = self.get_prim_change_value(field, row - 1)

                start_idx = start_idx + len(self.prim_var_change)  # go to the start pos of fields in field_names for dual_var_change
                for i in range(len(self.dual_var_change)): # for each dual_var_change
                    field = field_names[start_idx + i]
                    if row == 0: # for row 0 (iter 0), we will set '/' to the change of dual variables
                        temp_dict[field] = '/'
                    else:
                        temp_dict[field] = self.get_dual_change_value(field, row - 1)

                start_idx = start_idx + len(self.dual_var_change) # go the the start pos of fields in field_names for fea_conditions
                for i in range(len(self.fea_conditions)):  # for each fea_condition
                    field = field_names[start_idx + i]
                    temp_dict[field] = self.get_fea_condition_value(field, row)

                writer.writerow(temp_dict)

        # we also save the value of primal values if not saved
        if not self.pdv_to_csv:
            self.save_last_prims()



if __name__ == "__main__":
    '''
    m = ResultManager("/home/ubuntu/work/res")

    for i in range(100):
        val1 = np.random.randn()
        m.add_obj_value(val1)
        m.add_prim_change_value('W_nr', i)
        m.add_prim_change_value('H_nr', i + 3)

    m.write_to_csv()
    '''
    import re
    match = re.match(r"([a-z]+)([0-9]+)" ,'foofo21', re.I)
    if match:
        items = match.groups()
        print (items)
    print (items[0])
    print (items[1])
    #m.add_file("/home/ubuntu/work/Test/test.txt")
    #m.add_file("test1.csv")
    #m.add_file("pty/cluster/pp.png")
    #m.delete_file('/home/ubuntu/work/Test/test.txt')
    #m.delete_content_of_dir("")






