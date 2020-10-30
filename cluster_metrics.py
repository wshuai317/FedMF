#############################################################################################
# The script provides functions to evaluate clustering performance given a cluster assignment
#
#
# @author Wang Shuai
# @date 2019.05.30
#############################################################################################

from __future__ import division
import numpy as np

from sklearn.utils import linear_assignment_
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn import manifold

def calculate_purity(cluster_assignments, true_classes):
    """ This function is to calculate the purity, a measurement of quality for the clustering results.
    Each cluster is assigned to the true class which is most frequent in the cluster.
    Using these classes, the percent accuracy is then calculated

    Args:
        cluster_assignments (numpy array): an array contains cluster ids indicating the clustering
                 assignment of each data point with the same order in the data set

        true_classes (numpy array): an array contains class ids indicating the true labels of each
                            data point with the same order in the data set

    Returns:
        A number between 0 and 1. Poor clusterings have a purity close to 0 while a perfect clustering has a purity of 1
    """
    # get the set of unique cluster ids
    cluster_ids = np.unique(cluster_assignments)
    cluster_indices = {}

    # find out the index of data points for each cluster in the data set
    for cls_id in cluster_ids:
        cluster_indices[cls_id] = np.where(cluster_assignments == cls_id)[0]
    # find out the index of data points for each class in the data set
    class_ids = np.unique(true_classes)
    class_indices = {}
    for cla_id in class_ids:
        class_indices[cla_id] = np.where(true_classes == cla_id)[0]

    # find out the true class which is most frequent for each cluster
    # record the number of correct classfications
    num_accuracy = 0
    for cls_id in cluster_ids:
        #max_id = class_ids[0]
        max_count = 0
        for cla_id in class_ids:
            tmp = len(np.intersect1d(cluster_indices[cls_id], class_indices[cla_id]))
            if max_count < tmp:
                max_count = tmp
        num_accuracy = num_accuracy + max_count
    return float(num_accuracy) / len(cluster_assignments)


def calculate_rand_index(cluster_assignments, true_classes):
    """ This function is to calculate the Rand Index, a measurement of quality for the clustering results.
    It is essentially the percent accuracy of the clustering.

    The clustering is viewed as a series of decisions. There are N * (N-1) / 2
    pairs of samples in the dataset to be considered. The decision is considered
    correct if the pairs have the same label and are in the same cluster, or have
    different labels and are in different clusters. The number of correct decisions
    divided by the total number of decisions givens the Rand index

    Args:
        cluster_assignments (numpy array): an array contains cluster ids indicating the clustering
                  assignment of each data point with the same order in the data set

        true_classes (numpy array): an array contains class ids indicating the true labels of each
                  data point with the same order in the data set

    Returns:
	A number between 0 and 1. Poor clusterings have a purity close to 0 while a perfect clustering has a purity of 1
    """
    '''
    correct = 0
    total = 0
    # itertools.combinations will return all unordered pairs of indices of data points(0->N-1)
    for index_combo in itertools.combinations(range(len(true_classes)), 2):
        index1 = index_combo[0]
        index2 = index_combo[1]
        same_class = (true_classes[index1] == true_classes[index2])
        same_cluster = (cluster_assignments[index1] == cluster_assignments[index2])
        if same_class and same_cluster:
            correct = correct + 1
        elif not same_class and not same_cluster:
            correct = correct + 1
        else:
            pass # no nothing
        total = total + 1
    return float(correct) / total
    '''
    return adjusted_rand_score(cluster_assignments, true_classes)



def calculate_accuracy(cluster_assignments, true_classes):
    """
    The function calculate the clustering accurary which use the ratio of correctly
    clustered points over the total number of points (in [0, 1], the higher the better)

        AC = sum_{i from 1 to N}   delta(si, map(ri))   / N

    where N is the total number of documents and delta(x, y) is the delta function
    that equals to one if x = y and 0 otherwise. ri and si are the obtained cluster
    label and the true label for the i-th data sample. Map(ri) is the permutation
    mapping function that maps each cluster label ri to the equivalent label in true labels.

    Args:
        cluster_assignments (numpy array): an array contains cluster ids indicating the clustering
                    assignment of each data point with the same order in the data set

        true_classes (numpy array): an array contains class ids indicating the true labels of each
                    data point with the same order in the data set

    Returns:
	A number between 0 and 1. Poor clusterings have a purity close to 0 while a perfect clustering has a purity of 1
    """

    ca = best_map(true_classes, cluster_assignments)
    #print 'best map'
    #print ca
    return accuracy_score(ca, true_classes)


def best_map(L1, L2):
    """ The function is to compute the best mapping between two partitions

    For example, if one partition is  1 1 1 2 2 4  and another is 2 2 2 3 3 1
    Therefore the best mapping is 1-2, 2-3, 4 -1. And after the mapping,
    the labels of the first partition will be 2 2 2 3 3 1.

    Args:
	L1 (numpy array): the first partition
	L2 (numpy array): the second partition
    Returns:
	the labels of L1 after mapping
    """
    if L1.__len__() != L2.__len__():
        print('size(L1) must == size(L2)')

    Label1 = np.unique(L1)
    nClass1 = Label1.__len__()
    Label2 = np.unique(L2)
    nClass2 = Label2.__len__()

    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[i][j] = np.nonzero((L1 == Label1[i]) * (L2 == Label2[j]))[0].__len__()
    c = linear_assignment_.linear_assignment(-G.T)[:, 1]
    #c = linear_sum_assignment(-G.T)[:, 1]
    #print (c)
    #raise ValueError('Error: just stop!')

    newL2 = np.zeros(L2.__len__())
    for i in range(nClass2):
        for j in np.nonzero(L2 == Label2[i])[0]:
            if len(Label1) > c[i]:
                newL2[j] = Label1[c[i]]
    return newL2


def calculate_NMI(cluster_assignments, true_classes):
    """ The function is to calculate NMI (the normalized mutual information) metric.

    Let C denote the set of clusters obtained from the ground truth and C' obtained
    from an algorithm. Their mutual information metric MI(C, C') is defined as follows:

    MI(C, C') = sum_{ci in C, cj' in C') p(ci, cj') * log2 (p(ci, cj') /(p(ci)p(cj')))

    where p(ci) and p(cj') are the probabilities that a data sample arbitrarily selected
    from the data set belongs to the clusters ci and cj', respectively, and p(ci, cj')
    is the joint probability that the arbitrarily selected data sample belongs to the
    clusters ci as well as cj' at the same time.

    Then the NMI is calculated as:

           NMI(C, C') = MI(C, C') / max(H(C), H(C'))

    where H(C) and H(C') are the entropies of C and C', respectively. It is easy to
    check that NMI(C, C') ranges from 0 to 1. NMI = 1 if two sets of clusters are identical,
    and NMI = 0 if the two sets are independent.

    Args:
        cluster_assignments (numpy array): an array contains cluster ids indicating the clustering
                assignment of each data point with the same order in the data set

        true_classes (numpy array): an array contains class ids indicating the true labels of each
                data point with the same order in the data set

    Returns:
	A number between 0 and 1.
    """

    return adjusted_mutual_info_score(cluster_assignments, true_classes)


def visualize_data(data_arr, dim = 2, partition_idx = None, dat_path = None, \
        xlabel = '', ylabel = '', title = ''):
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

    #tsne = manifold.TSNE(n_components = dim)
    #dat_embeded = tsne.fit_transform(data_arr)
    dat_embeded = data_arr
    colormap = np.array(['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', \
            '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', \
            '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', \
            '#000000'])
    if np.max(partition_idx) < len(colormap):
        color_used = colormap[list(map(int, partition_idx))]
    else:
        color_used = partition_idx
    plt.scatter(dat_embeded[:, 0], dat_embeded[:, 1], c = color_used)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(dat_path, bbox_inches = 'tight')


#def visualize_data(data_arr, dim = 2, partition_idx = None, dat_path = None)
    """ This function is used to visulize the data points in a 2-D space. Specifically, it
    will plot these data points in a figure after dimension-reduction with T-SNE. The
    data points belonging to the same partition will be plotted with the same color.

    At first, we will check the number of features of input data. If the number of features (dim)
    > 3, we will consider it as a high-dimensional data and plot it after dimension reduction = 2.
    If the number of features = 3, we will plot it in 3D.

    In order to visulize data in low dimension, we utilize the hypertools library which is designed
    to facilitate dimensionality reduction-based visual explorations of high-dimensional data:
        1."Heusser AC, Ziman K, Owen LLW, Manning JR (2018) HyperTools: A python toolbox for gaining
        geometric insights inot high-dimensional data. JMLR, 18(152): 1--6."
        2. https://github.com/ContextLab/hypertools.

    It is very convenient to use with simple functions to plot high-dimensional datasets in 2/3D.
    Args:
        data_arr (numpy array): the data array to be plotted with each column being a data point
        dim (int): the dimension on which the data points to be plotted, default: 2
        partition_idx (list): a list of indexes indicating the partition of these data points
        dat_path (string): an absolute file path to store the 2D figure
    Returns:
        None
    """
    """
    num_of_features = data_arr.shape[0]
    if dim > num_of_features:
        raise ValueError('Error: plot low-dimensional data into a high-dimensional space!')
    else:
        if num_of_features > 3: dim = 2  # only when the number of features = 3, we plot data points in 3D
        else: dim = num_of_features
    """
    '''
    f_path = os.path.join(self.root_dir, '1.pdf')  # create the file path with existed parent directory
    # use hypertools library plot function to plot and save
    # fig = hyp.plot(data_arr.transpose(), '.', hue = partition_idx, save_path = f_path)
    fig = hyp.plot(data_arr.transpose(), ".", reduce = 'TSNE', ndims = dim, hue = partition_idx, save_path = f_path)
    '''

