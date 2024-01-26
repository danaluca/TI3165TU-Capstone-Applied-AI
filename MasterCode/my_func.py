# All outside functions written for the purpose of implementing the modularity-based clustering proposed by Newman and
# applying it to chaotic systems
# Original paper by Newman:
# Newman, M. E. J. (2006). Modularity and community structure in networks.
# Proceedings of the National Academy of Sciences of the United States of America, 103(23), 8577–82.
# https://doi.org/10.1073/pnas.0601602103
#
# Urszula Golyska 2022

import numpy as np
import scipy.sparse as sp
import networkx as nx
import graphviz as gv
from modularity_maximization import spectralopt
import matplotlib.pyplot as plt
import numpy.linalg
import csv

def find_extr_paths_loop(P,local_path, cluster_from, ready_paths, extr_clusters):
    ''' Inner function for finding loops leading to given extreme cluster

    :param P: sparse deflated probability matrix
    :param local_path: local path leading to extreme cluster (first element is the extreme cluster)
    :param cluster_from: cluster from which we will continue looking into the probability matrix
    :param ready_paths: vector of ready paths, when a full circle had been made (from extreme cluster y back to extreme
    cluster y or any other cluster that is already on the path)
    :param extr_clusters: vector of extreme clusters
    :return: vector of ready paths (tuples)
    '''
    next_clusters = P.col[P.row == cluster_from]    # look at all the possible clusters as the next step on the path
    next_clusters = np.delete(next_clusters, np.where(next_clusters==cluster_from))  # exclude looping inside oneself

    for next_cluster in next_clusters:  # look at all paths
        if next_cluster not in extr_clusters:   # if the next cluster is not extreme
            loc_local_path = local_path     # reset
            if next_cluster not in loc_local_path:  # if next cluster is not already on the path
                loc_local_path.append(next_cluster)  # append and go deeper
                find_extr_paths_loop(P, loc_local_path, next_cluster,ready_paths, extr_clusters)
            else:
                if tuple(loc_local_path) not in ready_paths:    # if we this path is not in the ready paths yet - add to vector of all paths
                    ready_paths.append(tuple(loc_local_path))
        else:       # if the next cluster is extreme
            if tuple(local_path) not in ready_paths:  # if we this path is not in the ready paths yet - add to vector of all paths
                ready_paths.append(tuple(local_path))
    return ready_paths

def find_extr_paths(extr_clusters,P):
    '''Outer function for finding loops leading to given extreme cluster

    :param extr_clusters: vector of extreme clusters
    :param P: sparse deflated probability matrix
    :return: returns vector of ready paths (tuples) for all extreme clusters
    '''

    final_paths = list()
    for extr_cluster in extr_clusters:  # for all extreme clusters

        clusters_from = P.col[P.row==extr_cluster] # look at all the possible clusters as the next step on the path
        clusters_from = np.delete(clusters_from,np.where(clusters_from==extr_cluster))   # exclude looping inside oneself

        for cluster_from in clusters_from:
            if cluster_from not in extr_clusters:   # if next cluster is not extreme
                local_path = [extr_cluster, cluster_from]  # start/restart path
                ready_paths = []

                find_extr_paths_loop(P, local_path, cluster_from, ready_paths, extr_clusters)     # find paths to given extreme cluster
                final_paths.extend(ready_paths)
    return final_paths

def prob_to_extreme(cluster_nr,paths, T, P, clusters):
    '''Function to find the maximum probability, minimum average time  and shortest path to an extreme event

    :param cluster_nr: index of cluster we are currently looking at
    :param paths: vector of ready paths (tuples) for all extreme clusters
    :param T: maximum time of data series
    :param P: sparse deflated probability matrix
    :param clusters: all defined clusters with their properties
    :return: return maximum probability, minimum average time and shortest path to an extreme event for the given cluster_nr cluster
    '''
    prob = 0
    time = T
    length = np.size(P)

    if clusters[cluster_nr].is_extreme ==2: # if the cluster is extreme
        prob = 1    # all values are irrelevant
        time = 0
        length = 0
    else:
        for i in range(len(paths)):     # for all paths
            loc_prob = 1
            loc_time = 0
            loc_path = np.asarray(paths[i])
            if cluster_nr in loc_path:     # find path with our cluster
                # take into account only part of path to our cluster
                loc_end = np.where(loc_path==cluster_nr)[0]
                loc_end = loc_end[0]
                loc_path = loc_path[0:loc_end+1]

                for j in range(len(loc_path)):  # loop through the whole length of the path
                    if j!=len(loc_path)-1:  # skip last step
                        temp = P.data[P.col[P.row == loc_path[j]]] # find probability of transitioning from cluster j to ,cluster j+1 on the path
                        temp = temp[P.col[P.row == loc_path[j]]==loc_path[j+1]]
                        loc_prob = loc_prob*temp    # multiply probabilities on path

                    if j!=0:   # exclude first and last path
                        loc_time += clusters[loc_path[j]].avg_time  # add average time in clusters to total time to extreme

                if loc_prob > prob: # if the current path has a maximum probability
                    prob = loc_prob
                if loc_time < time: # if the current path has a minimum time
                    time = loc_time
                if len(loc_path)-1<length: # if the current path has a shortest path
                    length = len(loc_path)-1
    return prob,time,length

class cluster(object):
    '''Object cluster, defined by it's number (id), the nodes that belong to it, it's center, the clusters to and from which
        it transitions'''
    def __init__(self, nr, nodes, center_coord,center_coord_tess,avg_time, nr_instances, P, extr_clusters, from_clusters):
        self.nr = nr    # cluster index (staring from 0)
        self.nodes = nodes  # node/ hypercube - ids in tessellated space

        is_extreme = 0  # normal state cluster
        if nr in extr_clusters:
            is_extreme=2    # extreme cluster
        elif nr in from_clusters:
            is_extreme=1  # precursor cluster

        self.is_extreme = is_extreme

        clusters_to = P.row[P.col==nr]   # clusters to which the current cluster can transition
        clusters_from = P.col[P.row==nr]     # clusters which can transition to current cluster

        self.transition_to = clusters_to
        self.transition_from = clusters_from

        # cluster center in phase space and tessellated phase space
        self.center = center_coord
        self.center_tess = center_coord_tess

        # average time spent in cluster
        self.avg_time = avg_time

        # number of times the cluster appears in given time series
        self.nr_instances = nr_instances

def plot_cluster_statistics(clusters, T, min_prob=None, min_time=None, length=None):
    ''' Function for plotting cluster statistics, both for systems than exhibit and do not exhinit extreme events

    :param clusters: all defined clusters with their properties
    :param T: last time of data series
    :param min_prob: maximum probability of transitioning to an extreme event, default is None (for systems without extreme events)
    :param min_time: minimum average time of transitioning to an extreme event, default is None (for systems without extreme events)
    :param length: shortest path to an extreme event, default is None (for systems without extreme events)
    :return: none, plots statistics and saves them
    '''
    numbers = np.arange(len(clusters))
    color_pal = ['#1f77b4'] * (max(cluster.nr for cluster in clusters)+1)   # default blue

    if min_prob is not None: # if extreme events are present
        # set color palette
        is_extreme = np.array([cluster.is_extreme for cluster in clusters])
        # set colors for cluster types
        for i in range(len(is_extreme)):
            if is_extreme[i]==2:
                color_pal[i] = '#d62728'    # red
            elif is_extreme[i]==1:
                color_pal[i] = '#ff7f0e'    # orange

        # Probability of transitioning to extreme event
        min_prob[min_prob==1] = 0 # change the probability in extreme clusters to zero so they don't appear in the plots
        fig, ax = plt.subplots(figsize=(12,8))
        ax.bar(numbers, min_prob, color=color_pal)
        ax.grid('minor')
        temp_labels = ax.containers[0]
        for p in temp_labels:
            x, y = p.get_xy()
            w, h = p.get_width(), p.get_height()
            if h != 0:  # anything that have a height of 0 will not be annotated
                ax.text(x + 0.5 * w, y + h, '%0.2e' % h, va='bottom', ha='center')
        # ax.bar_label(ax.containers[0], label_type='edge')
        ax.set_xlabel("Cluster", fontsize=20)
        ax.set_ylabel("Probability of transitioning to extreme", fontsize=20)
        plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.99)
        plt.savefig('stat_prob.pdf')

        # Minimum average time to extreme event
        fig, ax = plt.subplots(figsize=(12,8))
        ax.bar(numbers, np.round(min_time,2), color=color_pal)
        ax.grid('minor')
        temp_labels = ax.containers[0]
        for p in temp_labels:  # skip the last patch as it is the background
                x, y = p.get_xy()
                w, h = p.get_width(), p.get_height()
                if h != 0:  # anything that have a height of 0 will not be annotated
                    ax.text(x + 0.5 * w, y + h, '%0.2f' % h, va='bottom', ha='center')
        # ax.bar_label(temp_labels, label_type='edge')
        ax.set_xlabel("Cluster", fontsize=20)
        ax.set_ylabel("Average time to extreme [s]", fontsize=20)
        plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.99)
        plt.savefig('stat_time.pdf')

        # Shortest path to extreme event
        fig, ax = plt.subplots(figsize=(12,8))
        ax.bar(numbers, length, color=color_pal)
        ax.grid('minor')
        temp_labels = ax.containers[0]
        for p in temp_labels:  # skip the last patch as it is the background
            x, y = p.get_xy()
            w, h = p.get_width(), p.get_height()
            if h != 0:  # anything that have a height of 0 will not be annotated
                ax.text(x + 0.5 * w, y + h, '%0.0f' % h, va='bottom', ha='center')
        # ax.bar_label(ax.containers[0], label_type='edge')
        ax.set_xlabel("Cluster", fontsize=20)
        ax.set_ylabel("Length of shortest path to extreme", fontsize=20)
        plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.99)
        plt.savefig('stat_path.pdf')

    # For all chaotic systems

    # Average time spent in cluster
    fig, ax = plt.subplots(figsize=(12,8))
    ax.bar(numbers, [np.round(cluster.avg_time,2) for cluster in clusters], color=color_pal)
    ax.grid('minor')
    ax.bar_label(ax.containers[0], label_type='edge')
    ax.set_xlabel("Cluster", fontsize=20)
    ax.set_ylabel("Average time spent in cluster [s]", fontsize=20)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.99)
    plt.savefig('stat_circ_time.pdf')

    # Percentage of total time spent in cluster
    fig, ax = plt.subplots(figsize=(12,8))
    ax.bar(numbers, [np.round((cluster.avg_time*cluster.nr_instances)/T*100,3) for cluster in clusters], color=color_pal)
    ax.grid('minor')
    ax.bar_label(ax.containers[0], label_type='edge')
    ax.set_xlabel("Cluster", fontsize=20)
    ax.set_ylabel("% time spent in cluster", fontsize=20)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.99)
    plt.savefig('stat_percent_time.pdf')

    # Cluster size (number of nodes)
    fig, ax = plt.subplots(figsize=(12,8))
    ax.bar(numbers, [cluster.nodes.size for cluster in clusters], color=color_pal)
    ax.grid('minor')
    ax.bar_label(ax.containers[0], label_type='edge')
    ax.set_xlabel("Cluster", fontsize=20)
    ax.set_ylabel("# nodes in cluster", fontsize=20)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.99)
    plt.savefig('stat_nr_nodes.pdf')

    # Number of instances in whole data frame
    fig, ax = plt.subplots(figsize=(12,8))
    ax.bar(numbers, [cluster.nr_instances for cluster in clusters], color=color_pal)
    ax.grid('minor')
    ax.bar_label(ax.containers[0], label_type='edge')
    ax.set_xlabel("Cluster", fontsize=20)
    ax.set_ylabel("# instances in time series", fontsize=20)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.99)
    plt.savefig('stat_nr_instances.pdf')

    # write all statistics to csv file
    with open('cluster_stat.csv', 'w') as file:
        writer = csv.writer(file)
        if min_prob is not None:    # extreme events present
            csv_header = ['nr', 'avg_time_in_cluster', 'percent_time_in_cluster', 'nr_nodes', 'nr_instances', 'is_extreme', 'prob_to_extreme', 'avg_time_to_extreme', 'path_to_extreme']
            writer.writerow(csv_header)
            for i in numbers:
                csv_data = [i, clusters[i].avg_time, (clusters[i].avg_time*clusters[i].nr_instances)/T*100, clusters[i].nodes.size, clusters[i].nr_instances, is_extreme[i], min_prob[i], min_time[i], length[i]]
                writer.writerow(csv_data)
        else:   # no extreme events
            csv_header = ['nr', 'avg_time_in_cluster', 'percent_time_in_cluster', 'nr_nodes', 'nr_instances']
            writer.writerow(csv_header)
            for i in numbers:
                csv_data = [i, clusters[i].avg_time, (clusters[i].avg_time * clusters[i].nr_instances) / T * 100, clusters[i].nodes.size, clusters[i].nr_instances]
                writer.writerow(csv_data)
    return 1

def backwards_avg_time_to_extreme(is_extreme,dt):
    ''' For analyzing time series - Finds average time of transitioning from the first instance of precursor cluster to
     extreme cluster, looking backwards in time from extreme event

    :param is_extreme: vector defining which clusters are extreme (value 2) and which are precursors (value 1)
    :param dt: time step
    :param clusters: all defined clusters with their properties
    :return: returns value of the average time from entering a precursor stage to the occurrence of an extreme event',
    number or instances of extreme events with a precursor, number of instances of extreme events and number of instances of precursors
    '''

    extreme_events_t = np.where(is_extreme==2)[0]   # vector of all extreme clusters
    precursors_t = np.where(is_extreme==1)[0]   # vector of all precursor clusters
    instances_extreme_with_precursor=0
    time = 0
    instances_extreme_no_precursor=0
    instances_precursor_no_extreme=0
    instances_precursor_after_extreme=0

    for i in range(len(extreme_events_t)-1):    # loop through all extreme events
        # we are looking only at the first one step of each instance
        # isolate first case
        if i==0 or (extreme_events_t[i+1]==extreme_events_t[i]+1 and extreme_events_t[i-1]!=extreme_events_t[i]-1):
            temp_ee_t = extreme_events_t[i] # time step of first extreme step
            if is_extreme[temp_ee_t-1]!=1:
                instances_extreme_no_precursor+=1
            # look at its precursors
            for j in range(len(precursors_t)):
                if precursors_t[j]+1 == temp_ee_t: # find the instance that precedes the current extreme event
                    k=j   # loop backwards to find its first step
                    while k>=0:
                        if precursors_t[k-1] != precursors_t[k]-1 or k==0:  # we have found the end
                            temp_prec_t = precursors_t[k]
                            instances_extreme_with_precursor += 1
                            time += (temp_ee_t - temp_prec_t) * dt  # add time between the two transitions to total time
                            break
                        k-=1
                    break

    for i in range(len(precursors_t)-1):    # loop through all extreme events
        # we are looking only at the last step of each instance
        # isolate last case
        if i==len(precursors_t)-1 or (precursors_t[i-1]==precursors_t[i]-1 and precursors_t[i+1]!=precursors_t[i]+1):
            temp_prec_t = precursors_t[i] # last step of precursor step
            if is_extreme[temp_prec_t+1]!=2:
                instances_precursor_no_extreme+=1

        # look at first instance to see if there is an extreme event before
        if i==len(precursors_t)-1 or (precursors_t[i+1]==precursors_t[i]+1 and precursors_t[i-1]!=precursors_t[i]-1):
            temp_prec_t_first = precursors_t[i] # first step of precursor step
            if is_extreme[temp_prec_t_first-1]==2:
                instances_precursor_after_extreme+=1

    avg_to_extreme = time/instances_extreme_with_precursor  # divide total time by number of instances

    return avg_to_extreme, instances_extreme_with_precursor, instances_extreme_no_precursor, instances_precursor_no_extreme, instances_precursor_after_extreme

def calculate_statistics(extr_dim, clusters, P, T):
    ''' Function for calculating the extreme event statistics and plotting them

    :param extr_dim: dimensions used for the definition of extreme events
    :param clusters: all defined clusters with their properties
    :param P: sparse deflated transition probability matrix
    :param T: maximum time of data series
    :return: none, creates new variables used for plotting and statistics calculations
    '''
    if np.size(extr_dim)>0:    # extreme events are present
        extr_clusters = np.empty(0, int)
        for i in range(len(clusters)):  # loop through all clusters
            loc_cluster = clusters[i]
            if loc_cluster.is_extreme == 2:    # if current cluster is extreme
                extr_clusters = np.append(extr_clusters, i)        # append its id to vector of extreme clusters

        paths = find_extr_paths(extr_clusters, P)   # find all paths to extreme clusters

        min_prob = np.zeros((len(clusters)))
        min_time = np.zeros((len(clusters)))
        length = np.zeros((len(clusters)))

        for i in range(len(clusters)):  # loop through all clusters
            loc_prob, loc_time, loc_length = prob_to_extreme(i, paths, T, P, clusters)  # calculate extreme statistics of current cluster
            # append results to respective vectors
            min_prob[i] = loc_prob
            min_time[i] = loc_time
            length[i] = loc_length

        plot_cluster_statistics(clusters, T, min_prob, min_time, length)
    else:   # no extreme events present
        plot_cluster_statistics(clusters, T)
    return min_prob, min_time

def tesselate(x,N,ex_dim,nr_dev=7):
    """ Tessellate trajectory defined by data points x in phase space defined by N spaces in each direction

    :param x: vector of point phase space coordinates in consequent time steps
    :param N: number of discretizations of the phase space in each direction
    :param ex_dim: dimensions used for identifying the extreme events
    :param nr_dev: scalar defining how far away from the mean (in multiples of the standard deviation) will be considered an extreme event
    :return: returns matrix tess_ind which includes the indices of the hypercubes of the subsequent data points and the
    index of the identified extreme event
    """
    dim = int(np.size(x[0,:])) # number of dimensions of the phase space
    y = np.zeros_like(x)
    tess_ind = np.empty((0,dim), dtype=int)  # matrix od indices of sparse matrix

    for i in range(dim):
        y[:,i] = np.divide((x[:,i]-min(x[:,i])),abs(max(x[:,i])-min(x[:,i])))   # rescaling in all dimensions to [0,1]

    # Tessellate phase space - starting from the lower left corner
    for k in range(np.size(x[:,0])): # loop through all data points
        point_ind = np.floor(y[k,:]*N).astype(int)  # vector of indices of the given point in all dimensions, rounding down
        point_ind[point_ind==N] = N-1   # for all points located at the end (max) - move them to the last cell
        tess_ind = np.vstack([tess_ind, point_ind])   # translate the points into the indices of the tesselation
        # (to get the tessellated space, just take the unique rows of tess_ind)
    m = np.zeros_like(ex_dim,dtype=float)
    dev = np.zeros_like(ex_dim,dtype=float)

    # Find tessellated indices of extreme events
    for i in range(len(ex_dim)):    # loop through all dimensions used to define the extreme events
        loc_ex_dim = ex_dim[i]
        m[i] = np.mean(x[:, loc_ex_dim])   # mean of current dimension
        dev[i] = np.std(x[:,loc_ex_dim])    # standard deviation of current dimension

        # Extreme event - if it is within >=nr_dev dev away from the mean
        if i==0:    # first extreme dimension
            temp = abs(x[:,ex_dim[i]])>=m[i]+nr_dev*dev[i] # define extreme event as nr_dev the standard deviation away from the mean
        else:   # for other dimensions - combine this condition with the ones coming from the previous dimensions
            temp2 = np.logical_and((temp==True), abs(x[:, ex_dim[i]]) >= m[i] + nr_dev * dev[i])

    if len(ex_dim)>0:   # if extreme events are present
        if len(ex_dim) >= 2:
            extr_id = np.unique(tess_ind[temp2,:],axis=0)   # take only unique rows to get rid of repetitions
        else:
            extr_id = np.unique(tess_ind[temp,:], axis=0)
    else:   # extreme events are not present
        extr_id=[]
    return tess_ind, extr_id    # returns indices of occupied spaces and the indices of the identified extreme event (if present)

def tess_to_lexi(x,N,dim):
    """Translated tessellated space of any dimensions to lexicographic order

    :param x: array of tessellated phase space coordinates of the system's trajectory
    :param N: number of discretsations in each direction
    :param dim: dimensions of the phase space
    :return: returns 1d array of tessellated space
    """
    x2 = np.zeros_like(x)
    for i in range(dim):  # loop through all phase space dimensions
        # if x.size>dim:    # for more than one data point
        x2[:,i]=x[:,i]*N**i # contribution of i-th dimension to lexicographic tessellation id
        # else:
        #     x2[i] = x[i]*N**i

    if x.size>dim:    # for more than one data point
        x_trans = np.sum(x2[:,:dim], axis=1)    # find lexicographic tessellation id
    else:
        x_trans = np.sum(x2[:dim])

    return x_trans

def prob_to_sparse(P,N, extr_id):
    """"Translates the transition probability matrix of any dimensions into a Python sparse 2D matrix

    :param P: probability transition matrix as described in trans_matrix
    :param N: number of tessellation sections in each direction
    :param extr_id: indices in the tessellated phase space of the identified extreme events
    :return: returns Python (scipy) sparse coordinate 2D matrix and the translated indices of the extreme events
    """
    dim = int((np.size(P[0, :])-1)/2)  # number of phase space dimensions

    data = P[:,-1]  # store probability data in separate vector and delete it from the probability matrix
    P = np.delete(P, -1, axis=1)

    # translate points into lexicographic order
    row = tess_to_lexi(P[:,:dim],N, dim)
    col = tess_to_lexi(P[:, dim:], N, dim)

    if len(extr_id)!=0: # if extreme events are present
        extr_trans = tess_to_lexi(np.array(extr_id), N, dim)
    else:   # no extreme events present
        extr_trans=0

    P = sp.coo_matrix((data, (row, col)), shape=(N**dim, N**dim)) # create sparse matrix

    return P, extr_trans    # returns sparse probability matrix with points in lexicographic order and the extreme event point

def community_aff(P_com_old, P_com_new, N, dim, type, printing):
    """Creates a community affiliation matrix D, in which each node or old cluster is matched with the new cluster they
    were assigned to

    :param P_com_old: clustered old community P
    :param P_com_new: refined and reclustered community P
    :param N: number of tessellation sections in each direction
    :param dim: number of phase space dimensions
    :param type: 'first' or 'iteration', defines whether we are clustering nodes ('first') or clusters ('iteration')
    :param printing: bool parameter if the communities and their nodes should be printed on screen
    :return: returns a dense Dirac matrix of the affiliation of points to the identified clusters
    """
    nr_com_new = int(np.size(np.unique(np.array(list(P_com_new.values())))))    # number of new communities

    if type=='iteration':
        nr_com_old = int(np.size(np.unique(np.array(list(P_com_old.values())))))    # number of old communities
        D = np.zeros((nr_com_old, nr_com_new)) # number of old communities by number of new communities
    elif type=='first':
        D = np.zeros((N ** dim, nr_com_new))  # number of points by number of communities
    if printing:
        print('Total number of new communities: ', nr_com_new)

    for com in np.unique(np.array(list(P_com_new.values()))):   # loop through new communities
        if printing:
            print("Community: ", com)
            print("Nodes: ", end='')
        if type=='iteration':
            for key, value in P_com_old.items():    # loop through all old communities
                if value == com:    # if current old community belongs to current new community com
                    if printing:
                        print(key, end=', ')
                    D[value,com] = 1  # prescribe old community to new community
        elif type=='first':
            for key, value in P_com_new.items():  # loop through all new communities
                if value == com:
                    if printing:
                        print(key, end=', ')  # print nodes in the community
                    D[key, value] = 1  # prescribe nodes to new community
        if printing:
            print('')
    return D

def community_aff_sparse(P_com_old, P_com_new, N, dim, type, printing):
    """ Creates a sparse community affiliation matrix D, in which each node or old cluster is matched with the new cluster they
    were assigned to

    :param P_com_old: clustered old community P
    :param P_com_new: refined and reclustered new community P
    :param N: number of tessellation sections in each direction
    :param dim: number of dimensions of the system in phase space
    :param type: 'first' or 'iteration', defines whether we are clustering nodes ('first') or clusters ('iteration')
    :param printing: bool parameter if the communities and their nodes should be printed on screen
    :return: returns a sparse Dirac matrix of the affiliation of points to the identified clusters
    """
    D = np.empty((0,3), dtype=int)  # matrix of indices of sparse matrix
    nr_com_new = int(np.size(np.unique(np.array(list(P_com_new.values())))))

    for com in np.unique(np.array(list(P_com_new.values()))):   # loop through new communities
        if printing:
            print("Community: ", com)
            print("Nodes: ", end='')
        for key, value in P_com_new.items():  # loop all old communities (keys)
            if value == com:    # if the current old community belongs to current new community
                if printing:
                    print(key, end=', ')  # print nodes in the community
                row = [key, value, 1]  # prescribe nodes to new community
                D = np.vstack([D, row])
        if printing:
            print('')

    if type=='iteration':
        nr_com_old = int(np.size(np.unique(np.array(list(P_com_old.values())))))
        D_sparse = sp.coo_matrix((D[:, 2], (D[:,0], D[:,1])), shape=(nr_com_old, nr_com_new))
    elif type=='first':
        D_sparse = sp.coo_matrix((D[:,2], (D[:,0],D[:,1])), shape=(N ** dim, nr_com_new))
    if printing:
        print('Total number of new communities: ', nr_com_new)

    return D_sparse

def to_graph(P):
    """ Translates a transition probability matrix into graph form

    :param P: transition probability matrix (directed with weighted edges)
    :return: returns graph version of matrix P
    """
    P = P.transpose()   # due to the different definition of the P matrix - for matric form it is P[to_nodes, from_node],
    # for graph form - P[from_node,to_node]

    P_graph = nx.DiGraph()
    for i in range(len(P[:, 0])):   # loop through all rows
        for j in range(len(P[0, :])):   # loop through all columns
            if P[i, j] != 0:    # if non-zero probability of transitioning
                P_graph.add_edge(i, j, weight=P[i, j])
    return P_graph

def plot_graph(P_graph, labels, type):
    """Function for plotting the graph representation of the probability matrix

    :param P_graph: graph form of the probability matrix
    :param labels: bool property defining whether the labels should  be displayed
    :param type: defines type of considered system, function can be modified to fit the best representation for the currently considered system
    :return: none, plots graph representation of probability matrix
    """
    # Visualize graph
    plt.figure()
    if type=='sine':
        nx.draw_kamada_kawai(P_graph, with_labels=labels)
    else:
        # nx.draw(P_graph,with_labels=True)
        nx.draw_spring(P_graph,with_labels=labels)

    return 1

def to_graph_sparse(P):
    """Translates a sparse transition probability matrix into graph form

    :param P: sparse transition proabbility matrix (directed with weighted edges)
    :return: returns graph version of matrix P
    """
    columns = P.row   # due to the different definition of the P matrix - for matric form it is P[to_nodes, from_node],
    # for graph form - P[from_node,to_node]
    rows = P.col
    data = P.data
    P_graph = nx.DiGraph()

    for i in range(len(columns)):   # for all sparse entries
        P_graph.add_edge(rows[i], columns[i], weight=data[i])

    return P_graph

def to_graph_gv(P):
    """Translates a sparse tranition probability matrix into graph form for the gv package used for large communities

    :param P: sparse transition probability matrix (directed with weighted edges)
    :return: returns graph version of matrix P
    """
    columns = P.row   # due to the different definition of the P matrix - for matric form it is P[to_nodes, from_node],
    # for graph form - P[from_node,to_node]
    rows = P.col
    data = P.data
    P_graph = gv.Digraph('G', filename='cluster.gv')

    for i in range(len(columns)):   # for all sparse entries
        P_graph.edge(str(rows[i]), str(columns[i]), label=str(data[i]))

    return P_graph

def probability(tess_ind, type):
    """Computes transition probability matrix of tessellated data in both the classic and the backwards sense (Schmid (2018)).

    :param tess_ind:  matrix including the indices of the hypercubes of the subsequent data points, can
    be obtained from the tesselate(x,N) function
    :param type: "classic" - traditional approach of calculating the probability of transitioning from state j to state i,
    "backwards" - calculating probability of having transitioned from point j when already in point i
    :return: returns sparse transition probability matrix P, where a row contains the coordinate of the point i to which
    the transition occurs, point j from which the transition occurs and the value of probability of the transition
    """
    # for type = 'backwards' node 1 is the node to which the transition is made, node 2 is the node from which the transition is made;
    # for type = 'classic', node 1 is the node from which the transition is made, node 2 is the node to which the transition is made

    dim = int(np.size(tess_ind[0, :]))  # number of dimensions of the phase space
    P = np.empty((0, 2 * dim + 1))  # probability matrix dim*2+1 for the value of the probability
                                    # P[0,:] = [to_index(dim), from_index(dim), prob_value(1)]
    u_1, index_1, counts_1 = np.unique(tess_ind, axis=0, return_index=True,
                                                return_counts=True)  # sorted hypercubes that are occupied at some point by the trajectory
    if type=='classic':
        corr_point = tess_ind[-1]   # correction point accounting for the last point
    elif type=='backwards':
        corr_point = tess_ind[0]    # correction point accounting for the first point

    for j in range(len(u_1[:, 0])):  # for each unique entry (each hypercube)
        point_1 = u_1[j]  # index of the point j (in current hypercube)
        denom = counts_1[j]  # denominator; for calculating the probability
        if (point_1==corr_point).all(): # if current point is the correction point
            denom=denom-1
        temp = np.all(tess_ind == point_1, axis=1)  # rows of tess_ind with point j
        if type=='classic':
            temp = np.append([False], temp[:-1])  # indices of the row just below (i); adding a false to the beginning
        elif type=='backwards':
            temp = np.append(temp[1:], [False])  # indices of the row just above (j); adding a false to the end
        u_2, index_2, counts_2 = np.unique(tess_ind[temp], axis=0, return_index=True,
                                              return_counts=True)  # sorted points occupied just before going to i

        for i in range(len(counts_2)):  # loop through all instances of i
            point_2 = u_2[i]
            if type=='classic':
                temp = np.append([[point_2], [point_1]], [counts_2[i] / denom])
            elif type=='backwards':
                temp = np.append([[point_1], [point_2]], [counts_2[i] / denom])
            P = np.vstack([P, temp])  # add row to sparse probability matrix

    return P

def plot_prob_matrix(P_dense):
    """Function for plotting probability matrix

    :param P_dense: dense representation of transition probability matrix
    :return: none, plots probability matrix
    """
    # Visualize probability matrix
    plt.figure(figsize=(7, 7))
    plt.imshow(P_dense,interpolation='none', cmap='binary')
    plt.colorbar()
    return 1

def extr_iden(extr_trans, D_nodes_in_clusters, P_old):
    """Function for Identifying extreme and precursor clusters

    :param extr_trans: hypercubes with the identified extreme events
    :param D_nodes_in_clusters: matrix of affiliation of point to the community clusters
    :param P_old: final version of transition probability matrix
    :return: returns tuple of the extreme cluster and its preceding precursor cluster
    """
    if type(extr_trans)==np.int32 or type(extr_trans)==int: # only one extreme hypercube
        extr_cluster = D_nodes_in_clusters.col[D_nodes_in_clusters.row == extr_trans]   # identify the cluster they belong to
        from_cluster = P_old.col[P_old.row == extr_cluster] # identify its precursor clusters
    else:
        extr_cluster=[]
        for point in extr_trans:    # for all extreme hypercubes
            loc_cluster =  int(D_nodes_in_clusters.col[D_nodes_in_clusters.row==point]) # identify the cluster they belong to
            if loc_cluster not in extr_cluster: # if the cluster hasn't been identified yet
                extr_cluster.append(loc_cluster)

        from_cluster = []
        for cluster in extr_cluster:    # for all extreme clusters
            from_cluster_loc = P_old.col[P_old.row==cluster]  # identify all precursor clusters
            for loc_cluster in from_cluster_loc:
                if loc_cluster not in from_cluster: # if precursor cluster hasn't been identified yet
                    from_cluster.append(loc_cluster)

    return (extr_cluster,from_cluster)

def clustering_loop(P_community_old, P_graph_old, P_old, D_nodes_in_clusters):
    """ Inside of the re-clustering loop (for a iterative process)

    :param P_community_old: dictionary of previous network affiliations (to be deflated)
    :param P_graph_old: previous network in graph form
    :param P_old: transition probability matrix of previous network (to be deflated)
    :param D_nodes_in_clusters: matrix of affiliation of points to the previously defined community clusters
    :return: returns a set of the same parameters, but after one optimization run, can be then directly fed to the
     function again to obtain more optimized results
    """

    P_community_new = spectralopt.partition(P_graph_old)  # New community; returns dictionary of new network affiliations
    D_new = community_aff(P_community_old, P_community_new, 0, 0, 'iteration', 1)  # new affiliation matrix

    # Deflate the Markov matrix
    P_new = np.matmul(np.matmul(D_new.transpose(), P_old.transpose()), D_new)
    # print(np.sum(P_new,axis=0).tolist())  # For checking purposes - should be approx.(up to rounding errors) equal to number of nodes in each cluster

    # Graph form
    P_graph_old = to_graph(P_new)
    P_community_old = P_community_new   # make the new network "old" for the next iteration
    P_old = P_new

    # Translation of which nodes belong to the new clusters - create new affiliation matrix D
    D_nodes_in_clusters = np.matmul(D_nodes_in_clusters, D_new)
    return P_community_old, P_graph_old, P_old, D_nodes_in_clusters

def clustering_loop_sparse(P_community_old, P_graph_old, P_old, D_nodes_in_clusters):
    """Inside of the re-clustering loop (for a iterative process); sparse version

    :param P_community_old: dictionary of previous network affiliations (to be deflated; sparse)
    :param P_graph_old: previous network in graph form (sparse)
    :param P_old: transition probability matrix of previous network (to be deflated; sparse)
    :param D_nodes_in_clusters: matrix of affiliation of points to the previously defined community clusters
    :return: returns a set of the same parameters, but after one optimization run, can be then directly fed to the
     function again to obtain more optimized results
    """

    P_community_new = spectralopt.partition(P_graph_old)  # New community; returns dictionary of new network affiliations
    D_new = community_aff_sparse(P_community_old, P_community_new, 0, 0, 'iteration', 1)  # new affiliation matrix

    # Deflate the Markov matrix
    P_new = sp.coo_matrix((D_new.transpose() * P_old) * D_new)
    # print(np.sum(P_new,axis=0).tolist())  # For checking purposes - should be approx.(up to rounding errors) equal to number of nodes in each cluster

    # Graph form
    P_graph_old = to_graph_sparse(P_new)
    P_community_old = P_community_new # make the new network "old" for the next iteration
    P_old = P_new

    # Translation of which nodes belong to the new clusters - create new affiliation matrix D
    D_nodes_in_clusters = sp.coo_matrix(D_nodes_in_clusters*D_new)
    return P_community_old, P_graph_old, P_old, D_nodes_in_clusters

def data_to_clusters(tess_ind_trans, D_nodes_in_clusters, x=[], *clusters):
    '''Translates data points to their affiliated cluster id

    :param tess_ind_trans: time series; already translated to tessellated lexicographic ordering
    :param D_nodes_in_clusters: matrix of affiliation of all points to communities
    :param x: data series of occupied phase space
    :param clusters: list of all identified clusters and their propoerties
    :return: returns vector of the time series with their cluster affiliation
    '''
    tess_ind_cluster = np.zeros_like(tess_ind_trans)
    for point in np.unique(tess_ind_trans):     # for all unique points in tessellated phase space
        if D_nodes_in_clusters.col[D_nodes_in_clusters.row==point].size>0:  # if the point was already present in the data series (and therefore is tessellated)
            cluster_aff = int(D_nodes_in_clusters.col[D_nodes_in_clusters.row==point])  # find affiliated cluster
        else:   # this point has not been encountered yet
            # Find the cluster with the closest center (in the untessellated phase space)
            y = x[np.where(tess_ind_trans==point),:]    # the point in physical space
            cluster_aff = find_closest_center(y[0,0,:],clusters[0])    # find the (one) closest point
        tess_ind_cluster[tess_ind_trans==point] = cluster_aff
    return tess_ind_cluster

def cluster_centers(x,tess_ind, tess_ind_cluster, D_nodes_in_clusters,dim):
    ''' Function for calculating the cluster centers in both physical and tessellated phase space

    :param x: data series
    :param tess_ind: matrix including the indices of the hypercubes of the subsequent data points, can
    be obtained from the tesselate(x,N) function
    :param tess_ind_cluster: vector of time series with their cluster affiliation
    :param D_nodes_in_clusters: matrix of affiliation of all the possible points to clusters
    :param dim: number of dimensions of the phase space
    :return: returns two vectors: cluster centers in phase space and in tessellated phase space
    '''
    coord_clust = np.zeros((D_nodes_in_clusters.shape[1], dim))
    for i in range(D_nodes_in_clusters.shape[1]):   # for each cluster
        coord_clust[i,:] = np.mean(x[tess_ind_cluster==i,:], axis=0)    # center in phase space is the average of the untessellated data series

    pts, indices = np.unique(tess_ind, return_index=True, axis=0)       # unique points in phase space
    coord_clust_tess = np.zeros((D_nodes_in_clusters.shape[1], dim))

    num_clust = np.zeros((D_nodes_in_clusters.shape[1], 1))
    for i in range(D_nodes_in_clusters.shape[1]):   # for each cluster
        num_clust[i]=np.sum(D_nodes_in_clusters.col==i) # count number of points in cluster

    for i in range(len(pts[:, 0])):  # for each unique point
        loc_clust = tess_ind_cluster[indices[i]]    # identify local cluster
        coord_clust_tess[loc_clust,:] += pts[i, :]

    for i in range(D_nodes_in_clusters.shape[1]):  # for each cluster
        if num_clust[i]!=0:
            coord_clust_tess[i,:] = coord_clust_tess[i,:] / num_clust[i] # center in tessellated phase space is the average of the tessellated data series
        else:
            print('Overlapping clusters (due to tessellation)') # this is possible when using a clustering algorithm
                # that does not require tessellating the phase space beforehand (such as the k-means); and two or more identified
                # clusters fall into the same tessellated hypercube
            coord_clust_tess[i,:] =[0,0]    # artificial!!
    return coord_clust, coord_clust_tess

def plot_phase_space(x, type):
    """Function for plotting the data in phase space - for the six chaotic systems

    :param x: data series
    :param type: string defining the type of system ("MFE_burst"; "MFE_dissipation"; "sine"; "LA"; "CDV"; "PM"; "kolmogorov"; "kolmogorov_k_D")
    :return: none, plots data x in equivalent phase space
    """
    if type=='MFE_burst':
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(x[:,0], x[:,1], x[:,2], lw=0.5)
        ax.set_xlabel("Roll & streak")
        ax.set_ylabel("Mean shear")
        ax.set_zlabel("Burst")
        ax.set_title("Self-sustaining process")

    if type=='kolmogorov':
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(x[:,0], x[:,1], x[:,2], lw=0.5)
        ax.set_xlabel("D")
        ax.set_ylabel("k")
        ax.set_zlabel("|a(1,4)|")

    if type=='MFE_dissipation' or type=='kolmogorov_kD':
        plt.figure()
        plt.plot(x[:,1], x[:,0])
        # plt.title("Dissipation vs energy")
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("D", fontsize=20)
        plt.xlabel("k", fontsize=20)

    if type=='sine':
        plt.figure(figsize=(7, 7))
        plt.plot(x[:,0],x[:,1])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("$x$", fontsize = 20)
        plt.ylabel("$y$", fontsize = 20)

    if type=='LA':
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(x[:,0], x[:,1], x[:,2], lw=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Lorenz Attractor")

    if type=='CDV':
        plt.figure(figsize=(6, 6))
        plt.scatter(x[:, 0], x[:, 3])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_4$", fontsize=20)
        plt.xlabel("$x_1$", fontsize=20)

    if type=='PM':
        plt.figure(figsize=(6,6))
        plt.plot(x[:,2], x[:,4])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_5$", fontsize=20)
        plt.xlabel("$x_3$", fontsize=20)

    return 1

def plot_tesselated_space(tess_ind,type):
    """Function for plotting the tessellated phase space - for the six chaotic systems

    :param tess_ind: matrix including the indices of the hypercubes of the subsequent data points, can
    be obtained from the tesselate(x,N) function
    :param type: string defining the type of system ("MFE_burst"; "MFE_dissipation"; "sine"; "LA"; "CDV"; "PM"; "kolmogorov"; "kolmogorov_kD")
    :return: none, plots data x in equivalent tessellated phase space
    """
    if type=='MFE_burst':
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter3D(tess_ind[:, 0], tess_ind[:, 1], tess_ind[:, 2])
        ax.set_xlabel("Roll & streak")
        ax.set_ylabel("Mean shear")
        ax.set_zlabel("Burst")

    if type=='kolmogorov':
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter3D(tess_ind[:, 0], tess_ind[:, 1], tess_ind[:, 2])
        ax.set_xlabel("D")
        ax.set_ylabel("k")
        ax.set_zlabel("|a(1,4)|")

    if type=='MFE_dissipation' or type=='kolmogorov_kD':
        plt.figure(figsize=(7, 7))
        plt.scatter(tess_ind[:,1], tess_ind[:,0], s=10, marker='s', facecolors = 'None', edgecolor = 'blue') #I should relate somehow s to N and the fig size
        plt.xlim([-0.5, 99.5])
        plt.ylim([-0.5, 99.5])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("k")
        plt.ylabel("D")

    if type=='sine':
        plt.figure(figsize=(7, 7))
        plt.scatter(tess_ind[:, 0], tess_ind[:, 1], s=200, marker='s', facecolors='None', edgecolor='blue')
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("x")
        plt.ylabel("y")

    if type=='LA':
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter3D(tess_ind[:,0], tess_ind[:,1], tess_ind[:,2])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    if type=='CDV':
        plt.figure(figsize=(6, 6))
        plt.scatter(tess_ind[:, 0], tess_ind[:, 3], s=200, marker='s', facecolors='None', edgecolor='blue')
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_4$")
        plt.xlabel("$x_1$")

    if type=='PM':
        plt.figure(figsize=(6, 6))
        plt.scatter(tess_ind[:, 2], tess_ind[:, 4], s=200, marker='s', facecolors='None', edgecolor='blue')
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_5$")
        plt.xlabel("$x_3$")

    return 1

def plot_phase_space_clustered(x,type,D_nodes_in_clusters,tess_ind_cluster, coord_centers, extr_clusters,nr_dev,palette):
    """Function for plotting phase space with cluster affiliation - for the six chaotic systems

    :param x: data series
    :param type: string defining the type of system ("MFE_burst"; "MFE_dissipation"; "sine"; "LA"; "CDV"; "PM"; "kolmogorov"; "kolmogorov_kD")
    :param D_nodes_in_clusters: matrix of affiliation of points to the community clusters
    :param tess_ind_cluster: vector of time series with their cluster affiliation
    :param coord_centers: cluster centers in phase space
    :param extr_clusters: vector of extreme cluster ids
    :param nr_dev: scalar defining how far away from the mean (in multiples of the standard deviation) will be considered an extreme event
    :param palette: color palette decoding a unique color code for each cluster
    :return: none, plots the phase space colored by cluster affiliation
    """
    if type=='MFE_burst':
        plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(D_nodes_in_clusters.shape[1]):   # for all communities
            ax.scatter(x[tess_ind_cluster==i,0], x[tess_ind_cluster==i,1], x[tess_ind_cluster==i,2])
            if i in extr_clusters:
                t = ax.text(coord_centers[i,0], coord_centers[i,1], coord_centers[i,2], str(i),color='r', backgroundcolor='1')  # display cluster id
                t.set_bbox(dict(facecolor='black', alpha=0.35))
            else:
                t = ax.text(coord_centers[i,0], coord_centers[i,1], coord_centers[i,2], str(i), color='white', backgroundcolor='1')  # display cluster id
                t.set_bbox(dict(facecolor='black', alpha=0.35))
        ax.set_xlabel("Roll & streak")
        ax.set_ylabel("Mean shear")
        ax.set_zlabel("Burst")
        ax.set_title("Self-sustaining process")

    if type == 'kolmogorov':
        plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(D_nodes_in_clusters.shape[1]):  # for all communities
            ax.scatter(x[tess_ind_cluster == i, 0], x[tess_ind_cluster == i, 1], x[tess_ind_cluster == i, 2],color=palette.colors[i,:])  # I should relate somehow s to N and the fig size
            if i in extr_clusters:
                t = ax.text(coord_centers[i, 0], coord_centers[i, 1], coord_centers[i, 2], str(i),
                        color='r', backgroundcolor='1')  # display cluster id
                t.set_bbox(dict(facecolor='black', alpha=0.35))
            else:
                t = ax.text(coord_centers[i, 0], coord_centers[i, 1], coord_centers[i, 2],
                        str(i), color='white', backgroundcolor='1')  # display cluster id
                t.set_bbox(dict(facecolor='black', alpha=0.35))
        ax.set_xlabel("D")
        ax.set_ylabel("k")
        ax.set_zlabel("|a(1,4)|")

    if type=='MFE_dissipation' or type=='kolmogorov_kD':
        plt.figure(figsize=(7, 7))
        plt.axhline(y=np.mean(x[:,0])+nr_dev*np.std(x[:, 0]), color='r', linestyle='--') # plot horizontal cutoff
        plt.axvline(x=np.mean(x[:, 1]) + nr_dev*np.std(x[:, 1]), color='r', linestyle='--')  # plot horizontal cutoff
        for i in range(D_nodes_in_clusters.shape[1]):  # for all communities
            plt.scatter(x[tess_ind_cluster == i,1], x[tess_ind_cluster == i,0], color=palette.colors[i,:])

            if i in extr_clusters:     # if cluster is extreme - plot number in red
                t = plt.text(coord_centers[i,1], coord_centers[i,0], str(i),color='r', backgroundcolor='1')  # display cluster id
                t.set_bbox(dict(facecolor='black', alpha=0.35))
            else:
                t = plt.text(coord_centers[i,1], coord_centers[i,0], str(i),color='white', backgroundcolor='1')  # display cluster id
                t.set_bbox(dict(facecolor='black', alpha=0.35))
        plt.minorticks_on()
        plt.xlabel("k")
        plt.ylabel("D")

    if type=='sine':
        plt.figure(figsize=(7, 7))
        plt.axhline(y=np.mean(x[:,0])+nr_dev*np.std(x[:, 0]), color='r', linestyle='--') # plot horizontal cutoff
        plt.axvline(x=np.mean(x[:,1]) + nr_dev*np.std(x[:, 1]), color='r', linestyle='--')  # plot horizontal cutoff
        for i in range(D_nodes_in_clusters.shape[1]):  # for all communities
            plt.scatter(x[tess_ind_cluster == i,0],
                        x[tess_ind_cluster == i,1], color=palette.colors[i,:])
            if i in extr_clusters:      # if cluster is extreme - plot number in red
                t = plt.text(coord_centers[i,0], coord_centers[i,1], str(i),color='r', backgroundcolor='1')  # display cluster id
                t.set_bbox(dict(facecolor='black', alpha=0.35))
            else:
                t = plt.text(coord_centers[i,0], coord_centers[i,1], str(i), color='white',backgroundcolor='1')  # display cluster id
                t.set_bbox(dict(facecolor='black', alpha=0.35))
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("x")
        plt.ylabel("y")

    if type=='LA':
        plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(D_nodes_in_clusters.shape[1]):   # for all communities
            ax.scatter3D(x[tess_ind_cluster==i,0], x[tess_ind_cluster==i,1], x[tess_ind_cluster==i,2],color=palette.colors[i,:])
            if i in extr_clusters:
                t = ax.text(coord_centers[i,0], coord_centers[i,1],coord_centers[i,2], str(i), color='r', backgroundcolor='1')  # display cluster id
                t.set_bbox(dict(facecolor='black', alpha=0.35))
            else:
                t = ax.text(coord_centers[i,0], coord_centers[i,1],coord_centers[i,2], str(i),color='white', backgroundcolor='1')  # display cluster id
                t.set_bbox(dict(facecolor='black', alpha=0.35))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    if type=='CDV':
        plt.figure(figsize=(6, 6))
        for i in range(D_nodes_in_clusters.shape[1]):  # for all communities
            plt.scatter(x[tess_ind_cluster == i, 0],
                        x[tess_ind_cluster == i, 3], color=palette.colors[i, :])
            if i in extr_clusters:  # if cluster is extreme - plot number in red
                plt.text(coord_centers[i, 0], coord_centers[i, 3], str(i), color='r')  # display cluster id
            else:
                plt.text(coord_centers[i, 0], coord_centers[i, 3], str(i), color='white')  # display cluster id
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_4$")
        plt.xlabel("$x_1$")

    if type=='PM':
        plt.figure(figsize=(6,6))
        for i in range(D_nodes_in_clusters.shape[1]):  # for all communities
            plt.scatter(x[tess_ind_cluster == i, 2],
                        x[tess_ind_cluster == i, 4], color=palette.colors[i, :])
            t = plt.text(coord_centers[i, 2], coord_centers[i, 4], str(i), color='white', backgroundcolor='1') # display cluster id
            t.set_bbox(dict(facecolor='black', alpha=0.35))
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_5$")
        plt.xlabel("$x_3$")

    return 1

def plot_phase_space_tess_clustered(tess_ind, type, D_nodes_in_clusters, tess_ind_cluster, coord_centers_tess, extr_clusters, palette):
    """Function for plotting tessellated phase space with cluster affiliation - for the six chaotic systems

    :param tess_ind: matrix including the indices of the hypercubes of the subsequent data points, can
    be obtained from the tesselate(x,N) function
    :param type: string defining the type of system ("MFE_burst"; "MFE_dissipation"; "sine"; "LA"; "CDV"; "PM"; 'kolmogorov'; "kolmogorov_kD")
    :param D_nodes_in_clusters: matrix of affiliation of points to the community clusters
    :param tess_ind_cluster: vector of time series with their cluster affiliation
    :param coord_centers_tess: cluster centers in tessellated phase space
    :param extr_clusters: vector of extreme cluster ids
    :param palette: color palette decoding a unique color code for each cluster
    :return: none, plots the tessellated phase space colored by cluster affiliation
    """
    if type=='MFE_dissipation' or type=='kolmogorov_kD':
        x,indices=np.unique(tess_ind,return_index=True,axis=0)  # unique hypercubes in tessellated space

        plt.figure(figsize=(7, 7))
        for i in range(len(x[:,0])): # for each unique point
            loc_clust = tess_ind_cluster[indices[i]]
            loc_col = palette.colors[loc_clust,:]
            plt.scatter(x[i,1], x[i,0], s=10, marker='s', facecolors = loc_col, edgecolor = loc_col)

        for i in range(D_nodes_in_clusters.shape[1]):  # for each cluster
            if i in extr_clusters:
                t = plt.text(coord_centers_tess[i,1], coord_centers_tess[i,0], str(i),color='r', backgroundcolor='1')  # display cluster id
                t.set_bbox(dict(facecolor='black', alpha=0.35))
            else:
                t = plt.text(coord_centers_tess[i,1], coord_centers_tess[i,0], str(i), color='white', backgroundcolor='1')  # display cluster id
                t.set_bbox(dict(facecolor='black', alpha=0.35))
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("k")
        plt.ylabel("D")
        plt.xlim([-0.5, 19.5])
        plt.ylim([-0.5, 19.5])

    if type=='kolmogorov':
        x, indices = np.unique(tess_ind, return_index=True, axis=0) # unique hypercubes in tessellated space

        plt.figure(figsize=(7, 7))
        ax = plt.axes(projection='3d')
        for i in range(len(x[:, 0])):  # for each unique point
            loc_clust = tess_ind_cluster[indices[i]]
            loc_col = palette.colors[loc_clust, :]
            ax.scatter3D(x[i, 0], x[i, 1], x[i,2],color=loc_col)

        for i in range(D_nodes_in_clusters.shape[1]):  # for each cluster
            if i in extr_clusters:
                t = ax.text(coord_centers_tess[i, 0], coord_centers_tess[i, 1], coord_centers_tess[i, 2], str(i),
                         color='r', backgroundcolor='1')  # display cluster id
                t.set_bbox(dict(facecolor='black', alpha=0.35))
            else:
                t = ax.text(coord_centers_tess[i, 0], coord_centers_tess[i, 1], coord_centers_tess[i, 2], str(i), color='white', backgroundcolor='1')  # display cluster id
                t.set_bbox(dict(facecolor='black', alpha=0.35))
        plt.grid('minor', 'both')
        plt.minorticks_on()
        ax.set_xlabel("D")
        ax.set_ylabel("k")
        ax.set_zlabel("|a(1,4)|")

    if type=='sine':
        x,indices=np.unique(tess_ind,return_index=True,axis=0)  # unique hypercubes in tessellated space

        plt.figure(figsize=(7, 7))
        for i in range(len(x[:,0])): # for each unique point
            loc_clust = tess_ind_cluster[indices[i]]
            loc_col = palette.colors[loc_clust,:]
            plt.scatter(x[i,0], x[i,1], s=200, marker='s', facecolors = loc_col, edgecolor = loc_col)

        for i in range(D_nodes_in_clusters.shape[1]):  # for each cluster
            if i in extr_clusters:
                t = plt.text(coord_centers_tess[i,0], coord_centers_tess[i,1], str(i),color='r', backgroundcolor='1')  # display cluster id
                t.set_bbox(dict(facecolor='black', alpha=0.35))
            else:
                t = plt.text(coord_centers_tess[i,0], coord_centers_tess[i,1], str(i), color='white', backgroundcolor='1')  # display cluster id
                t.set_bbox(dict(facecolor='black', alpha=0.35))
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("x")
        plt.ylabel("y")

    if type=='CDV':
        x, indices = np.unique(tess_ind, return_index=True, axis=0) # unique hypercubes in tessellated space

        plt.figure(figsize=(6, 6))
        for i in range(len(x[:,0])): # for each unique point
            loc_clust = tess_ind_cluster[indices[i]]
            loc_col = palette.colors[loc_clust,:]
            plt.scatter(x[i,0], x[i,3], s=200, marker='s', facecolors = loc_col, edgecolor = loc_col)

        for i in range(D_nodes_in_clusters.shape[1]):  # for each cluster
            if i in extr_clusters:
                plt.text(coord_centers_tess[i,0], coord_centers_tess[i,3], str(i),color='r')  # display cluster id
            else:
                plt.text(coord_centers_tess[i,0], coord_centers_tess[i,3], str(i), color='white')  # display cluster id
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_4$")
        plt.xlabel("$x_1$")

    if type=='PM':
        x, indices = np.unique(tess_ind, return_index=True, axis=0) # unique hypercubes in tessellated space

        plt.figure(figsize=(6, 6))
        for i in range(len(x[:, 0])):  # for each unique point
            loc_clust = tess_ind_cluster[indices[i]]
            loc_col = palette.colors[loc_clust, :]
            plt.scatter(x[i, 2], x[i, 4], s=200, marker='s', facecolors=loc_col,
                        edgecolor=loc_col)

        for i in range(D_nodes_in_clusters.shape[1]):  # for each cluster
            if i in extr_clusters:
                t = plt.text(coord_centers_tess[i, 2], coord_centers_tess[i, 4], str(i), color='r', backgroundcolor='1')  # display cluster id
                t.set_bbox(dict(facecolor='black', alpha=0.35))
            else:
                t = plt.text(coord_centers_tess[i, 2], coord_centers_tess[i, 4], str(i), color='white', backgroundcolor='1')  # display cluster id
                t.set_bbox(dict(facecolor='black', alpha=0.35))
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_5$")
        plt.xlabel("$x_3$")

    if type == 'LA':
        x, indices = np.unique(tess_ind, return_index=True, axis=0)  # unique hypercubes in tessellated space

        plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(len(x[:, 0])):  # for each unique point
            loc_clust = tess_ind_cluster[indices[i]]
            loc_col = palette.colors[loc_clust, :]
            ax.scatter3D(x[i, 0], x[i, 1], x[i, 2], color=loc_col)

        for i in range(D_nodes_in_clusters.shape[1]):  # for each cluster
            t = ax.text(coord_centers_tess[i, 0], coord_centers_tess[i, 1], coord_centers_tess[i, 2], str(i), color='white',
                            backgroundcolor='1') # display cluster id
            t.set_bbox(dict(facecolor='black', alpha=0.35))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    return 1

def plot_time_series(x,t, type):
    """Function for plotting the time series - for the six chaotic systems

    :param x: data matrix
    :param t: time vector
    :param type: string defining the type of system ("MFE_burst"; "MFE_dissipation"; "sine"; "LA"; "CDV"; "PM"; "kolmogorov"; "kolmogorov_kD")
    :return: none, plots time series of data
    """
    if type=='MFE_burst':
        fig, axs = plt.subplots(3)
        fig.suptitle("Dynamic behavior of the MFE flow")
        plt.subplot(3,1,1)
        plt.plot(t,x[:,0])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("Roll & streak")
        plt.xlabel("t")
        plt.subplot(3,1,2)
        plt.plot(t,x[:,1])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("Mean shear")
        plt.xlabel("t")
        plt.subplot(3,1,3)
        plt.plot(t,x[:,2])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("Burst")
        plt.xlabel("t")

    if type == 'kolmogorov':
        fig, axs = plt.subplots(3)
        fig.suptitle("Dynamic behavior of the Kolmogorov flow")
        plt.subplot(3, 1, 1)
        plt.plot(t, x[:, 0])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("D")
        plt.xlabel("t")
        plt.subplot(3, 1, 2)
        plt.plot(t, x[:, 1])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("k")
        plt.xlabel("t")
        plt.subplot(3, 1, 3)
        plt.plot(t, x[:, 2])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("|a(1,4)|")
        plt.xlabel("t")

    if type=='MFE_dissipation' or type=='kolmogorov_kD':
        fig, axs = plt.subplots(2)
        if type=='MFE_dissipation':
            fig.suptitle("Dynamic behavior of the MFE flow")
        if type=='kolmogorov':
            fig.suptitle("Dynamic behavior of the Kolmogorov flow")
        plt.subplot(2, 1, 1)
        plt.plot(t, x[:, 0])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("D", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(2, 1, 2)
        plt.plot(t, x[:, 1])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("k", fontsize=20)
        plt.xlabel("t", fontsize=20)

    if type=='sine':
        fig, axs = plt.subplots(2)
        fig.suptitle("Dynamic behavior of the sine wave")
        plt.subplot(2,1,1)
        plt.plot(t, x[:,0])
        plt.ylabel("$x$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(2,1,2)
        plt.plot(t, x[:,1])
        plt.ylabel("$y$", fontsize=20)
        plt.xlabel("t", fontsize=20)

    if type=='LA':
        fig, axs = plt.subplots(3)
        fig.suptitle("Dynamic behavior of the Lorenz Attractor")
        plt.subplot(3,1,1)
        plt.plot(t, x[:,0])
        plt.ylabel("$x$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(3,1,2)
        plt.plot(t, x[:,1])
        plt.ylabel("$y$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(3,1,3)
        plt.plot(t, x[:,2])
        plt.ylabel("$z$", fontsize=20)
        plt.xlabel("t", fontsize=20)

    if type=='CDV':

        fig, axs = plt.subplots(3)
        fig.suptitle("Dynamic behavior of the CDV flow")
        plt.subplot(3,1,1)
        plt.plot(t, x[:,0])
        plt.xlim([t[0],t[-1]])
        plt.ylabel("$x_1$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(3,1,2)
        plt.plot(t, x[:,1])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_2$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(3,1,3)
        plt.plot(t, x[:,2])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_3$", fontsize=20)
        plt.xlabel("t", fontsize=20)

        fig, axs = plt.subplots(3)
        fig.suptitle("Dynamic behavior of the CDV flow")

        plt.subplot(3, 1, 1)
        plt.plot(t, x[:, 3])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_4$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(3, 1, 2)
        plt.plot(t, x[:, 4])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_5$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(3, 1, 3)
        plt.plot(t, x[:, 5])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_6$", fontsize=20)
        plt.xlabel("t", fontsize=20)

    if type=='PM':
        fig, axs = plt.subplots(3)
        fig.suptitle("Dynamic behavior of the PM flow")

        plt.subplot(3,1,1)
        plt.plot(t, x[:,0])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_1$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(3,1,2)
        plt.plot(t, x[:,1])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_2$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(3,1,3)
        plt.plot(t, x[:,2])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_3$", fontsize=20)
        plt.xlabel("t", fontsize=20)

        fig, axs = plt.subplots(2)
        fig.suptitle("Dynamic behavior of the PM flow")

        plt.subplot(2,1,1)
        plt.plot(t, x[:,3])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_4$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(2,1,2)
        plt.plot(t, x[:,4])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_5$", fontsize=20)
        plt.xlabel("t", fontsize=20)

    return 1

def plot_time_series_clustered(y,t, tess_ind_cluster, palette, type):
    """Function for plotting the time series of data with cluster affiliation - for the six chaotic systems

    :param y: vector of the parameter to plot; for type=="MFE_burst" y should be the burst; for type=="MFE_dissipation"
    y should be the dissipation
    :param t: time vector
    :param tess_ind_cluster: vector of time series with their cluster affiliation
    :param palette: color palette decoding a unique color code for each cluster
    :param type: string defining the type of system ("MFE_burst"; "MFE_dissipation"; "sine"; "LA"; "CDV"; "PM"; "kolmogorov")
    :return: none, plots time series with cluster affiliation
    """
    plt.figure()
    plt.plot(t, y)
    for i in range(len(tess_ind_cluster)-1):
        if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:   # find the cluster transition in the data series
            loc_col = palette.colors[tess_ind_cluster[i]]
            # plt.axvline(x=(t[i] + t[i + 1]) / 2, color=loc_col, linestyle='--')
            plt.scatter((t[i] + t[i + 1]) / 2, (y[i] + y[i + 1]) / 2,marker='s', facecolors = 'None', edgecolor = loc_col)
            # plt.text(t[i], y[i], str(tess_ind_cluster[i]))  # numbers of clusters
    if type=='MFE_burst':
        plt.title("Burst component vs time")
        plt.ylabel("$b$")
    if type=='MFE_dissipation' or type=='kolmogorov':
        plt.title("Dissipation vs time")
        plt.ylabel("$D$")
    if type=='sine':
        plt.title("$x$ vs time")
        plt.ylabel("$x$")
    if type == 'LA':
        plt.title("$x$ vs time")
        plt.ylabel("$x$")
    if type=='CDV':
        plt.title("$x_0$ vs time")
        plt.ylabel("$x_0$")
    if type=='PM':
        plt.title("$x_5$ vs time")
        plt.ylabel("$x_5$")
    plt.xlabel("t")

    return 1

def plot_time_series_extr_iden(y,t, tess_ind_cluster, from_cluster, extr_clusters, type):
    '''Function for plotting time series with extreme event and precursor identification - for the six chaotic systems

    :param y: vector of the parameter to plot; for type=="MFE_burst" y should be the burst; for type=="MFE_dissipation"
    y should be the dissipation
    :param t: time vector
    :param tess_ind_cluster: vector of time series with their cluster affiliation
    :param from_cluster: vector of precursor cluster ids
    :param extr_clusters: vector of extreme clusterids
    :param type: string defining the type of system ("MFE_burst"; "MFE_dissipation"; "sine"; "kolmogorov")
    :return: none, plots time series with extreme event (blue) and precursor (red) identification
    '''
    plt.figure()
    plt.plot(t, y)
    for i in range(len(tess_ind_cluster)-1):
        if tess_ind_cluster[i] in from_cluster and tess_ind_cluster[i+1] in extr_clusters:
            plt.scatter(t[i], y[i], marker='s', facecolors = 'None', edgecolor = 'blue')
            plt.scatter(t[i+1], y[i+1], marker='s', facecolors='None', edgecolor='red')
    if type=='MFE_burst':
        plt.title("Burst component vs time")
        plt.ylabel("$b$")
    if type=='MFE_dissipation' or type=='kolmogorov':
        plt.title("Dissipation vs time")
        plt.ylabel("$D$")
    plt.xlabel("t")
    return 1

def avg_time_in_cluster(cluster_id,tess_ind_cluster,t):
    ''' Function for calculating the average time spent in a cluster

    :param cluster_id: id of local cluster
    :param tess_ind_cluster: vector of time series with their cluster affiliation
    :param t: time vector
    :return: returns average time spent in given cluster and the number of occurrences in the data series
    '''
    # Find all instances of cluster in the data series
    ind = np.where(tess_ind_cluster==cluster_id) # returns indices of the occurrences
    ind=ind[0]

    nr_cycles = 1
    t_cluster=[]

    t_start= t[ind[0]]
    for i in range(len(ind)-1): # loop through all occurrences
        if ind[i+1]!=ind[i]+1:  # if the next time is not there (is in a different cluster)
            t_cluster.append(t[ind[i]]-t_start)     # time spent in cluster during current cycle
            nr_cycles+=1
            t_start = t[ind[i+1]]
    # Include last point
    t_cluster.append(t[ind[-1]] - t_start)
    avg_time = np.mean(t_cluster)   # calculate average of the cycle times

    return avg_time, nr_cycles

def find_closest_center(y,clusters):
    ''' Function for finding the closest cluster (by distance to center); used if a data point does not belong to the
    previously considered tessellated hypercubes

    :param y: phase space position of data point
    :param clusters: list of identified clusters and all their parameters
    :return: returns id of the closest cluster
    '''
    min_dist = numpy.linalg.norm(y) # set minimum as distance from origin
    closest_cluster =-1
    for cluster in clusters:    # loop through all clusters
        dist = numpy.linalg.norm(cluster.center-y)  # calculate distance to local cluster center
        if dist<min_dist:
            closest_cluster=cluster.nr
    return closest_cluster

def extreme_event_identification_process(t,x,M,extr_dim,type, min_clusters, max_it, prob_type='classic',nr_dev=7,plotting=True, first_refined=False):
    """Main loop for the calculation for different systems

    :param t: time vector
    :param x: data matrix
    :param M: number of tesselation sections per dimension
    :param extr_dim: dimensions used for the definition of extreme events
    :param type: string defining the type of system ("MFE_burst"; "MFE_dissipation"; "sine"; "LA"; "CDV"; "PM"; "kolmogorov"; "kolmogorov_kD")
    :param min_clusters: minimum number of clusters which breaks the clustering and deflation loop
    :param max_it: maximum number of iterations which breaks the clustering anddeflation loop
    :param prob_type: string defining the type of probability to be calculated, either "classic" (default) or "backwards" (explained in probability(tess_ind, type))
    :param nr_dev: scalar defining how far away from the mean (multiples of the standard deviation) will be considered an extreme event (default is 7)
    :param plotting: bool property defining whether to plot the data
    :param first_refined: bool property defining whether the first clustering should be done with refinement (default is False)
    :return: returns list of clusters and their properties, the final hypercube to clusters affiliation matrix, the final
    transition probability matrix; additionally plots and saves the statistical results
    """
    dim = x.shape[1]    # number of dimensions of the phase space
    tess_ind, extr_id = tesselate(x, M, extr_dim,nr_dev)  # tessellate the data

    # # Additional loop for finding and saving the "actual" extreme events of the data series
    # temp = np.zeros_like(t)
    # for i in range(len(t)):
    #     for j in range(len(extr_id[:,0])):
    #         if tess_ind[i,0]==extr_id[j,0] and tess_ind[i,1]==extr_id[j,1]:
    #             temp[i]=2
    #             print(x[i,:])
    # np.save('actual_extreme', temp)

    # Transition probability
    P = probability(tess_ind, prob_type)  # create sparse transition probability matrix

    if plotting:
        plot_time_series(x,t,type)
        plot_phase_space(x,type)
        # plot_tesselated_space(tess_ind, type)

    tess_ind_trans = tess_to_lexi(tess_ind, M, dim) # translate tessellated data points to lexicographic ordering
    P, extr_trans = prob_to_sparse(P, M, extr_id)  # translate transition probability matrix into 2D sparse array with
                # points in lexicographic order, also translates the extreme event points

    # Graph form
    P_graph = to_graph_sparse(P)  # translate matrix to dictionary readable for clustering algorithm

    if plotting:
        if dim<3:  # only for small systems - otherwise the matrix is too large
            # Visualize unclustered graph
            plot_graph(P_graph,False,type)
            # Visualize probability matrix
            plot_prob_matrix(P.toarray())

    # Clustering - first step
    P_community = spectralopt.partition(P_graph, refine=first_refined)  # First clustering iteration; returns dictionary
                        # where hypercubes are keys and their affiliated communities are the values
    D_sparse = community_aff_sparse(0, P_community, M, dim, 'first', 1)  # create matrix of point-to-cluster affiliation

    # Deflate the Markov matrix
    P1 = sp.coo_matrix((D_sparse.transpose() * P) * D_sparse)
    # print(np.sum(P1,axis=0).tolist())  # For checking purposes - should be approx.(up to rounding errors) equal to number of nodes in each cluster

    # Graph form
    P1_graph = to_graph(P1.toarray())

    # For more iterations - enter the clustering loop
    P_community_old = P_community
    P_old = P1
    P_graph_old = P1_graph
    D_nodes_in_clusters = D_sparse
    int_id = 0

    # Deflation and refinement loop
    while int(np.size(np.unique(np.array(list(P_community_old.values()))))) > min_clusters and int_id < max_it:  # while the conditions are met
        int_id = int_id + 1
        print('iteration ', int_id)
        P_community_old, P_graph_old, P_old, D_nodes_in_clusters = clustering_loop_sparse(P_community_old, P_graph_old,
                                                                                          P_old, D_nodes_in_clusters)   # cluster and deflate the network
        # print(np.sum(D_nodes_in_clusters,axis=0).tolist()) # For checking purposes - should be approx.(up to rounding errors) equal to number of nodes in each cluster

    if plotting:
        # Visualize clustered graph
        plot_graph(P_graph_old,True, type)
        # Define color palette for visualizing different clusters
        palette = plt.get_cmap('viridis', D_nodes_in_clusters.shape[1])

    # Translate data points to cluster affiliation
    tess_ind_cluster = data_to_clusters(tess_ind_trans, D_nodes_in_clusters)

    # Calculate cluster centers
    coord_clust_centers, coord_clust_centers_tess = cluster_centers(x,tess_ind, tess_ind_cluster, D_nodes_in_clusters,dim)

    # Identify extreme and precursor clusters
    extr_clusters, from_clusters = extr_iden(extr_trans, D_nodes_in_clusters, P_old)
    # print('From cluster: ', from_clusters, 'To extreme cluster: ', extr_clusters)

    for i in range(P_old.shape[0]): # for all clusters
        denom = np.sum(D_nodes_in_clusters,axis=0)
        denom = denom[0,i]  # Calculate number of all hypercubes in cluster
        P_old.data[P_old.row == i] = P_old.data[P_old.row == i]/denom   # Correct the final transition probability matrix
                    # to get probability values as defined by probability theory

    if plotting:
    #     # Plot time series with clusters
    #     if type=='MFE_burst':
    #         plot_time_series_clustered(x[:,2], t, tess_ind_cluster, palette, type)
    #     if type=='MFE_dissipation' or type=='sine'or type=='LA' or type=='CDV' or type=='kolmogorov':
    #         plot_time_series_clustered(x[:,0], t, tess_ind_cluster, palette, type)
    #     if type == 'PM':
    #         plot_time_series_clustered(x[:, 4], t, tess_ind_cluster, palette, type)

        # Visualize phase space trajectory with clusters
        plot_phase_space_clustered(x, type, D_nodes_in_clusters, tess_ind_cluster, coord_clust_centers, extr_clusters,nr_dev, palette)

        # Plot time series with extreme event identification
        # if type == 'burst':
        #     plot_time_series_extr_iden(x[:,2], t, tess_ind_cluster, from_cluster, extr_cluster, type)
        # if type == 'dissipation':
        #     plot_time_series_extr_iden(x[:,0], t, tess_ind_cluster, from_cluster, extr_cluster, type)

        # Plot tessellated phase space with clusters
        plot_phase_space_tess_clustered(tess_ind, type, D_nodes_in_clusters, tess_ind_cluster, coord_clust_centers_tess, extr_clusters, palette)

        # Visualize probability matrix
        # plot_prob_matrix(P_old.toarray())

    clusters = []   # Create new list of class type objects

    # Define individual properties of clusters:
    for i in range(D_nodes_in_clusters.shape[1]):   # loop through all clusters
        nodes = D_nodes_in_clusters.row[D_nodes_in_clusters.col==i] # Identify hypercubes that belong to them

        center_coord=coord_clust_centers[i,:]
        center_coord_tess=coord_clust_centers_tess[i,:]

        # Calculate the average time spent in the cluster and the total number of instances
        avg_time, nr_instances = avg_time_in_cluster(i,tess_ind_cluster,t)

        # Add the cluster with all its properties to the final list of clusters
        clusters.append(cluster(i, nodes, center_coord, center_coord_tess, avg_time, nr_instances, P_old, extr_clusters, from_clusters))

    return clusters, D_nodes_in_clusters, P_old      # returns list of clusters (class:cluster), matrix of cluster affiliation and deflated probability matrix (sparse)