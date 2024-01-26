# Comparison case  using k-means clustering for MFE system
# Urszula Golyska 2022

from sklearn.cluster import KMeans
from my_func import *

def MFE_read_DI(filename, dt=0.25):
    """Function for reading MFE data including the dissipation and energy of the flow

    :param filename: part of name of .npy files containing the dissipation and energy data
    :param dt: time step
    :return: returns time vector t and matrix of dissipation and energy x (of size Nt*2)
    """
    D = np.load(filename+'_dissipation.npy')
    I = np.load(filename+'_energy.npy')
    t = np.arange(len(I))*dt
    x = np.append(D, I, axis=1)
    return t,x

plt.close('all') # close all open figures

type='MFE_dissipation'
filename = 'MFE_Re600'
dt = 0.25
t,x = MFE_read_DI(filename, dt)     # read pre-generated data

extr_dim = [0,1]    # Both turbulent kinetic energy and energy dissipation will be used to define extreme events
nr_dev = 7

# Number of tessellation sections per phase space dimension
M = 20

prob_type = 'classic'
nr_clusters_vec = [5,10,15,20,27,35] # different number of clusters to test

for nr_clusters in nr_clusters_vec:
    dim = x.shape[1]
    tess_ind, extr_id = tesselate(x, M, extr_dim, nr_dev)  # tessellate the data

    # Transition probability
    P = probability(tess_ind, prob_type)  # create sparse transition probability matrix
    tess_ind_trans = tess_to_lexi(tess_ind, M, dim)
    P, extr_trans = prob_to_sparse(P, M, extr_id)  # translate transition probability matrix into 2D sparse array with
    # points in lexicographic order, also translates the extreme event points

    # Graph form
    P_graph = to_graph_sparse(P)  # translate to dict readable for partition
    # Clustering --  kmeans++
    kmeans = KMeans(init="k-means++",n_clusters=nr_clusters, n_init=10,max_iter=300,random_state=42)
    kmeans.fit(x) # fitting directly the data series rather than the tessellated version

    D = np.empty((0,3), dtype=int)  # matrix of indices of sparse matrix

    nodes, indices = np.unique(tess_ind_trans, return_index=True)

    for i in range(len(nodes)):   # for all unique hypercubes
        row = [nodes[i], kmeans.labels_[indices[i]], 1]  # prescribe hypercubes to communities
        D = np.vstack([D, row])

    D_sparse = sp.coo_matrix((D[:,2], (D[:,0],D[:,1])), shape=(M ** dim, nr_clusters))

    # Deflate the Markov matrix
    P1 = sp.coo_matrix((D_sparse.transpose() * P) * D_sparse)

    # Graph form
    P1_graph = to_graph(P1.toarray())

    # Define color palette for visualizing different clusters
    palette = plt.get_cmap('viridis', D_sparse.shape[1])

    tess_ind_cluster = kmeans.labels_ # translate data points to affiliated cluster id

    # Calculate cluster centers
    coord_clust_centers, coord_clust_centers_tess = cluster_centers(x,tess_ind, tess_ind_cluster, D_sparse,dim)
    coord_clust_centers = kmeans.cluster_centers_

    extr_clusters, from_clusters = extr_iden(extr_trans, D_sparse, P1)  # identify extreme and precursor clusters

    for i in range(P1.shape[0]): # for all clusters
        denom = np.sum(D_sparse,axis=0)
        denom = denom[0,i]  # Calculate number of all hypercubes in cluster
        P1.data[P1.row == i] = P1.data[P1.row == i]/denom # Correct the final transition probability matrix
                    # to get probability values as defined by probability theory

    # Visualize phase space trajectory with clusters
    plot_phase_space_clustered(x, type, D_sparse, tess_ind_cluster, coord_clust_centers, extr_clusters,nr_dev, palette)

    # Plot tessellated phase space with clusters
    plot_phase_space_tess_clustered(tess_ind, type, D_sparse, tess_ind_cluster, coord_clust_centers_tess, extr_clusters, palette)

    # list of class type objects
    clusters = []

    # Define individual properties of clusters:
    for i in range(D_sparse.shape[1]):   # loop through all clusters
        nodes = D_sparse.row[D_sparse.col==i] # Identify hypercubes that belong to them

        center_coord=coord_clust_centers[i,:]
        center_coord_tess=coord_clust_centers_tess[i,:]

        # Calculate the average time spent in the cluster and the total number of instances
        avg_time, nr_instances = avg_time_in_cluster(i,tess_ind_cluster,t)

        # Add the cluster with all its properties to the final list of clusters
        clusters.append(cluster(i, nodes, center_coord, center_coord_tess, avg_time, nr_instances, P1, extr_clusters, from_clusters))

    # Calculate the statistics of the identified clusters
    calculate_statistics(extr_dim, clusters, P1, t[-1])
    # plt.show()

    # Check on "new" data series
    # Here we take the old data series and feed it to the algorithm as if it was new
    x_tess,temp = tesselate(x,M,extr_dim,nr_dev)    # Tessellate data set (without extreme event identification)
    x_tess = tess_to_lexi(x_tess, M, x.shape[1])
    x_clusters = kmeans.labels_ # Translate data set to already identified clusters

    is_extreme = np.zeros_like(x_clusters)
    for loc_cluster in clusters:
        is_extreme[np.where(x_clusters==loc_cluster.nr)]=loc_cluster.is_extreme # New data series, determining whether the current
                        # state of the system is extreme (2), precursor (1) or normal state (0)

    # Calculate the false positive and false negative rates
    avg_time, instances, instances_extreme_no_precursor, instances_precursor_no_extreme,instances_precursor_after_extreme = backwards_avg_time_to_extreme(is_extreme,dt)
    print('-------------------', nr_clusters, ' CLUSTERS --------------')
    print('Average time from precursor to extreme:', avg_time, ' s')
    print('Nr times when extreme event had a precursor:', instances)
    print('Nr extreme events without precursors (false negative):', instances_extreme_no_precursor)
    print('Percentage of false negatives:', instances_extreme_no_precursor/(instances+instances_extreme_no_precursor)*100, ' %')
    print('Percentage of extreme events with precursor (correct positives):', instances/(instances+instances_extreme_no_precursor)*100, ' %')
    print('Nr precursors without a following extreme event (false positives):', instances_precursor_no_extreme)
    print('Percentage of false positives:', instances_precursor_no_extreme/(instances+instances_precursor_no_extreme)*100, ' %')
    print('Nr precursors following an extreme event:', instances_precursor_after_extreme)
    print('Corrected percentage of false positives:', (instances_precursor_no_extreme-instances_precursor_after_extreme)/(instances+instances_precursor_no_extreme)*100, ' %')
