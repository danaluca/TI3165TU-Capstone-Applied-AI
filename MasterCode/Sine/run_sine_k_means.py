# Comparison case using k-means clustering - sine wave with artificial extreme events case
# Urszula Golyska 2022

from sklearn.cluster import KMeans
from my_func import *

def sine_data_generation(t0, tf, dt, nt_ex, rand_threshold=0.9, rand_amplitude=2, rand_scalar=1):
    """Function for generating data of the sine wave with artificial extreme events

    :param t0: starting time of data series
    :param tf: final time of data series
    :param dt: time step
    :param nt_ex: number of time steps that the extreme event will last
    :param rand_threshold: threshold defining the probability of the extreme event
    :param rand_amplitude: amplitude defining the jump of the extreme event
    :param rand_scalar: scalar value defining the size of the extreme event
    :return: returns time vector t and matrix of data (of size 2*Nt)
    """
    # Time discretization
    t = np.arange(t0,tf,dt)

    # Phase space dimensions
    u1 = np.sin(t)
    u2 = np.cos(t)

    # Generate random extreme events
    for i in range(len(t)):
        if u1[i]>=0.99 and abs(u2[i])<=0.015:   # If the trajectory is at a certain position
            if np.random.rand() >=rand_threshold:   # randomness
                # Shift the phase space parameters
                u1[i:i+nt_ex] = rand_scalar*u1[i:i+nt_ex]+np.linspace(rand_amplitude/2, rand_amplitude, num=nt_ex)
                u2[i:i+nt_ex] = rand_scalar*u2[i:i+nt_ex]+np.linspace(rand_amplitude/2, rand_amplitude, num=nt_ex)

    u = np.hstack([np.reshape(u1, (len(t),1)),np.reshape(u2, (len(t),1))])  # Combine into one matrix
    return t, u

plt.close('all') # close all open figures
type='sine'

# Time discretization
t0 = 0.0
tf = 1000.0
dt = 0.01

# Extreme event parameters
nt_ex = 50  # number of time steps of the extreme event
rand_threshold = 0.9
rand_amplitude = 2
rand_scalar = 1

# Generate data
t, x = sine_data_generation(t0, tf, dt, nt_ex, rand_threshold, rand_amplitude, rand_scalar)

extr_dim = [0,1]   # Both phase space coordinates will be used to define extreme events
dim = x.shape[1]

M = 20  # Number of tessellation sections per phase space dimension

nr_dev = 2
prob_type='classic'
nr_clusters_vec=[2,5,8,10,15,20,27] # different number of clusters to test

tess_ind, extr_id = tesselate(x, M, extr_dim, nr_dev)  # tessellate the data

# Transition probability
P = probability(tess_ind, prob_type)  # create sparse transition probability matrix

tess_ind_trans = tess_to_lexi(tess_ind, M, dim)  # translate tessellated data points to lexicographic ordering
P, extr_trans = prob_to_sparse(P, M, extr_id)  # translate transition probability matrix into 2D sparse array with
# points in lexicographic order, also translates the extreme event points

# Graph form
P_graph = to_graph_sparse(P)  # translate to dict readable for partition

for nr_clusters in nr_clusters_vec: # for a certain number of clusters
    # Clustering - kmeans++
    kmeans = KMeans(init="k-means++",n_clusters=nr_clusters, n_init=10,max_iter=300,random_state=42)
    kmeans.fit(x)   # fitting directly the data series rather than the tessellated version

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

    tess_ind_cluster = kmeans.labels_   # translate data points to affiliated cluster id

    # Calculate cluster centers
    coord_clust_centers, coord_clust_centers_tess = cluster_centers(x,tess_ind, tess_ind_cluster, D_sparse,dim)
    coord_clust_centers = kmeans.cluster_centers_

    extr_clusters, from_clusters = extr_iden(extr_trans, D_sparse, P1) # identify extreme and precursor clusters

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

plt.show()