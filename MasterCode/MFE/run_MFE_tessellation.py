# Testing different tessellation sizes - MFE system
# Urszula Golyska 2022

import h5py
from my_func import *
import numpy.linalg as linalg
import numpy as np

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

type ='MFE_dissipation'
filename = 'MFE_Re600'
dt = 0.25
t,x = MFE_read_DI(filename, dt) # read pre-generated data
extr_dim = [0,1]    # Both turbulent kinetic energy and energy dissipation will be used to define extreme events
nr_dev = 7

# Number of tessellation sections per phase space dimension
M_vector = [5,10,50,100] #[100]

plotting = True
min_clusters = 30
max_it = 10

for M in M_vector:
    # Tessellating and clustering loop
    clusters, D, P = extreme_event_identification_process(t,x,M,extr_dim,type, min_clusters, max_it, 'classic', nr_dev,plotting, False)

    # Calculate the statistics of the identified clusters
    calculate_statistics(extr_dim, clusters, P, t[-1])
    plt.show()

    # Check on "new" data series
    # Here we take the old data series and feed it to the algorithm as if it was new
    x_tess,temp = tesselate(x,M,extr_dim,nr_dev)    # Tessellate data set (without extreme event identification)
    x_tess = tess_to_lexi(x_tess, M, x.shape[1])
    x_clusters = data_to_clusters(x_tess, D, x, clusters) # Translate data set to already identified clusters

    is_extreme = np.zeros_like(x_clusters)
    for cluster in clusters:
        is_extreme[np.where(x_clusters==cluster.nr)]=cluster.is_extreme # New data series, determining whether the current
                        # state of the system is extreme (2), precursor (1) or normal state (0)

    # save_file_name = 'MFE_tess_'+str(M)
    # np.save(save_file_name, is_extreme)

    # Calculate the false positive and false negative rates
    avg_time, instances, instances_extreme_no_precursor, instances_precursor_no_extreme,instances_precursor_after_extreme = backwards_avg_time_to_extreme(is_extreme,dt)
    print('Average time from precursor to extreme:', avg_time, ' s')
    print('Nr times when extreme event had a precursor:', instances)
    print('Nr extreme events without precursors (false negative):', instances_extreme_no_precursor)
    print('Percentage of false negatives:', instances_extreme_no_precursor/(instances+instances_extreme_no_precursor)*100, ' %')
    print('Percentage of extreme events with precursor (correct positives):', instances/(instances+instances_extreme_no_precursor)*100, ' %')
    print('Nr precursors without a following extreme event (false positives):', instances_precursor_no_extreme)
    print('Percentage of false positives:', instances_precursor_no_extreme/(instances+instances_precursor_no_extreme)*100, ' %')
    print('Nr precursors following an extreme event:', instances_precursor_after_extreme)
    print('Corrected percentage of false positives:', (instances_precursor_no_extreme-instances_precursor_after_extreme)/(instances+instances_precursor_no_extreme)*100, ' %')
