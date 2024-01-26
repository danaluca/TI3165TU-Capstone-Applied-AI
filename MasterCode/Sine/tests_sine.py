# Testing different parameters - sine wave with artificial extreme events case
# Urszula Golyska 2022
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

####################### Increased probability of transitioning to extreme event ########################################
t0 = 0.0
tf = 1000.0
dt = 0.01
nt_ex = 50  # number of time steps of the extreme event

rand_threshold_array = [0.7, 0.75, 0.9, 0.95, 0.99]
rand_amplitude = 2
rand_scalar = 1
M = 20  # number of tessellation sections in each dimension
extr_dim = [0, 1]  # Both phase space coordinates will be used to define extreme events

plotting=True

min_clusters = 15
max_it = 5

for rand_threshold in rand_threshold_array:
    t, x = sine_data_generation(t0, tf, dt, nt_ex, rand_threshold, rand_amplitude, rand_scalar)
    clusters, D, P = extreme_event_identification_process(t, x, M, extr_dim, type, min_clusters, max_it, 'classic', 1.5, plotting,
                                         False)
    # calculate_statistics(extr_dim, clusters, P, tf)
    plt.show()


###################################### Increased duration of state being within the extreme event ###############################3
plt.close('all') # close all open figures
t0 = 0.0
tf = 1000.0
dt = 0.01
nt_ex_array = [100]  # number of time steps of the extreme event

rand_threshold= 0.95
rand_amplitude = 2
rand_scalar = 1
M = 20  # number of tessellation sections in each dimension
extr_dim = [0, 1]  # Both phase space coordinates will be used to define extreme events

min_clusters = 15
max_it = 5

plotting=True

for nt_ex in nt_ex_array:
    t, x = sine_data_generation(t0, tf, dt, nt_ex, rand_threshold, rand_amplitude, rand_scalar)
    clusters, D, P = extreme_event_identification_process(t, x, M, extr_dim, type, min_clusters, max_it, 'classic', 1.5, plotting,
                                             False)
    # calculate_statistics(extr_dim, clusters, P, tf)
    plt.show()

################################### Increased distance (in phase-space) of the extreme event #################################
plt.close('all') # close all open figures
t0 = 0.0
tf = 1000.0
dt = 0.01
nt_ex = 50  # number of time steps of the extreme event

rand_threshold= 0.95
rand_amplitude_array = [1.6,1.7,2.8, 3]
rand_scalar = 1

M = 20  # number of tessellation sections in each dimension
extr_dim = [0, 1]  # Both phase space coordinates will be used to define extreme events

min_clusters = 15
max_it = 5

plotting=True

for rand_amplitude in rand_amplitude_array:
    t, x = sine_data_generation(t0, tf, dt, nt_ex, rand_threshold, rand_amplitude, rand_scalar)
    clusters, D, P = extreme_event_identification_process(t, x, M, extr_dim, type, min_clusters, max_it, 'classic', 1.5, plotting,
                                                 False)
    # calculate_statistics(extr_dim, clusters, P, tf)
    plt.show()

############################## Increased size (in phase-space) of the extreme event #####################################
t0 = 0.0
tf = 1000.0
dt = 0.01
nt_ex = 50  # number of time steps of the extreme event

rand_threshold= 0.95
rand_amplitude = 2
rand_scalar_array = [0.5,0.8,1,1.5,2]

M = 20  # number of discretizations in each dimension
extr_dim = [0, 1]  # define both phase space coordinates as extreme event

min_clusters = 15
max_it = 5

plotting=True # do we want to plot everything (otherwise plots only the last plot with extreme event identification)

for rand_scalar in rand_scalar_array:
    t, x = sine_data_generation(t0, tf, dt, nt_ex, rand_threshold, rand_amplitude, rand_scalar)
    clusters, D, P = extreme_event_identification_process(t, x, M, extr_dim, type, min_clusters, max_it, 'classic', 1.5, plotting,
                                                 False)
    # calculate_statistics(extr_dim, clusters, P, tf)
    plt.show()