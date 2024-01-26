# Testing the created algorithm - sine wave with artificial extreme events case
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

# Time discretization
t0 = 0.0
tf = 1000.0
dt = 0.01

# extreme event parameters
nt_ex = 50  # number of time steps of the extreme event
rand_threshold = 0.9
rand_amplitude = 2
rand_scalar = 1

# Generate data
t, x = sine_data_generation(t0, tf, dt, nt_ex, rand_threshold, rand_amplitude, rand_scalar)

extr_dim = [0,1]   # Both phase space coordinates will be used to define extreme events

M = 20  # Number of tessellation sections per phase space dimension

plotting = False
min_clusters = 15
max_it = 5

# Tessellating and clustering loop
clusters, D, P = extreme_event_identification_process(t,x,M,extr_dim,type, min_clusters, max_it, 'classic', 1.5,plotting, True)

# Calculate the statistics of the identified clusters
calculate_statistics(extr_dim, clusters, P, tf)
plt.show()

# Check on "new" data series
# Here we take the old data series and feed it to the algorithm as if it was new
x_tess,temp = tesselate(x,M,extr_dim,7)    # Tessellate data set (without extreme event identification)
x_tess = tess_to_lexi(x_tess, M, x.shape[1])
x_clusters = data_to_clusters(x_tess, D, x, clusters)   # Translate data set to already identified clusters

is_extreme = np.zeros_like(x_clusters)
for cluster in clusters:
    is_extreme[np.where(x_clusters==cluster.nr)]=cluster.is_extreme     # New data series, determining whether the current
                        # state of the system is extreme (2), precursor (1) or normal state (0)

# Calculate the false positive and false negative rates
avg_time, instances, instances_extreme_no_precursor, instances_precursor_no_extreme, instances_precursor_after_extreme = backwards_avg_time_to_extreme(is_extreme,dt)
print('Average time from precursor to extreme:', avg_time, ' s')
print('Nr times when extreme event had a precursor:', instances)
print('Nr extreme events without precursors (false negative):', instances_extreme_no_precursor)
print('Percentage of false negatives:', instances_extreme_no_precursor/(instances+instances_extreme_no_precursor)*100, ' %')
print('Percentage of extreme events with precursor (correct positives):', instances/(instances+instances_extreme_no_precursor)*100, ' %')
print('Nr precursors without a following extreme event (false positives):', instances_precursor_no_extreme)
print('Percentage of false positives:', instances_precursor_no_extreme/(instances+instances_precursor_no_extreme)*100, ' %')


# # Plot the changes of states on time series
# colors = ['#1f77b4', '#ff7f0e', '#d62728']     # blue, orange, red
#
# fig, axs = plt.subplots(2)
# plt.subplot(2, 1, 1)
# plt.plot(t,x[:,0])
# plt.ylabel("D")
# plt.xlabel("t")
#
# plt.subplot(2, 1, 2)
# plt.plot(t,x[:,1])
# plt.ylabel("k")
# plt.xlabel("t")
#
# for i in range(len(t) - 1):   # loop through whole data series
#     if is_extreme[i]!=is_extreme[i+1]:    # if in the next time step the state of the system is different
#         loc_col=colors[is_extreme[i+1]]
#
#         # Plot vertical line of appropriate color
#         plt.subplot(2, 1, 1)
#         plt.axvline(x=t[i+1], color=loc_col, linestyle='--')
#         plt.subplot(2, 1, 2)
#         plt.axvline(x=t[i + 1], color=loc_col, linestyle='--')
#
# plt.show()
