# Testing the created algorithm - Pomeau and Manneville case
# Urszula Golyska 2022

from my_func import *

def PM_dt(x):
    '''Function for calculating the time derivatives for the PM system

    :param x: time series of the 5 dimensions of the PM system
    :return : returns the data series of the 5 derivatives of the PM system
    '''

    dxdt = np.zeros(x.shape)
    dxdt[:, 0] = x[:,1]
    dxdt[:, 1] = -x[:,0]**3 - 2*x[:,0]*x[:,2] + x[:,0]*x[:,4] - mu*x[:,1]
    dxdt[:, 2] = x[:,3]
    dxdt[:, 3] = -x[:,2]**3-nu[0]*x[:,0]**2 + x[:,4]*x[:,2] - nu[1]*x[:,3]
    dxdt[:, 4] = -nu[2]*x[:,4] - nu[3]*x[:,0]**2 - nu[4] *(x[:,2]**2 - 1)

    return dxdt

plt.close('all') # close all open figures
type='PM'

# model coefficients
mu = 1.815
nu = [1.0, 1.815, 0.44, 2.86, 2.86]

# Time discretization
t0 = 0.0
tf = 1000.0
dt = 0.1
t = np.arange(t0,tf,dt)
N = np.size(t)

extr_dim = []   # no extreme events present for this system

x = np.zeros((N,5))
x[0,:] = np.array([1,1,0,0,0]) # initial condition

# Generate data
for i in range(N-1):
    q = PM_dt(x[i:i+1,:])
    x[i+1,:] = x[i,:] + dt*q[0,:]

# Delete first 500 ts to avoid numerical instabilities
x = x[500:,:]
t = t[500:]

M = 20  # Number of tessellation sections per phase space dimension

plotting = True
min_clusters = 30
max_it = 10

# Tessellating and clustering loop
clusters, D, P = extreme_event_identification_process(t,x,M,extr_dim,type, min_clusters, max_it, 'classic', 7,plotting, False)

# Calculate the statistics of the identified clusters
calculate_statistics(extr_dim, clusters, P, t[-1])
plt.show()
