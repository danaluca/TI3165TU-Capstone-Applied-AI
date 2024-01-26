# Testing the created algorithm - Lorenz attractor model for Rayleigh-Bernard convection case
# Urszula Golyska 2022

from my_func import *

def LA_dt(x):
    ''' Explicit Euler scheme for the calculation of the time derivatives of the Lorenz Attractor parameters

    :param x: time series of the 3 dimensions of the LA system
    :return: returns the data series of the 3 derivatives of the LA system
    '''

    dxdt = np.zeros(x.shape)
    dxdt[0] = sigma*(x[1] - x[0])
    dxdt[1] = x[0]*(r-x[2])-x[1]
    dxdt[2] = x[0]*x[1]-b*x[2]
    return dxdt

plt.close('all') # close all open figures
type='LA'

# Model coefficients - same as in Kaiser et al. 2014
sigma = 10
b = 8/3
r = 28

# Time discretization
t0 = 0.0
tf = 110.0
dt = 0.01
t = np.arange(t0,tf,dt)
N = np.size(t)

x = np.zeros((N,3))
x[0,:] = np.array([0, 1.0, 1.05]) # initial conditions

# Generate data
for i in range(N-1):
    q = LA_dt(x[i,:])
    x[i+1,:] = x[i,:] + dt*q

# Delete first 500 ts to avoid numerical instabilities
x = x[500:,:]
t = t[500:]

M = 20  # Number of tessellation sections per phase space dimension
extr_dim = []    # no extreme events present for this system

# Tessellating and clustering loop
clusters, D, P = extreme_event_identification_process(t,x,M,extr_dim,type, 20, 20, 'classic',7,True, False)

# Calculate the statistics of the identified clusters
calculate_statistics(extr_dim, clusters, P, tf)
plt.show()