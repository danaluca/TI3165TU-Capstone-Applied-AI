# Testing the created algorithm - Charney-DeVore case
# Urszula Golyska 2022
# Data generation by Dr Anh Khoa Doan
import sys
sys.path.append('c:\\Users\\alial\\OneDrive\\Belgeler\\GitHub\\TI3165TU-Capstone-AI\\MasterCode')
from my_func import *
import networkx as nx
from modularity_maximization import spectralopt
import scipy.sparse as sp
import h5py

def CDV_dt(x):
    '''Function for calculating the time derivatives for the CDV system

    :param x: time series of the 6 dimensions of the CDV system
    :return: returns the data series of the 6 derivatives of the CDV system
    '''

    dxdt = np.zeros(x.shape)
    dxdt[:,0] = gamma_m_star[0]*x[:,2] - C*(x[:,0] - x1s)
    dxdt[:,1] = -(alpha[0]*x[:,0] - beta[0])*x[:,2] - C*x[:,1] - delta[0]*x[:,3]*x[:,5]
    dxdt[:,2] = (alpha[0]*x[:,0] - beta[0])*x[:,1] - gamma_m[0]*x[:,0] - C*x[:,2] + delta[0]*x[:,3]*x[:,4]
    dxdt[:,3] = gamma_m_star[1]*x[:,5] - C*(x[:,3] - x4s) + epsilon*(x[:,1]*x[:,5] - x[:,2]*x[:,4])
    dxdt[:,4] = -(alpha[1]*x[:,0] - beta[1])*x[:,5] - C*x[:,4] - delta[1]*x[:,2]*x[:,3]
    dxdt[:,5] = (alpha[1]*x[:,0] - beta[1])*x[:,4] - gamma_m[1]*x[:,3] - C*x[:,5] + delta[1]*x[:,3]*x[:,1]

    return dxdt

def save_data_h5(fln,x,t,dt,x1s,x4s,C,beta0,gamma,m,alpha,beta,delta,gamma_m,gamma_m_star,epsilon):
    '''Function for saving the generated CDV data to a h5 file

    :param fln: file name
    :param x: time series of the 6 components of the CDV
    :param t: time vector
    :param dt: time step
    :param x1s, x4s, C, beta0, gamma, m, alpha, beta, delta, gamma_m, gamma_m_star, epsilon: parameters of the system
    :return: none, saves data to file
    '''
    hf = h5py.File(fln,'w')
    hf.create_dataset('x',data=x)
    hf.create_dataset('t',data=t) 
    hf.create_dataset('dt',data=dt)
    #hf.create_dataset('L',data=L)
    hf.create_dataset('x1s',data=x1s)
    hf.create_dataset('x4s',data=x4s)
    hf.create_dataset('C',data=C)
    hf.create_dataset('beta0',data=beta0)
    hf.create_dataset('gamma',data=gamma)
    #hf.create_dataset('b',data=b)
    hf.create_dataset('m',data=m)
    hf.create_dataset('alpha',data=alpha)
    hf.create_dataset('beta',data=beta)
    hf.create_dataset('delta',data=delta)
    hf.create_dataset('gamma_m',data=gamma_m)
    hf.create_dataset('gamma_m_star',data=gamma_m_star)
    hf.create_dataset('epsilon',data=epsilon)
    hf.close()

plt.close('all') # close all open figures
type='CDV'

# Model coefficients
x1s = .95
x4s = -.76095
C = .1
beta0 = 1.25
gamma = .2
b = .5

m = np.array([1,2])
alpha = 8*np.sqrt(2)*m**2*(b**2+m**2-1)/np.pi/(4*m**2-1)/(b**2+m**2)
beta = beta0*b**2/(b**2+m**2)
delta = 64*np.sqrt(2)*(b**2-m**2+1)/15/np.pi/(b**2+m**2)
gamma_m = gamma*4*np.sqrt(2)*m**3*b/np.pi/(4*m**2-1)/(b**2+m**2)
gamma_m_star = gamma*4*np.sqrt(2)*m*b/np.pi/(4*m**2-1)
epsilon = 16*np.sqrt(2)/5/np.pi

# Time discretization
t0 = 0.0
tf = 24000.0
dt = 0.1
t = np.arange(t0,tf,dt)
N = np.size(t)

x = np.zeros((N,6))
x[0,:] = np.array([.11,.22,.33,.44,.55,.66]) # initial conditions

# Generate data
for i in range(N-1):
    q = CDV_dt(x[i:i+1,:])
    x[i+1,:] = x[i,:] + dt*q[0,:]

# Delete first 500 ts to avoid numerical instabilities
x = x[500:,:]
t = t[500:]

# fln = 'CDV_T' + str(N) + '_DT01.h5'
# save_data_h5(fln,x,t,dt,x1s,x4s,C,beta0,gamma,m,alpha,beta,delta,gamma_m,gamma_m_star,epsilon) # save data to file

M = 20  # Number of tessellation sections per phase space dimension
extr_dim = []   # no extreme events present for this system

plotting = True
min_clusters = 30
max_it = 10

# Tessellating and clustering loop
clusters, D, P = extreme_event_identification_process(t,x,M,extr_dim,type, min_clusters, max_it, 'classic',7,plotting, False)

# Calculate the statistics of the identified clusters
calculate_statistics(extr_dim, clusters, P, tf)
plt.show()