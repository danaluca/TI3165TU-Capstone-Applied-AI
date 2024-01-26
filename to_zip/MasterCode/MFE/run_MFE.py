# Testing the created algorithm - Moehlis-Faisst-Eckhardt case
# Urszula Golyska 2022

import h5py
from my_func import *
import numpy.linalg as linalg
import numpy as np

def MFE_get_param(alpha, beta, gamma, Re):
    """ Function for calculating the parameters of the MFE system

    :param alpha, beta, gamma: parameters of the system
    :param Re: Reynolds number of the system
    :return: returns 10 parameters, zeta and xi1-xi9
    """
    kag = np.sqrt(alpha ** 2. + gamma ** 2.)  # k alpha gamma
    kbg = np.sqrt(beta ** 2. + gamma ** 2.)  # k beta gamma
    kabg = np.sqrt(alpha ** 2. + beta ** 2. + gamma ** 2.)  # k alpha beta gamma
    k1 = (alpha * beta * gamma) / (kag * kbg)
    k2 = (alpha * beta * gamma) / (kag * kabg)

    # linear and forcing term
    zeta = np.array([beta ** 2., 4. * beta ** 2. / 3 + gamma ** 2., beta ** 2. + gamma ** 2.,
                     (3. * alpha ** 2. + 4 * beta ** 2.) / 3., alpha ** 2. + beta ** 2.,
                     (3. * alpha ** 2. + 4. * beta ** 2. + 3. * gamma ** 2.) / 3.,
                     alpha ** 2. + beta ** 2. + gamma ** 2., alpha ** 2. + beta ** 2. + gamma ** 2., 9. * beta ** 2.]
                    ) / Re
    zeta = np.diag(zeta)

    # non-linear coupling coefficients
    xi1 = np.array([np.sqrt(3. / 2.) * beta * gamma / kabg, np.sqrt(3. / 2.) * beta * gamma / kbg])

    xi2 = np.array([(5. * np.sqrt(2.) * gamma ** 2.) / (3. * np.sqrt(3) * kag), gamma ** 2. / (np.sqrt(6.) * kag),
                    k2 / np.sqrt(6.), xi1[1], xi1[1]])

    xi3 = np.array([2. * k1 / np.sqrt(6.),
                    (beta ** 2. * (3. * alpha ** 2. + gamma ** 2.) - 3. * gamma ** 2. * (alpha ** 2. + gamma ** 2.)) / (
                                np.sqrt(6.) * kag * kbg * kabg)])

    xi4 = np.array([alpha / np.sqrt(6.), 10. * alpha ** 2. / (3. * np.sqrt(6.) * kag), np.sqrt(3. / 2.) * k1,
                    np.sqrt(3. / 2.) * alpha ** 2. * beta ** 2. / (kag * kbg * kabg),
                    alpha / np.sqrt(6.)])

    xi5 = np.array([xi4[0], alpha ** 2. / (np.sqrt(6.) * kag), xi2[2], xi4[0], xi3[0]])

    xi6 = np.array(
        [xi4[0], xi1[0], (10. * (alpha ** 2. - gamma ** 2.)) / (3. * np.sqrt(6.) * kag), 2. * np.sqrt(2. / 3.) * k1,
         xi4[0], xi1[0]])

    xi7 = np.array([xi4[0], (gamma ** 2. - alpha ** 2.) / (np.sqrt(6.) * kag), k1 / np.sqrt(6.)])

    xi8 = np.array([2. * k2 / np.sqrt(6.), gamma ** 2. * (3. * alpha ** 2. - beta ** 2. + 3. * gamma ** 2.) / (
                np.sqrt(6.) * kag * kbg * kabg)])

    xi9 = np.array([xi1[1], xi1[0]])

    return zeta, xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9

def MFE_RHS(u, zeta, xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9):
    """Function for calculating the right hand side of the MFE system

    :param u: matrix of Fourier coefficients (of size Nt*9)
    :param zeta: system parameters
    :param xi1-xi9: full system coefficients
    :return: returns right hand side of the MFE system
    """
    RHS = - np.matmul(u, zeta)

    RHS[0] = RHS[0] + zeta[0, 0] - xi1[0] * u[5] * u[7] + xi1[1] * u[1] * u[2]

    RHS[1] = RHS[1] + xi2[0] * u[3] * u[5] - xi2[1] * u[4] * u[6] - xi2[2] * u[4] * u[7] - xi2[3] * u[0] * u[2] - xi2[
        4] * u[2] * u[8]
    RHS[2] = RHS[2] + xi3[0] * (u[3] * u[6] + u[4] * u[5]) + xi3[1] * u[3] * u[7]
    RHS[3] = RHS[3] - xi4[0] * u[0] * u[4] - xi4[1] * u[1] * u[5] - xi4[2] * u[2] * u[6] - xi4[3] * u[2] * u[7] - xi4[
        4] * u[4] * u[8]
    RHS[4] = RHS[4] + xi5[0] * u[0] * u[3] + xi5[1] * u[1] * u[6] - xi5[2] * u[1] * u[7] + xi5[3] * u[3] * u[8] + xi5[
        4] * u[2] * u[5]
    RHS[5] = RHS[5] + xi6[0] * u[0] * u[6] + xi6[1] * u[0] * u[7] + xi6[2] * u[1] * u[3] - xi6[3] * u[2] * u[4] + xi6[
        4] * u[6] * u[8] + xi6[5] * u[7] * u[8]
    RHS[6] = RHS[6] - xi7[0] * (u[0] * u[5] + u[5] * u[8]) + xi7[1] * u[1] * u[4] + xi7[2] * u[2] * u[3]
    RHS[7] = RHS[7] + xi8[0] * u[1] * u[4] + xi8[1] * u[2] * u[3]
    RHS[8] = RHS[8] + xi9[0] * u[1] * u[2] - xi9[1] * u[5] * u[7]

    # return [RHS[0], RHS[1], RHS[2], RHS[3], RHS[4], RHS[5], RHS[6], RHS[7], RHS[8] ]
    return RHS

def MFE_data_generation(Lx= 4*np.pi,Lz= 2*np.pi,Re=600, dt = 0.25, Tmax = 5000., plotting=0):
    """Function for the 2D data generation of the MFE system

    :param Lx: Domain size in x direction
    :param Lz: Domain size in z direction
    :param Re: Reynolds number of the flow
    :param dt: time step size
    :param Tmax: stopping time (assuming starting time is 0)
    :param plotting: bool property defining whether to plot the results
    :return: saves parameters, together with calculated Fourier coefficients u to hf file, no direct return
    """
    fln = 'MFE_Re' + str(int(Re)) + '_T' + str(int(Tmax)) + '.h5' # file name

    alpha = 2. * np.pi / Lx
    beta = np.pi / 2.
    gamma = 2. * np.pi / Lz

    zeta, xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9 = get_MFE_param(alpha, beta, gamma, Re)

    Nt = int(Tmax / dt)
    # EI = np.sqrt(1.1 - 1.) / 2.

    # values from Joglekar, Deudel & Yorke, "Geometry of the edge of chaos in a low dimensional turbulent shear layer model", PRE 91, 052903 (2015)
    u0 = np.array([1.0, 0.07066, -0.07076, 0.0 + 0.001 * np.random.rand(), 0.0, 0.0, 0.0, 0.0, 0.0])

    t = np.linspace(0, Tmax, Nt)

    # Energy0 = (1. - u0[0]) ** 2. + np.sum(u0[1:] ** 2.)
    # print(Energy0)

    u = np.zeros((Nt, 9))
    u[0, :] = u0

    for i in range(Nt - 1):
        # RHS = MFE_RHS(u[i,:], zeta, xi1,xi2,xi3,xi4,xi5,xi6,xi7,xi8,xi9)
        # u[i+1,:] = u[i,:] + dt*RHS
        k1 = dt * MFE_RHS(u[i, :], zeta, xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9)
        k2 = dt * MFE_RHS(u[i, :] + k1 / 3., zeta, xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9)
        k3 = dt * MFE_RHS(u[i, :] - k1 / 3. + k2, zeta, xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9)
        k4 = dt * MFE_RHS(u[i, :] + k1 - k2 + k3, zeta, xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9)

        u[i + 1, :] = u[i, :] + (k1 + 3. * k2 + 3. * k3 + k4) / 8.

    # "turbulent energy"
    Energy = (1. - u[:, 0]) ** 2. + np.sum(u[:, 1:] ** 2., 1)

    f = h5py.File(fln, 'w')
    f.create_dataset('I', data=Energy)
    f.create_dataset('u', data=u)
    f.create_dataset('t', data=t)
    f.create_dataset('Re', data=Re)
    f.create_dataset('dt', data=dt)
    f.create_dataset('u0', data=u0)
    f.create_dataset('Lx', data=Lx)
    f.create_dataset('Lz', data=Lz)
    f.create_dataset('alpha', data=alpha)
    f.create_dataset('beta', data=beta)
    f.create_dataset('gamma', data=gamma)
    f.create_dataset('zeta', data=zeta)
    xiall = [xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9]

    for i in range(1, 10):
        var_name = 'xi' + str(i)
        f.create_dataset(var_name, data=xiall[i - 1])
    f.close()

    if plotting:
        plt.figure(figsize=(13, 7))
        plt.subplot(511)
        plt.plot(t, u[:, 0])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("$u_1$")
        plt.subplot(512)
        plt.plot(t, u[:, 1])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("$u_2$")
        plt.subplot(513)
        plt.plot(t, u[:, 2])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("$u_3$")
        plt.subplot(514)
        plt.plot(t, u[:, 3])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("$u_4$")
        plt.subplot(515)
        plt.plot(t, u[:, 4])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("$u_5$")

        plt.figure(figsize=(13, 7))
        plt.subplot(511)
        plt.plot(t, u[:, 5])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("$u_6$")
        plt.subplot(512)
        plt.plot(t, u[:, 6])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("$u_7$")
        plt.subplot(513)
        plt.plot(t, u[:, 7])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("$u_8$")
        plt.subplot(514)
        plt.plot(t, u[:, 8])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("$u_9$")

        # plt.figure(3)
        plt.subplot(515)
        plt.plot(t, Energy)
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("E")

        plt.show()

    return 1

def MFE_read_Fourier(filename):
    """Function for reading MFE data including Fourier coefficients of the flow

    :param filename: name of hf file containing the data
    :return: returns time vector t and matrix of Fourier coefficients u (of size 9*Nt)
    """
    hf = h5py.File(filename, 'r')
    u = np.array(hf.get('/u'))
    t = np.array(hf.get('/t'))
    return t,u

def MFE_to_burst(u):
    """Function for translating Fourier coefficient data to mean shear, roll streak and burst components

    :param u: matrix of Fourier coefficients (of size Nt*9)
    :return: returns matrix of mean shear, roll streak and burst components x (of size Nt*3)
    """
    mean_shear = np.absolute(1-u[:,0])    # from a1,a9
    roll_streak = linalg.norm(u[:,1:3].transpose(),axis=0)  # from a2,a3,a4
    burst = linalg.norm(u[:,3:5].transpose(),axis=0)    # from a5,a6,a7,a8
    x = np.vstack([roll_streak, mean_shear, burst]).transpose()
    return x

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
t,x = MFE_read_DI(filename, dt) # read pre-generated data

extr_dim = [0,1]    # Both turbulent kinetic energy and energy dissipation will be used to define extreme events
nr_dev = 7

# Number of tessellation sections per phase space dimension
M = 20

plotting = False
min_clusters = 20
max_it = 10

# Tessellating and clustering loop
clusters, D, P = extreme_event_identification_process(t,x,M,extr_dim,type, min_clusters, max_it, 'classic', nr_dev,plotting, False)

# Calculate the statistics of the identified clusters
calculate_statistics(extr_dim, clusters, P, t[-1])
plt.show()

# Check on "new" data series
# Here we take the old data series and feed it to the algorithm as if it was new
x_tess,temp = tesselate(x,M,extr_dim,nr_dev)    # Tessellate data set (without extreme event identification)
x_tess = tess_to_lexi(x_tess, M, x.shape[1])
x_clusters = data_to_clusters(x_tess, D, x, clusters)  # Translate data set to already identified clusters


is_extreme = np.zeros_like(x_clusters)
for cluster in clusters:
    is_extreme[np.where(x_clusters==cluster.nr)]=cluster.is_extreme # New data series, determining whether the current
                        # state of the system is extreme (2), precursor (1) or normal state (0)

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


# ################# ALTERNATIVE DATA ANALYSIS ##### BURST #########
# type='MFE_burst'
# filename = 'MFE_Re400_T10000.h5'
# t,u = MFE_read_Fourier(filename)
#
# # Read data
# x = MFE_to_burst(u)
# extr_dim = [2]   # Only the value of burst will be used to define extreme events
# nr_dev = 7
#
# # Number of tessellation sections per phase space dimension
# M = 20
#
# plotting = True
# min_clusters = 30
# max_it = 10
#
# # Tessellating and clustering loop
# clusters, D, P = extreme_event_identification_process(t,x,M,extr_dim,type, min_clusters, max_it, 'classic', nr_dev,plotting, False)
#
# # Calculate the statistics of the identified clusters
# calculate_statistics(extr_dim, clusters, P, t[-1])
# plt.show()