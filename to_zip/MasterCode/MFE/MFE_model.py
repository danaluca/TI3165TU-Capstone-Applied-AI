# Script for generating flow data for the Moehlis-Faisst-Eckhardt system
# Dr Anh Koan Doan with modifications by Urszula Golyska 2022

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import h5py

def get_MFE_param(alpha,beta,gamma,Re):
    '''Function for calculating the parameters of the MFE system

    :param alpha, beta, gamma: MFE system parameters
    :param Re: Reynolds number
    :return: returns other parameters of the MFE system
    '''
    kag = np.sqrt(alpha**2. + gamma**2.) # k alpha gamma
    kbg = np.sqrt(beta**2. + gamma**2.) # k beta gamma
    kabg = np.sqrt(alpha**2. + beta**2. + gamma**2.) # k alpha beta gamma
    k1 = (alpha*beta*gamma) / (kag * kbg)
    k2 = (alpha*beta*gamma) / (kag * kabg)
    
    # linear and forcing term
    zeta = np.array( [beta**2., 4.*beta**2. /3 + gamma**2., beta**2. + gamma**2. ,
          (3.*alpha**2.+4*beta**2.)/3. , alpha**2.+beta**2., (3.*alpha**2.+4.*beta**2.+3.*gamma**2.)/3. ,
          alpha**2.+beta**2.+gamma**2. , alpha**2.+beta**2.+gamma**2., 9.*beta**2. ] 
            ) / Re
    zeta = np.diag(zeta)
    
    # non-linear coupling coefficients
    xi1 = np.array( [ np.sqrt(3./2.) * beta * gamma / kabg, np.sqrt(3./2.)*beta*gamma/kbg] )
    
    xi2 = np.array( [ (5.*np.sqrt(2.) * gamma**2.) / (3.*np.sqrt(3) * kag), gamma**2. / (np.sqrt(6.)*kag), k2/np.sqrt(6.), xi1[1], xi1[1] ] )
    
    xi3 = np.array( [ 2.*k1 / np.sqrt(6.), ( beta**2. * (3.*alpha**2.+gamma**2.) - 3.*gamma**2.*(alpha**2.+gamma**2.) )/(np.sqrt(6.)*kag*kbg*kabg) ])
    
    xi4 = np.array( [ alpha/np.sqrt(6.), 10.*alpha**2./(3.*np.sqrt(6.)*kag), np.sqrt(3./2.) * k1, np.sqrt(3./2.)*alpha**2. * beta**2. / (kag*kbg*kabg),
                     alpha/np.sqrt(6.) ] )
    
    xi5 = np.array( [ xi4[0], alpha**2. / (np.sqrt(6.)*kag), xi2[2], xi4[0], xi3[0] ] )
    
    xi6 = np.array( [ xi4[0], xi1[0], (10.* (alpha**2. - gamma**2.) ) / (3.*np.sqrt(6.) * kag) , 2.*np.sqrt(2./3.) * k1, xi4[0], xi1[0]  ] )
    
    xi7 = np.array( [ xi4[0], (gamma**2. - alpha**2.)/(np.sqrt(6.) * kag) , k1/np.sqrt(6.)  ] )
    
    xi8 = np.array( [ 2.*k2/np.sqrt(6.), gamma**2. * (3.*alpha**2. - beta**2. + 3. * gamma**2.)/( np.sqrt(6.)*kag*kbg*kabg ) ] )
    
    xi9 = np.array( [ xi1[1], xi1[0] ] )
    
    return zeta, xi1,xi2,xi3,xi4,xi5,xi6,xi7,xi8,xi9

def MFE_RHS(u, zeta,xi1,xi2,xi3,xi4,xi5,xi6,xi7,xi8,xi9):
    '''Function for calculating the right hand side - for calculating the MFE system

    :param u: modes of the MFE system
    :param zeta, xi1 - xi9: system parameters
    :return: returns the right hand side of the ode
    '''
    RHS = - np.matmul(u,zeta)
    
    RHS[0] = RHS[0] + zeta[0,0] - xi1[0]*u[5]*u[7] + xi1[1]*u[1]*u[2]
    
    RHS[1] = RHS[1] + xi2[0]*u[3]*u[5] - xi2[1]*u[4]*u[6] - xi2[2]*u[4]*u[7] - xi2[3]*u[0]*u[2] - xi2[4]*u[2]*u[8]
    RHS[2] = RHS[2] + xi3[0]*(u[3]*u[6] + u[4]*u[5] ) + xi3[1]*u[3]*u[7]
    RHS[3] = RHS[3] - xi4[0]*u[0]*u[4] - xi4[1]*u[1]*u[5] - xi4[2]*u[2]*u[6] - xi4[3]*u[2]*u[7] - xi4[4]*u[4]*u[8]
    RHS[4] = RHS[4] + xi5[0]*u[0]*u[3] + xi5[1]*u[1]*u[6] - xi5[2]*u[1]*u[7] + xi5[3]*u[3]*u[8] + xi5[4]*u[2]*u[5]
    RHS[5] = RHS[5] + xi6[0]*u[0]*u[6] + xi6[1]*u[0]*u[7] + xi6[2]*u[1]*u[3] - xi6[3]*u[2]*u[4] + xi6[4]*u[6]*u[8] + xi6[5]*u[7]*u[8]
    RHS[6] = RHS[6] - xi7[0]*(u[0]*u[5] + u[5]*u[8] ) + xi7[1]*u[1]*u[4] + xi7[2]*u[2]*u[3]
    RHS[7] = RHS[7] + xi8[0]*u[1]*u[4] + xi8[1]*u[2]*u[3]
    RHS[8] = RHS[8] + xi9[0]*u[1]*u[2] - xi9[1]*u[5]*u[7]
    
    #return [RHS[0], RHS[1], RHS[2], RHS[3], RHS[4], RHS[5], RHS[6], RHS[7], RHS[8] ]
    return RHS

if __name__ == "__main__":

    # Domain size
    Lx = 4*np.pi
    Lz = 2*np.pi

    #Re = 310.34
    #Re = 150.
    #Re = 425.
    #Re = 600. # 340.# 600. # new Re to get sustained turbulence
    Re = 400.

    # System parameters
    alpha = 2.*np.pi/ Lx
    beta = np.pi/2.
    gamma = 2.*np.pi/ Lz
    
    zeta, xi1,xi2,xi3,xi4,xi5,xi6,xi7,xi8,xi9 = get_MFE_param(alpha,beta,gamma,Re)
        
    dt = 0.25   # time step
    Tmax = 5000.
    Nt = int(Tmax/dt)
    t = np.linspace(0,Tmax,Nt)

    #Initial conditions
    # values from Joglekar, Deudel & Yorke, "Geometry of the edge of chaos in a low dimensional turbulent shear layer model", PRE 91, 052903 (2015)
    u0 = np.array( [ 1.0, 0.07066, -0.07076, 0.0+0.001*np.random.rand(), 0.0, 0.0, 0.0, 0.0, 0.0])

    Energy0 = (1. - u0[0])**2. + np.sum(u0[1:]**2.)
    print(Energy0)
    u = np.zeros((Nt,9))
    u[0,:] = u0
    
    for i in range(Nt-1):   # for all time steps
        #Runge-Kutta 4
        k1 = dt*MFE_RHS(u[i,:], zeta, xi1,xi2,xi3,xi4,xi5,xi6,xi7,xi8,xi9)
        k2 = dt*MFE_RHS(u[i,:]+k1/3., zeta, xi1,xi2,xi3,xi4,xi5,xi6,xi7,xi8,xi9)
        k3 = dt*MFE_RHS(u[i,:]-k1/3.+k2, zeta, xi1,xi2,xi3,xi4,xi5,xi6,xi7,xi8,xi9)
        k4 = dt*MFE_RHS(u[i,:]+k1-k2+k3, zeta, xi1,xi2,xi3,xi4,xi5,xi6,xi7,xi8,xi9)
        
        u[i+1,:] = u[i,:] + (k1 + 3.*k2 + 3.*k3 + k4) / 8.

    
    # "turbulent energy"
    Energy = (1. - u[:,0])**2. + np.sum(u[:,1:]**2.,1)

    # datafile name
    # fln = 'Schmid_Garcia_Re400'
    # fln = 'MFE_Re' + str(int(Re)) + '_T' + str(int(Tmax)) + '.h5'

    # # Save results to file
    # f = h5py.File(fln,'w')
    # f.create_dataset('I',data=Energy)
    # f.create_dataset('u',data=u)
    # f.create_dataset('t',data=t)
    # f.create_dataset('Re',data=Re)
    # f.create_dataset('dt',data=dt)
    # f.create_dataset('u0',data=u0)
    # f.create_dataset('Lx',data=Lx)
    # f.create_dataset('Lz',data=Lz)
    # f.create_dataset('alpha',data=alpha)
    # f.create_dataset('beta',data=beta)
    # f.create_dataset('gamma',data=gamma)
    # f.create_dataset('zeta',data=zeta)
    # xiall = [xi1,xi2,xi3,xi4,xi5,xi6,xi7,xi8,xi9]
    # for i in range(1,10):
    #     var_name = 'xi' + str(i)
    #     f.create_dataset(var_name,data=xiall[i-1])
    # f.close()


    # # Plot results
    # plt.figure(figsize=(13, 7))
    # plt.subplot(511)
    # plt.plot(t,u[:,0])
    # plt.grid('minor', 'both')
    # plt.minorticks_on()
    # plt.xlabel("t")
    # plt.ylabel("$u_1$")
    # plt.subplot(512)
    # plt.plot(t,u[:,1])
    # plt.grid('minor', 'both')
    # plt.minorticks_on()
    # plt.xlabel("t")
    # plt.ylabel("$u_2$")
    # plt.subplot(513)
    # plt.plot(t,u[:,2])
    # plt.grid('minor', 'both')
    # plt.minorticks_on()
    # plt.xlabel("t")
    # plt.ylabel("$u_3$")
    # plt.subplot(514)
    # plt.plot(t,u[:,3])
    # plt.grid('minor', 'both')
    # plt.minorticks_on()
    # plt.xlabel("t")
    # plt.ylabel("$u_4$")
    # plt.subplot(515)
    # plt.plot(t,u[:,4])
    # plt.grid('minor', 'both')
    # plt.minorticks_on()
    # plt.xlabel("t")
    # plt.ylabel("$u_5$")
    #
    #
    # plt.figure(figsize=(13, 7))
    # plt.subplot(511)
    # plt.plot(t,u[:,5])
    # plt.grid('minor', 'both')
    # plt.minorticks_on()
    # plt.xlabel("t")
    # plt.ylabel("$u_6$")
    # plt.subplot(512)
    # plt.plot(t,u[:,6])
    # plt.grid('minor', 'both')
    # plt.minorticks_on()
    # plt.xlabel("t")
    # plt.ylabel("$u_7$")
    # plt.subplot(513)
    # plt.plot(t,u[:,7])
    # plt.grid('minor', 'both')
    # plt.minorticks_on()
    # plt.xlabel("t")
    # plt.ylabel("$u_8$")
    # plt.subplot(514)
    # plt.plot(t,u[:,8])
    # plt.grid('minor', 'both')
    # plt.minorticks_on()
    # plt.xlabel("t")
    # plt.ylabel("$u_9$")
    #
    #
    # #plt.figure(3)
    # plt.subplot(515)
    # plt.plot(t,Energy)
    # plt.grid('minor', 'both')
    # plt.minorticks_on()
    # plt.xlabel("t")
    # plt.ylabel("E")
    #
    # plt.show()
    

    
