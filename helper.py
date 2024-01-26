#Normal import statements
import h5py
import numpy as np
import matplotlib.pyplot as plt

#Changing the path to retrieve AE functions
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'old'))
notebook_path = os.getcwd()

from FlowCompression.ClassAE import AE


def data_reading(re: float = 40.0, nx: int = 24, nu: int = 2) -> np.array:
        """
        Function to read H5 files with flow data, can change Re to run for different flows
        :param re: float, Reynolds Number (20.0, 30.0, 40.0, 50.0, 60.0, 100.0, 180.0)
        :param nx: int, size of the grid
        :param nu: int, components of the velocity vector (1, 2)
        :param shuf: boolean, if true returns shuffled data
        :return: numpy array, Time series for given flow with shape [#frames, nx, nx, nu]
        """
        # File selection
        # Read dataset
        t = np.load('t.npy')
        # Instantiating the velocities array with zeros
        u_all = np.zeros((nx, nx, len(t), nu))

        # Update u_all with data from file
        u_all[:, :, :, 0] = np.load('u_refined.npy')
        if nu == 2:
            u_all[:, :, :, 1] = np.load('v_refined.npy')

        # Transpose of u_all in order to make it easier to work with it
        # Old dimensions -> [nx, nx, frames, nu]
        # New dimensions -> [frames, nx, nx, nu]
        u_all = np.transpose(u_all, [2, 0, 1, 3])

        return u_all

def energy(nxnx2: np.array) -> np.array:
    """
    Function to calculate energy of a singular frame
    :param nxnx2: numpy array, time frame of velocities with shape [nx,nx,2]
    :return: numpy array, kinetic grid wise energy of one image without taking mass into account
    """

    u = nxnx2[:, :, 0]
    v = nxnx2[:, :, 1]
    return 0.5 * np.add(np.multiply(u, u), np.multiply(v, v))

def dissipation(nxnx2: np.array, re: float = 40.0) -> np.array:
    u = nxnx2[:, :, 0]
    v = nxnx2[:, :, 1]
    u_grad = np.gradient(u, axis=0)
    v_grad = np.gradient(v, axis=1)
    return 0.5 * np.add(np.multiply(u_grad, u_grad), np.multiply(v_grad, v_grad))

def calculate_energy(u: np.array, re = 40.0) -> tuple[np.array]:
    '''
    Function to calculate the kinetic energy and the energy dissipation rate
    for a given velocity field. The function uses the predecessor's energy
    implementation. 

    :param u: numpy array of the velocity field time series [t, nx, nx, 2]
    :param re: float value representative of the Reynolds Number

    :return: u_k and u_d both sizes [t, nx, nx]
    '''
    u_k, u_d = np.empty((u.shape[0], u.shape[1], u.shape[2])), np.empty((u.shape[0], u.shape[1], u.shape[2]))
    
    for t in range(len(u)):
        kinetic_component = energy(u[t]) / ((2*np.pi) ** 2)
        dissipation_component = dissipation(u[t]) / ((2 * np.pi) ** 2) / re

        u_k[t] = kinetic_component
        u_d[t] = dissipation_component
        
    return u_k, u_d

def plot_energy(nxnx2: np.array) -> None:
    """
    Function to plot energy/grid without mass/density
    :param nxnx2: numpy array, the energy to be plotted in the shape of [nx,nx,2]
    :return: None, plots image
    """
    plt.contourf(nxnx2)
    plt.show()

def plot_energy2(energy:np.array, frame_min: int = 500) -> None:
    '''
    Function to plot energy without mass/density invovled. The function is produced a GIF
    the whole time series and showcases how the energy changes. Energy here implies
    either the kinetic energy or the energy dissipation rate, both works.

    :param energy: numpy array of the energy to be plotted in shape [t, nx, nx]
    :return: None, plots an video
    '''
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML, display
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set up the initial plot
    im = ax.imshow(energy[0, :, :], cmap='viridis', origin='lower', aspect='auto')
    ax.set_title('Kinetic Energy Evolution Over Time')

    # Update function for animation
    def update(frame):
        im.set_array(energy[frame, :, :])
        ax.set_title(f'Time Step {frame + 1}')
        return im,

    # Create animation
    animation = FuncAnimation(fig, update, frames=min(frame_min, len(energy)), interval=500, blit=True)

    # Save the animation as an HTML5 video
    video = animation.to_jshtml()

    # Display the HTML5 video
    display(HTML(video))

    # Good practice to close the plt object
    plt.close()