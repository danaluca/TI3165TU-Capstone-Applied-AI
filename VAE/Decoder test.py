from VAEClass import VAE_Decoder, preprocess
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Load data
u_train, u_val, u_test = preprocess()
model = VAE_Decoder(decoder_file='vae_decoder_ph_v1.2.h5')


# Generate latent space trajectories
window_size=50	# moving average window size
N = 200			# number of timesteps
Z = 30			# latent space dimensions (dependent on VAE model)

x_rand = np.random.normal(scale=1.5, size=(N+window_size-1, Z))
x_rand = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window_size)/window_size, mode='valid') + np.array([np.random.normal(scale=0.3)]*N), axis=0, arr=x_rand)


# Plot latent space trajectories
for i in range(10):
	plt.plot(x_rand[:, i], label=f"component {i}")
plt.legend(), plt.tight_layout(), plt.show()


# Decode latent space trajectories
x_rand = tf.reshape(tf.convert_to_tensor(x_rand), shape=(N, 1, 1, Z))
x = model.forward(x_rand)


# Animate reconstructed flow field
def update(frame):
	ax.clear()
	ax.imshow(x[frame, :, :, 0], interpolation='bicubic')
	ax.set_title(f't = {frame}')
	return ax

fig, ax = plt.subplots(figsize=(6, 6))
animation = FuncAnimation(fig, update, frames=range(N), interval=50)
plt.show()


#plt.imshow(x[0, :, :, 0], interpolation='bicubic')
#plt.tight_layout(), plt.show()


"""
z = np.load('vae_latent_space.npy')
z = tf.reshape(tf.convert_to_tensor(z), shape=(3000, 1, 1, 30))

x = model.forward(z)
np.save('reconstructed_flows_v2.npy', x)

print(x.shape)
"""
