from VAEClass import VAE_Encoder, preprocess
import matplotlib.pyplot as plt


# Load data
u_train, u_val, u_test = preprocess()

# Load model
model = VAE_Encoder(encoder_file='vae_encoder_ph_v1.2.h5')

# Calculate latent space trajectories of first 150 timesteps
z = model.forward(u_train[0:150])


# Save/load model (optional)
#np.save('vae_latent_space', z)
#z = np.load('vae_latent_space.npy')


# Plot latent space components
for i in range(4):
	plt.plot(z[:, i], label=f"component {i}")
plt.legend(), plt.tight_layout(), plt.show()