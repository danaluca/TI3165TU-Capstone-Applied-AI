import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Reshape, AveragePooling2D, UpSampling2D, Conv2DTranspose, Layer
from keras.models import load_model

import h5py
from sklearn.utils import shuffle



@keras.saving.register_keras_serializable(package="Sampling")
class Sampling(Layer):
	"""
	Sampler used in the VAE network.
	Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
	"""

	def call(self, inputs, training=False):
		z_mean, z_log_var = inputs
		batch = tf.shape(z_mean)[0]
		dim = tf.shape(z_mean)[1]
		epsilon = tf.random.normal(shape=(batch, dim))
		if training:
			return z_mean + tf.exp(0.5 * z_log_var) * epsilon
		else:
			return z_mean




class VAE(keras.Model):
	"""
	A trianable VAE keras model
	"""
	def __init__(self, dimensions: list[int] = [240, 120, 60, 32], sampler_dim=45, activation_function: str = 'tanh', pooling: str = 'max', nx: int = 48, nu: int = 2,):
		"""
		:param dimensions: list, sets the dimensionality of output for each convolution layer.
						   dimensions[3] sets the size of the latent space
		:param sampler_dim: int, sets input size of the mean and variance FNN layers.
		:param activation_function: str
		:param pooling: str, pooling technique ('max','avg')
		:param nx: int, size of the grid
		:param nu: int, components of velocity vector (1,2)
		"""
		super().__init__()

		# Instantiate attributes
		self.dimensions = dimensions
		self.sampler_dim = sampler_dim
		self.activation_function = activation_function
		self.pooling = pooling


		# Input
		self.image = Input(shape=(nx, nx, nu))

		# Encoder
		x = Conv2D(self.dimensions[0], (3, 3), padding='same', activation=self.activation_function)(self.image)
		print(x.shape)
		x = self.pooling_function((2, 2), padding='same')(x)
		print(x.shape)
		x = Conv2D(self.dimensions[1], (3, 3), padding='same', activation=self.activation_function)(x)
		print(x.shape)
		x = self.pooling_function((3, 3), padding='same')(x) #(2, 2) for nx=24
		print(x.shape)
		x = Conv2D(self.dimensions[2], (3, 3), padding='same', activation=self.activation_function)(x)
		print(x.shape)
		x = self.pooling_function((3, 3), padding='same')(x) #(2, 2) for nx=24
		print(x.shape)

		# Sampler
		x = Flatten()(x)
		print(x.shape)
		x = Dense(self.sampler_dim, activation=self.activation_function)(x)
		print(x.shape)

		z_mean = Dense(self.dimensions[3])(x)
		z_log_var = Dense(self.dimensions[3])(x)
		z_sampled = Sampling()([z_mean, z_log_var])
		encoded = Reshape((1, 1, self.dimensions[3]))(z_sampled)

		print(encoded.shape)

		# Decoder
		print('Decoder')
		x = Conv2DTranspose(self.dimensions[3], (3, 3), padding='same', activation=self.activation_function)(encoded)
		print(x.shape)
		x = UpSampling2D((4, 4))(x) #(2,2) for nx=24
		print(x.shape)
		x = Conv2DTranspose(self.dimensions[2], (3, 3), padding='same', activation=self.activation_function)(x)
		print(x.shape)
		x = UpSampling2D((3, 3))(x) #(2,2) for nx=24
		print(x.shape)
		x = Conv2DTranspose(self.dimensions[1], (3, 3), padding='same', activation=self.activation_function)(x)
		print(x.shape)
		x = UpSampling2D((2, 2))(x)
		print(x.shape)
		x = Conv2DTranspose(self.dimensions[0], (3, 3), padding='same', activation=self.activation_function)(x)
		print(x.shape)
		x = UpSampling2D((2, 2))(x)
		print(x.shape)
		decoded = Conv2DTranspose(nu, (3, 3), activation='linear', padding='same')(x)
		print(decoded.shape)

		# Instantiate model
		self.autoencoder = tf.keras.models.Model(self.image, decoded)
		
		self.encoder = tf.keras.models.Model(self.image, [z_mean, z_log_var, z_sampled])
		
		encoded_input = Input(shape=(1, 1, encoded.shape[3]))  # latent vector definition

		#for l in self.autoencoder.layers:
		#	print(l)

		deco = self.autoencoder.layers[-9](encoded_input)  # re-use the same layers as the ones of the autoencoder
		for i in range(8):
			deco = self.autoencoder.layers[-8 + i](deco)
		self.decoder = tf.keras.models.Model(encoded_input, deco)

		# Loss tracker
		self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
		self.reconstruction_loss_tracker = keras.metrics.Mean(
			name="reconstruction_loss"
		)
		self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

	@property
	def metrics(self):
		return [
			self.total_loss_tracker,
			self.reconstruction_loss_tracker,
			self.kl_loss_tracker,
		]

	@property
	def pooling(self):
		"""
		Return pooling function
		:return: str, pooling function
		"""
		return self.pooling_function

	@pooling.setter
	def pooling(self, value):
		"""
		Set pooling function
		:param value: str, pooling function name ('max', 'avg')
		:return: str, pooling function
		"""
		if value == 'max':
			self.pooling_function = MaxPool2D
		elif value == 'avg':
			self.pooling_function = AveragePooling2D
		else:
			raise ValueError("Use a valid pooling function")

	def train_step(self, data):
		"""
		The custom training loop which minimises the reconstruction loss as well as Kullback-Leibler divergence.
		:param data: np.array
		"""
		with tf.GradientTape() as tape:
			z_mean, z_log_var, z_sampled = self.encoder(data)
			print('z_sampled', z_sampled.shape)
			z_reshaped = Reshape((1, 1, self.dimensions[3]))(z_sampled)

			reconstruction = self.decoder(z_reshaped)
			print('reconstruction', reconstruction.shape)
			reconstruction_loss = tf.reduce_mean(
				tf.reduce_sum(
					keras.losses.binary_crossentropy(data, reconstruction),
					axis=(1, 2),
				)
			)

			kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
			kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
			total_loss = reconstruction_loss + kl_loss

		grads = tape.gradient(total_loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
		self.total_loss_tracker.update_state(total_loss)
		self.reconstruction_loss_tracker.update_state(reconstruction_loss)
		self.kl_loss_tracker.update_state(kl_loss)

		return {
			"loss": self.total_loss_tracker.result(),
			"reconstruction_loss": self.reconstruction_loss_tracker.result(),
			"kl_loss": self.kl_loss_tracker.result(),
		}

	def call(self, inputs):
		self.autoencoder.call(inputs)


class VAE_Encoder():
	"""
	The encoder part of the VAE (not suitable for training)
	"""
	def __init__(self, encoder_file: str):
		dir_curr = os.path.split(__file__)[0]
		self.encoder = load_model(os.path.join(dir_curr, encoder_file))

	def forward(self, x_input):
		z_mean, z_log_var, z_sampled = self.encoder(x_input)

		return z_mean


class VAE_Decoder():
	"""
	The decoder part of the VAE (not suitable for training)
	"""
	def __init__(self, decoder_file: str):
		dir_curr = os.path.split(__file__)[0]
		self.decoder = load_model(os.path.join(dir_curr, decoder_file))

	def forward(self, z_input):
		x = self.decoder(z_input)

		return x



#############################
# Functions from ClassAE.py #
#############################


def data_reading(re: float = 40.0, nx: int = 24, nu: int = 2, shuf: bool = False, filename=None) -> np.array:
	"""
	Function to read H5 files with flow data, can change Re to run for different flows
	:param re: float, Reynolds Number (20.0, 30.0, 40.0, 50.0, 60.0, 100.0, 180.0)
	:param nx: int, size of the grid
	:param nu: int, components of the velocity vector (1, 2)
	:param shuf: boolean, if true returns shuffled data
	:return: numpy array, Time series for given flow with shape [#frames, nx, nx, nu]
	"""

	# File selection
	# T has different values depending on Re
	if re == 20.0 or re == 30.0 or re == 40.0:
		T = 20000
	else:
		T = 2000

	dir_curr = os.path.split(__file__)[0]
	if filename is not None:
		print(filename)
		path_rel = ('SampleFlows', filename)
	else:
		path_rel = ('SampleFlows', f'Kolmogorov_Re{re}_T{T}_DT01.h5')

	path = os.path.join(dir_curr, *path_rel)

	print(path)

	# Read dataset
	hf = h5py.File(path, 'r')
	t = np.array(hf.get('t'))
	# Instantiating the velocities array with zeros
	u_all = np.zeros((nx, nx, len(t), nu))

	# Update u_all with data from file
	u_all[:, :, :, 0] = np.array(hf.get('u_refined'))
	if nu == 2:
		u_all[:, :, :, 1] = np.array(hf.get('v_refined'))

	# Transpose of u_all in order to make it easier to work with it
	# Old dimensions -> [nx, nx, frames, nu]
	# New dimensions -> [frames, nx, nx, nu]
	u_all = np.transpose(u_all, [2, 0, 1, 3])
	hf.close()

	# Shuffle of the data in order to make sure that there is heterogeneity throughout the test set
	if shuf:
		u_all = shuffle(u_all, random_state=42)

	return u_all


def preprocess(u_all: np.array or None = None, re: float = 40.0, nx: int = 24, nu: int = 2, split: bool = True, norm: bool = True, filename=None) -> np.array or tuple[np.array]:
	"""
	Function to preprocess the dataset. It can split into train validation and test, and normalize the values
	:param u_all: numpy array, optional, time series flow velocities
	:param re: float, Reynolds Number (20.0, 30.0, 40.0, 50.0, 60.0, 100.0, 180.0)
	:param nx: int, size of the grid
	:param nu: int, components of the velocity vector (1, 2)
	:param split: bool, if True the data will be divided among train (75%), validation (20%) and test (5%)
	:param norm: bool, if True the data will be normalized for values between 0 and 1
	:return: numpy array(s), depending on "split" it will return the velocity time series after processing
	"""

	# Scenario where no data is provided by the user
	if u_all is None:
		u_all = data_reading(re, nx, nu, filename=filename)

	# Normalize data
	if norm:
		u_min = np.amin(u_all[:, :, :, 0])
		u_max = np.amax(u_all[:, :, :, 0])
		u_all[:, :, :, 0] = (u_all[:, :, :, 0] - u_min) / (u_max - u_min)
		if nu == 2:
			# Code to run if using velocities in 'y' direction as well
			v_min = np.amin(u_all[:, :, :, 1])
			v_max = np.amax(u_all[:, :, :, 1])
			u_all[:, :, :, 1] = (u_all[:, :, :, 1] - v_min) / (v_max - v_min)

	# Division of training, validation and testing data
	if split:
		val_ratio = int(np.round(0.75 * len(u_all)))
		test_ratio = int(np.round(0.95 * len(u_all)))

		u_train = u_all[:val_ratio, :, :, :].astype('float32')
		u_val = u_all[val_ratio:test_ratio, :, :, :].astype('float32')
		u_test = u_all[test_ratio:, :, :, :].astype('float32')
		return u_train, u_val, u_test

	return u_all


def performance(autoencoder, u_test, batch=10) -> dict[str, float]:
	"""
	Function to create dictionary with various performance metrics.
	Keys - 'mse', 'abs_percentage', 'abs_std', 'sqr_percentage', 'sqr_std'
	:return: dict, performance metrics
	"""
	d = dict()
	y_pred = autoencoder.predict(u_test)
	
	# Calculation of MSE
	#d['mse'] = autoencoder.evaluate(u_test, u_test, verbose=0)

	# Absolute percentage metric
	d['abs_percentage'] = np.average(1 - np.abs(y_pred - u_test) / u_test) * 100
	sqr_average_images = np.average((1 - np.abs(y_pred - u_test) / u_test), axis=(1, 2)) * 100
	d['abs_std'] = np.std(sqr_average_images)

	# Squared percentage metric, along with std
	d['sqr_percentage'] = np.average(1 - (y_pred - u_test) ** 2 / u_test) * 100
	sqr_average_images = np.average((1 - (y_pred - u_test) ** 2 / u_test), axis=(1, 2)) * 100
	d['sqr_std'] = np.std(sqr_average_images)

	return d

#Training loop
if __name__ == '__main__':
	# Load data
	u_train, u_val, u_test = preprocess(filename='Kolmogorov_Re40.0_T6000_DT001_res33.h5', nx=48)

	# Load model
	model = VAE(nx=48)

	# Train model
	model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00005)) #0.0005
	early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_total_loss', patience=15)
	model.fit(u_train, epochs=100, batch_size=10, shuffle=False, validation_data=(u_val, u_val), verbose=1, callbacks=[early_stop_callback]) #

	# Print performance metrics
	metrics = performance(model.autoencoder, u_test)
	print('abs accuracy:', f"{metrics['abs_percentage']} +- {metrics['abs_std']}")
	print('sqr accuracy:', f"{metrics['sqr_percentage']} +- {metrics['sqr_std']}")

	# Save model
	model.autoencoder.save('vae_ph_v5.0.h5')
	model.encoder.save('vae_encoder_ph_v5.0.h5')
	model.decoder.save('vae_decoder_ph_v5.0.h5')
