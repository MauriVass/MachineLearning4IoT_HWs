import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import time
import tensorflow_model_optimization as tfmot
import zlib

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='model version: a or b')
args = parser.parse_args()

#Set a seed to get repricable results
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

#Download and extract the .csv file. The result is cached to avoid to download everytime
zip_path = tf.keras.utils.get_file(
		origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
		fname='jena_climate_2009_2016.csv.zip',
		extract=True,
		cache_dir='.', cache_subdir='data')
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)

#Take the required columns
column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

#Separate the data in train, validation and test sets
n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]
print(f'Total length: {n}, Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}')

class WindowGenerator:
	def __init__(self, mean, std):
		self.input_width = 6
		self.output_width = 6
		self.label_options = 2
		self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
		self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

	def split_window(self, features):
		#print(1,features.shape)
		inputs = features[:, :-self.output_width, :]
		#print(2,inputs.shape)

		multi_step = True
		if(multi_step is False):
			labels = features[:, -self.output_width, :]
			#print(3,labels.shape)
			labels.set_shape([None, self.label_options])
			#print(5,labels.shape,'\n\n')
		else:
			labels = features[:, -self.output_width:, :]
			#print(3,labels.shape)
			labels.set_shape([None, self.output_width, self.label_options])
			#print(5,labels.shape,'\n\n')

		#labels = tf.expand_dims(labels, -1)

		inputs.set_shape([None, self.input_width, self.label_options])
		#print(4,inputs.shape)

		return inputs, labels

	def normalize(self, features):
		features = (features - self.mean) / (self.std + 1.e-6)
		return features

	def preprocess(self, features):
		inputs, labels = self.split_window(features)
		inputs = self.normalize(inputs)

		return inputs, labels

	def make_dataset(self, data, train):
		#The targets is None since the labels are already inside the data
		ds = tf.keras.preprocessing.timeseries_dataset_from_array(
						data=data,
						targets=None,
						sequence_length=self.input_width+self.output_width,
						sequence_stride=1,
						batch_size=32)
		ds = ds.map(self.preprocess)
		ds = ds.cache()
		if train is True:
			ds = ds.shuffle(100, reshuffle_each_iteration=True)

		return ds


#Calculate statistics for normalization
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

generator = WindowGenerator(mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')


class TempHumMAE(keras.metrics.Metric):
	def __init__(self, name='mean_absolute_error_cust', **kwargs):
		super().__init__(name, **kwargs)
		#initialiaze the variables used to calculate the loss
		self.count = self.add_weight(name='count', initializer='zeros')
		#The shape [2]('shape=(2,)' is equivalent) is for temperature ad humidity
		self.total = self.add_weight(name='total', initializer='zeros', shape=(2,))

	#Called at every batch of data
	def update_state(self, y_true, y_pred, sample_weight=None):
		#print('Prediction',y_pred)
		#print('True',y_true)
		error = tf.abs(y_pred-y_true)
		#Calculate mean over output_width and batch
		error = tf.reduce_mean(error, axis=(0,1))#
		#print(error)
		#You can just use + sign but it is better to use assign_add method
		self.total.assign_add(error)
		self.count.assign_add(1.)
		return
	def reset_states(self):
		self.count.assign(tf.zeros_like(self.count))
		self.total.assign(tf.zeros_like(self.total))
		return
	def result(self):
		results = tf.math.divide_no_nan(self.total, self.count)
		return results

def my_schedule(epoch, lr):
	if epoch < 10:
		return lr
	else:
		return lr * tf.math.exp(-0.1)

#https://www.tensorflow.org/tutorials/structured_data/time_series#single_step_models
class Model:
	def __init__(self,model_type,alpha=1,sparsity=None,version=''):
		self.alpha = alpha
		self.label=2
		self.n_output = 1 if self.label < 2 else 2
		self.metric = ['mae'] if self.label < 2 else [TempHumMAE()]
		self.model_type = model_type
		if(model_type=='MLP'):
			self.model = self.MLPmodel()
		elif(model_type=='CNN'):
			self.model = self.CNNmodel(alpha)
		elif(model_type=='LSTM'):
			self.model = self.LSTMmodel(alpha)
		self.output_name = f'Group14_th_{version}'

		#CALLBACKS
		self.callbacks = []
		self.checkpoint_path = 'THckp/'
		monitor_loss = 'mean_squared_error'
		self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath=self.checkpoint_path,
		save_weights_only=True,
		monitor=monitor_loss,
		mode='auto',
		save_best_only=True)
		self.callbacks.append(self.model_checkpoint_callback)

		self.early_stopping = tf.keras.callbacks.EarlyStopping(
		monitor=monitor_loss, min_delta=0.05, patience=3, verbose=1, mode='auto',
				baseline=None, restore_best_weights=True)
		#self.callbacks.append(self.early_stopping)

		self.lr_exp = tf.keras.callbacks.LearningRateScheduler(my_schedule, verbose=1)
		#self.callbacks.append(self.lr_exp)
		self.lr_onplateau = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_loss, factor=0.1,
				patience=2, min_lr=0.001, verbose=1)
		#self.callbacks.append(self.lr_onplateau)

		self.sparsity = sparsity
		if(self.sparsity is not None):
			prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
			self.model = prune_low_magnitude(self.model, **pruning_params)
			self.model_sparcity_callback = tfmot.sparsity.keras.UpdatePruningStep()
			self.callbacks.append(self.model_sparcity_callback)
			self.callbacks.append(tfmot.sparsity.keras.PruningSummaries(log_dir='PruningSumm/'))
			input_shape = [None, 6, 2]
			self.model.build(input_shape)

		self.model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),
									metrics=[self.metric,tf.keras.losses.MeanSquaredError()])
	 
	def MLPmodel(self):
		model = keras.Sequential([
					keras.layers.Flatten(),
					keras.layers.Dense(int(128*self.alpha), activation='relu'),
					keras.layers.Dense(int(128*self.alpha), activation='relu'),
					keras.layers.Dense(self.n_output*6),
					keras.layers.Reshape([6, 2])
				])
		return model
	def CNNmodel(self,alpha):
		model = keras.Sequential([
					keras.layers.Conv1D(filters=int(64*self.alpha),kernel_size=(3,),activation='relu'),
					keras.layers.Flatten(),
					keras.layers.Dense(int(64*self.alpha), activation='relu'),
					keras.layers.Dense(self.n_output*6),
					keras.layers.Reshape([6, 2])
				])
		return model
	def LSTMmodel(self,alpha):
		model = keras.Sequential([
					keras.layers.LSTM(units=int(64*alpha)),
					keras.layers.Flatten(),
					keras.layers.Dense(self.n_output*6),
					keras.layers.Reshape([6, 2])
				])
		return model

	def Train(self,train,validation,epoch):
		if(False):
			print('\nCallbacks used:')
			for c in self.callbacks:
				print(c)
			print()
		history = self.model.fit(train, batch_size=32, epochs=epoch, verbose=1,
												validation_data=validation, validation_freq=2, callbacks=self.callbacks)#
		return history

	def Test(self, test):
		self.model.load_weights(self.checkpoint_path)
		error = self.model.evaluate(test, verbose=1)
		return error[1]

	def SaveModel(self):
			self.model.load_weights(self.checkpoint_path)
			if(self.sparsity):
				self.Strip()
			run_model = tf.function(lambda x: self.model(x))
			concrete_func = run_model.get_concrete_function(tf.TensorSpec([1,6,2], tf.float32))

			print(f'\n### Saving model: {self.output_name} ###\n')
			self.model.save(self.output_name, signatures=concrete_func)
			return self.output_name

	def Strip(self):
		self.model = tfmot.sparsity.keras.strip_pruning(self.model)

### MAIN PARAMETERS ###
version = args.version
if(version=='a'):
	### Model A ###
	#It can be ['MLP', 'CNN', 'LSTM']
	model_type = 'MLP'
	#It can be (0,1]
	alpha = 0.25
	#Sparcity increases latency(may be a problem for KS) due to cache misses
	#it can be (0.3,1) or None(if you don't to use sparsity)
	sparsity = 0.9
	### ### ###
elif(version=='b'):
	### Model B ###
	#It can be ['MLP', 'CNN', 'LSTM']
	model_type = 'CNN'
	#It can be (0,1]
	alpha = 0.07
	#Sparcity increases latency(may be a problem for KS) due to cache misses
	#it can be (0.3,1) or None(if you don't to use sparsity)
	sparsity = 0.7
	### ### ###

pruning_params = {
	'pruning_schedule':
		tfmot.sparsity.keras.PolynomialDecay(
		initial_sparsity=0.30,
		final_sparsity=sparsity,
		begin_step=len(train_ds)*3,
		end_step=len(train_ds)*15)}

model = Model(model_type,alpha=alpha,sparsity=sparsity,version=version)  
#init = time.time()
hist = model.Train(train_ds, val_ds, 20)
# end = time.time()
# print(f'{end-init}')

error = model.Test(test_ds)
temp_loss, hum_loss = error
print(f'Loss: Temp={temp_loss}, Hum={hum_loss}')
output_model = model.SaveModel()

print(model.model.summary())

#Deployer, Optimizer W_WA
def representative_dataset_gen():
		for x, _ in train_ds.take(1000):
				yield [x]

def Optimize(saved_model_dir,quantization,zipping):
	converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
	if(quantization=='w'):
			#Quantization Weights only
			# print('Only Weight')
			converter.optimizations = [tf.lite.Optimize.DEFAULT]
			mini_float = False
			if(mini_float):
					converter.target_spec.supported_types = [tf.float16]
			# tflite_model_dir = saved_model_dir + '.tflite_W'
	elif(quantization=='wa'):
			#Quantization Weights and Activation
			# print('Weight Activation')
			converter.optimizations = [tf.lite.Optimize.DEFAULT]
			converter.representative_dataset = representative_dataset_gen
			# tflite_model_dir = saved_model_dir + 'tflite_WA'

	tflite_model_dir = saved_model_dir + '.tflite'
	tflite_model = converter.convert()

	#Compression
	if(zipping is False):
			with open(tflite_model_dir, 'wb') as fp:
					fp.write(tflite_model)
	else:
			# print('Compression')
			tflite_model_dir = tflite_model_dir + '.zlib'
			with open(tflite_model_dir, 'wb') as fp:
					tflite_compressed = zlib.compress(tflite_model)#,level=9
					fp.write(tflite_compressed)

	print('Saving: ', tflite_model_dir)
	size_tflite_model = os.path.getsize(tflite_model_dir)
	print(f'Tflite Model size: {(size_tflite_model/1024):.2f} kB')
	return tflite_model_dir

#Optimization for TH Forecasting
#any -> none
#w -> only weights
#wa -> weights and activation
#Both cases: a and b
quantization = 'w' 
zipping = True
saved_model_dir = output_model
output_tflite_model = Optimize(saved_model_dir,quantization,zipping)

# #Test Models

# saved_model_dir = output_tflite_model
# if(saved_model_dir.find('zip')>0):
#   raise KeyError('YOU CAN\'T TEST A .zip MODEL. (Use zipping=False in Optimize() method)')

# test_ds = test_ds.unbatch().batch(1)

# interpreter = tf.lite.Interpreter(model_path=saved_model_dir)
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# mae = [0,0]
# n = 0
# time_infe = 0
# #print(test_ds)

# for x,y in test_ds:
#   #print(x,y)
#   input_data = x
#   y_true = y.numpy()[0]
  
#   ti = time.time()
#   interpreter.set_tensor(input_details[0]['index'], input_data)
#   interpreter.invoke()
#   my_output = interpreter.get_tensor(output_details[0]['index'])[0]
#   time_infe += time.time()-ti

#   n+=1
#   #mae[0] += np.abs(y[0] - my_output[0])
#   #mae[1] += np.abs(y[1] - my_output[1])
#   error = tf.abs(my_output-y_true)
#   mae += tf.reduce_mean(error, axis=(0,))

# print(f'MAE: temp: {mae[0]/n}, humi: {mae[1]/n}, time: {(time_infe/n)*1000} ms')
