#Change tf version (Only colab)
# !pip install tensorflow==2.3.0
# import tensorflow as tf
# print(tf.__version__)

#Keyword Spotting
import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
#Only colab
# !pip install tensorflow_model_optimization
import tensorflow_model_optimization as tfmot


parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, choices=['big','little'], required=True, help='model version: big or little')
args = parser.parse_args()


#Set a seed to get repricable results
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

class SignalGenerator:
	def __init__(self, labels, sampling_rate, frame_length, frame_step, num_mel_bins=None, lower_frequency=None, upper_frequency=None, num_coefficients=None, mfcc=False, image_size=32):
		self.labels=labels
		self.sampling_rate=sampling_rate
		self.frame_length=frame_length
		self.frame_step=frame_step
		self.num_mel_bins = num_mel_bins
		self.lower_frequency = lower_frequency
		self.upper_frequency = upper_frequency
		self.num_coefficients = num_coefficients
		self.mfccs=mfcc
		self.image_size = image_size

		if(mfcc):
			num_spectrogram_bins = frame_length // 2 + 1
			self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
					self.num_mel_bins,
					num_spectrogram_bins,
					self.sampling_rate,
					self.lower_frequency,
					self.upper_frequency)
			self.preprocess = self.preprocess_with_mfcc
		else:
			self.preprocess = self.preprocess_with_stft


	def read(self, file_path):
		parts = tf.strings.split(file_path, os.path.sep)
		label = parts[-2]
		label_id = tf.argmax(label == self.labels)
		audio_bynary = tf.io.read_file(file_path)
		audio, _ = tf.audio.decode_wav(audio_bynary)
		#print('Sampling: ', np.array(r))
		audio = tf.squeeze(audio, axis=1)
		return audio, label_id

	def pad(self, audio):
		zero_padding = tf.zeros(self.sampling_rate - tf.shape(audio), dtype=tf.float32)
		audio = tf.concat([audio,zero_padding],0)
		audio.set_shape([self.sampling_rate])
		return audio

	def get_spectrogram(self, audio):
		#Calculate the STFT of the signal given frame_length and frame_step
		stft = tf.signal.stft(audio,
						frame_length=self.frame_length,
						frame_step=self.frame_step,
						fft_length=self.frame_length)
		#Transform the complex number in real number
		spectrogram = tf.abs(stft)
		return spectrogram

	def get_mfccs(self, spectrogram):
		mel_spectrogram = tf.tensordot(spectrogram,
						self.linear_to_mel_weight_matrix, 1)
		log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
		mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
		mfccs = mfccs[:, :self.num_coefficients]
		return mfccs

	def preprocess_with_stft(self, file_path):
		audio, label = self.read(file_path)
		audio = self.pad(audio)
		spectrogram = self.get_spectrogram(audio)
		spectrogram = tf.expand_dims(spectrogram, -1)
		spectrogram  = tf.image.resize(spectrogram, [self.image_size,self.image_size])
		return spectrogram, label

	def preprocess_with_mfcc(self, file_path):
		audio, label = self.read(file_path)
		audio = self.pad(audio)
		spectrogram = self.get_spectrogram(audio)
		mfccs = self.get_mfccs(spectrogram)
		mfccs = tf.expand_dims(mfccs, -1)
		return mfccs, label

	def make_dataset(self, files, train=False):
		#This method creates a dataset from a numpy array (our listfile path)
		ds = tf.data.Dataset.from_tensor_slices(files)
		#Different preprocess step depending on the input parameter
		ds = ds.map(self.preprocess, num_parallel_calls=4)
		ds = ds.batch(32)
		ds = ds.cache()

		if(train is True):
			ds = ds.shuffle(100, reshuffle_each_iteration=True)
		return ds

#Sparcity increases latency due to cache misses
class Model:
	def __init__(self,model_type,frame_length,frame_step,mfcc,num_mel_bins,num_coefficients,train_ds,image_size=32,alpha=1,sparsity=None):
		self.frame_length = frame_length
		self.frame_step = frame_step
		self.image_size = image_size
		self.num_coefficients = num_coefficients
		print('Summary: ',frame_length,frame_step,mfcc,num_mel_bins,num_coefficients,train_ds,alpha,sparsity)
		self.alpha = alpha
		self.sparsity=sparsity
		self.n_output = 8
		if(mfcc):
			self.strides = [2,1]
		else:
			self.strides = [2,2]

		self.model_type = model_type
		if(model_type=='MLP'):
			self.model = self.MLPmodel()
		elif(model_type=='CNN'):
			self.model = self.CNNmodel()
		elif(model_type=='DSCNN'):
			self.model = self.DSCNNmodel()
		else:
			raise KeyError('SPECIFY A MODEL TYPE [MLP, CNN, DSCNN]')

		self.mfcc = mfcc

		#CALLBACKS
		self.callbacks = []
		self.checkpoint_path = 'KSckp/'
		monitor = 'val_sparse_categorical_accuracy'
		self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
			filepath=self.checkpoint_path,
			save_weights_only=True,
			monitor=monitor,
			mode='max',
			save_best_only=True)
		self.callbacks.append(self.model_checkpoint_callback)

		self.early_stopping = tf.keras.callbacks.EarlyStopping(
			monitor=monitor, min_delta=0, patience=4, verbose=1, mode='auto',
			baseline=None, restore_best_weights=True)
		#self.callbacks.append(self.early_stopping)

		#self.lr_exp = tf.keras.callbacks.LearningRateScheduler(my_schedule, verbose=1)
		#self.callbacks.append(self.lr_exp)
		self.lr_onplateau = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.1,
			patience=3, min_lr=0.0001, verbose=1)
		#self.callbacks.append(self.lr_onplateau)

		self.sparsity = sparsity
		if(self.sparsity is not None):
			pruning_params = {
				'pruning_schedule':
				tfmot.sparsity.keras.PolynomialDecay(
				initial_sparsity=0.30,
				final_sparsity=sparsity,
				begin_step=len(train_ds)*3,
				end_step=len(train_ds)*15)}

			prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
			self.model = prune_low_magnitude(self.model, **pruning_params)
			self.model_sparcity_callback = tfmot.sparsity.keras.UpdatePruningStep()
			self.callbacks.append(self.model_sparcity_callback)
			self.callbacks.append(tfmot.sparsity.keras.PruningSummaries(log_dir='PruningSumm/'))

			dim1 = ((16000-frame_length)/frame_step)+1
			#print(frame_length,frame_step,dim1)
			if(mfcc):
				input_shape = [None, int(dim1) , num_coefficients, 1]
			else:
				input_shape = [None, image_size, image_size, 1]
			print('Input Shape Sparsity: ', input_shape)
			self.model.build(input_shape)

		self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
											 metrics=['sparse_categorical_accuracy'])

	def MLPmodel(self):
		model = keras.Sequential([
			keras.layers.Flatten(),
			keras.layers.Dense(int(256*self.alpha), activation='relu'),
			keras.layers.Dense(int(256*self.alpha), activation='relu'),
			keras.layers.Dense(int(256*self.alpha), activation='relu'),
			keras.layers.Dense(self.n_output)
			])
		return model

	#Strides = [2,2] if STFT, [2,1] if MFCC
	def CNNmodel(self):
		model = keras.Sequential([
			keras.layers.Conv2D(filters=int(128),kernel_size=[3,3],strides=self.strides,use_bias=False),
			keras.layers.BatchNormalization(momentum=0.1),
			keras.layers.Activation('relu'),
			keras.layers.Conv2D(filters=int(256*self.alpha),kernel_size=[3,3],strides=[1,1],use_bias=False),
			keras.layers.BatchNormalization(momentum=0.1),
			keras.layers.Activation('relu'),
			keras.layers.Conv2D(filters=int(256*self.alpha),kernel_size=[3,3],strides=[1,1],use_bias=False),
			keras.layers.BatchNormalization(momentum=0.1),
			keras.layers.Activation('relu'),
			keras.layers.GlobalAveragePooling2D(),
			keras.layers.Dense(int(256*self.alpha), activation='relu'),
			keras.layers.Dense(self.n_output)
			])
		return model

	def DSCNNmodel(self):
		model = keras.Sequential([
			keras.layers.Conv2D(filters=int(256*self.alpha),kernel_size=[3,3],strides=self.strides,use_bias=False), #input_shape=(32,32,1)
			keras.layers.BatchNormalization(momentum=0.1),
			keras.layers.Activation('relu'),
			keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
			keras.layers.Conv2D(filters=int(256*self.alpha),kernel_size=[1,1],strides=[1,1],use_bias=False),
			keras.layers.BatchNormalization(momentum=0.1),
			keras.layers.Activation('relu'),
			keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
			keras.layers.Conv2D(filters=int(256*self.alpha),kernel_size=[1,1],strides=[1, 1],use_bias=False),
			keras.layers.BatchNormalization(momentum=0.1),
			keras.layers.Activation('relu'),
			keras.layers.GlobalAveragePooling2D(),
			keras.layers.Dense(self.n_output)
			])
		return model

	def Train(self,train,validation,epoch):
		#TO REMOVE, SET TO FALSE
		if(False):
			for c in self.callbacks:
				print(c)
		print('Training')
		history = self.model.fit(train, batch_size=32, epochs=epoch, verbose=1,
				validation_data=validation, validation_freq=1, callbacks=self.callbacks)
		return history

	def Test(self, test, best=True):
		print('Evaluation')
		if(best):
				self.model.load_weights(self.checkpoint_path)
		loss, error = self.model.evaluate(test, verbose=1)
		return (loss, error)

	#TO ADJUST (OUTPUT SHOULD BE: little, big)
	def SaveModel(self,output,best=True):
		if(best):
			self.model.load_weights(self.checkpoint_path)
		if(self.sparsity is not None):
			self.Strip()
		print(f'Saving: {output}')
		self.model.save(output)
		return output

	def Strip(self):
		self.model = tfmot.sparsity.keras.strip_pruning(self.model)

#Download and extract the .csv file. The result is cached to avoid to download everytime
zip_path = tf.keras.utils.get_file(
	origin='http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip',
	fname='mini_speech_commands.zip',
	extract=True,
	cache_dir='.', cache_subdir='data')

data_dir = os.path.join('.', 'data', 'mini_speech_commands')

def readFile(file):
	elems = []
	fp = open(file,'r')
	for f in fp:
		elems.append(f.strip())
	return elems

#Spit the dataset following the .pdf requirements
if(os.path.exists('Temp')==False):
	!git clone https://github.com/MauriVass/Temp.git
train_files = readFile('Temp/ML4IoT/kws_train_split.txt')
validation_files = readFile('Temp/ML4IoT/kws_val_split.txt')
test_files = readFile('Temp/ML4IoT/kws_test_split.txt')
LABELS = readFile('Temp/ML4IoT/labels.txt')[0].split(' ')

print(LABELS)

# for m in ['MLP', 'CNN', 'DSCNN']:
#   for f in [False, True]:

model_version = args.version

### MAIN PARAMETERS ###
if(model_version=='little'):
	#It can be ['MLP', 'CNN', 'DSCNN']
	model = 'DSCNN'
	#It can be: True, False
	mfcc = True
	#It can be (0,1]
	alpha = 0.234
	#Sparcity increases latency(may be a problem for KS) due to cache misses
	#it can be (0.3,1) or None(if you don't to use sparsity)
	sparsity = None

	#Here you can change:
	#STFT(mfcc=False): frame_length, frame_step
	#MFCC(mfcc=True): frame_length, frame_step, num_mel_bins, num_coefficients
	frame_length = 480  #Default 640 (mfcc=True), 256(mfcc=False)
	frame_step = 320 #Default 320 (mfcc=True), 128(mfcc=False)
	num_mel_bins = 40 #Default 40 (only mfcc=True)
	num_coefficients = 10 #Default 10 (only mfcc=True)
	image_size = 32 #Default 32 (only mfcc=False)
else:
	#It can be ['MLP', 'CNN', 'DSCNN']
	model = 'DSCNN'
	#It can be: True, False
	mfcc = True
	#It can be (0,1]
	alpha = 1.5
	#Sparcity increases latency(may be a problem for KS) due to cache misses
	#it can be (0.3,1) or None(if you don't to use sparsity)
	sparsity = None

	#Here you can change:
	#STFT(mfcc=False): frame_length, frame_step
	#MFCC(mfcc=True): frame_length, frame_step, num_mel_bins, num_coefficients
	frame_length = 640  #Default 640 (mfcc=True), 256(mfcc=False)
	frame_step = 320 #Default 320 (mfcc=True), 128(mfcc=False)
	num_mel_bins = 40 #Default 40 (only mfcc=True)
	num_coefficients = 10 #Default 10 (only mfcc=True)
	image_size = 32 #Default 32 (only mfcc=False)

if(mfcc):
	sg = SignalGenerator(labels=LABELS, sampling_rate=16000, frame_length=int(frame_length), frame_step=int(frame_step),
				num_mel_bins=int(num_mel_bins), lower_frequency=20, upper_frequency=4000, num_coefficients=int(num_coefficients), mfcc=mfcc)
else:
	sg = SignalGenerator(labels=LABELS, sampling_rate=16000, frame_length=frame_length, frame_step=frame_step, image_size=image_size)
### END MAIN PARAMETERS ###

train_ds = sg.make_dataset(train_files,True)
val_ds = sg.make_dataset(validation_files)
test_ds = sg.make_dataset(test_files)
# print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')

# for x,y in train_ds.take(1):
#   print(x.shape,y.shape)

save_best = True
model = Model(model,frame_length=int(frame_length),frame_step=int(frame_step),mfcc=mfcc,train_ds=train_ds,num_mel_bins=int(num_mel_bins),num_coefficients=int(num_coefficients),image_size=image_size,alpha=alpha,sparsity=sparsity)
hist = model.Train(train_ds,val_ds,20)
loss, acc = model.Test(test_ds,save_best)
print('Accuracy test set: ',acc)
output_model = model.SaveModel(model_version,save_best)
# model.model.summary()

#Deployer, Optimizer W_WA
import argparse
import tensorflow as tf
import os

def representative_dataset_gen():
		for x, _ in train_ds.take(1000):
				yield [x]

#TO ADJUST (OUTPUT NAME SHOLD BE <something>.tflite.zlib)
def Optimize(saved_model_dir,quantization,zipping):
	converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
	if(quantization=='w'):
			print('Only Weight')
			#Quantization Weights only
			converter.optimizations = [tf.lite.Optimize.DEFAULT]
			mini_float = False
			if(mini_float):
					converter.target_spec.supported_types = [tf.float16]
			# tflite_model_dir = saved_model_dir + '.tflite_W'
	elif(quantization=='wa'):
			print('Weight Activation')
			#Quantization Weights and Activation
			converter.optimizations = [tf.lite.Optimize.DEFAULT]
			converter.representative_dataset = representative_dataset_gen
			# tflite_model_dir = saved_model_dir + '.tflite_WA'
			
	tflite_model_dir = saved_model_dir + '.tflite'
	tflite_model = converter.convert()

	#Compression
	if(zipping is False):
			with open(tflite_model_dir, 'wb') as fp:
					fp.write(tflite_model)
	else:
			print('Compression')
			import zlib
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
#wa -> weights and activation (have some problem with the shape/last reshape layer (maybe))
if(model_version=='little'):
	quantization = 'w' 
	zipping = True
elif(model_version=='big'):
	quantization = 'no' 
	zipping = False

saved_model_dir = output_model
output_tflite_model = Optimize(saved_model_dir,quantization,zipping)

#Decompress
# import zlib
# model_path = '' 
# if(model_path.find('zlib')<0):
#   raise KeyError('YOU CAN\'T DECOMPRESS A NON .zlib MODEL')
# with open(model_path, 'rb') as fp:
#     model = zlib.decompress(fp.read())
#     output_model = model_path[:-5]
#     file = open(output_model,'wb')
#     print('Saving: ',output_model)
#     file.write(model)
#     file.close()

#          Destination      Origin
# !zip -r ./th_test_stft.zip ./th_test_stft

# !unzip THFmodelCNN.zip ./THFmodelCNN

#Test Models
if(False):
	import time
	import tensorflow.lite as tflite

	saved_model_dir = output_tflite_model
	if(saved_model_dir.find('zlib')>0):
		raise KeyError('YOU CAN\'T TEST A .zlib MODEL. (Use zipping=False in Optimize() method)')
	test_ds1 = test_ds.unbatch().batch(1)

	interpreter = tf.lite.Interpreter(model_path=saved_model_dir)
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	acc = 0
	n = 0
	time_infe = 0
	print(test_ds1)

	for x,y in test_ds1:
		#print(x,y)
		input_data = x
		y_true = y.numpy()[0]
		
		ti = time.time()
		interpreter.set_tensor(input_details[0]['index'], input_data)
		interpreter.invoke()
		my_output = interpreter.get_tensor(output_details[0]['index'])[0]
		time_infe += time.time()-ti

		n+=1
		index_pred = np.argmax(my_output)
		if(index_pred==y_true):
			acc += 1

	print(f'Accuracy: {(acc/n):.3f}, time: {(time_infe/n)*1000} ms')

