import time
import datetime
import json
from board import D4
import adafruit_dht
import pyaudio
import wave
import base64
import numpy as np
import tensorflow as tf
import zlib
import os
import requests #REST

import sys #MQTT
sys.path.insert(0, './../')
from DoSomething import DoSomething

#Set a seed to get repricable results
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Decompress TO REMOVE OR NOT IDK
def Decompress(model_path):
	if(model_path.find('zlib')<0):
		raise KeyError('YOU CAN\'T DECOMPRESS A NON .zlib MODEL')
	with open(model_path, 'rb') as fp:
		model = zlib.decompress(fp.read())
		output_model = model_path[:-5]
		file = open(output_model,'wb')
		# print('Saving: ',output_model)
		file.write(model)
		file.close()
	return output_model

class SignalPreprocessor:
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
		return spectrogram, label, audio

	def preprocess_with_mfcc(self, file_path):
		audio, label = self.read(file_path)
		audio = self.pad(audio)
		spectrogram = self.get_spectrogram(audio)
		mfccs = self.get_mfccs(spectrogram)
		mfccs = tf.expand_dims(mfccs, -1)
		return mfccs, label, audio

	def PreprocessAudio(self, file):
		data, label, audio = self.preprocess(file)

		return data, label, audio

def readFile(file):
	elems = []
	fp = open(file,'r')
	for f in fp:
		elems.append(f.strip())
	return elems

def LoadData():
	#Download and extract the .csv file. The result is cached to avoid to download everytime
	zip_path = tf.keras.utils.get_file(
		origin='http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip',
		fname='mini_speech_commands.zip',
		extract=True,
		cache_dir='.', cache_subdir='data')

	data_dir = os.path.join('.', 'data', 'mini_speech_commands')

	#Spit the dataset following the .pdf requirements
	#Only Colab
	# if(os.path.exists('Temp')==False):
	# 	!git clone https://github.com/MauriVass/Temp.git
	# train_files = readFile('Temp/kws_train_split.txt')
	# validation_files = readFile('Temp/kws_val_split.txt')
	test_files = readFile('kws_test_split.txt')
	LABELS = readFile('labels.txt')
	print(LABELS)

	mfcc = True
	frame_length = 640  #Default 640 (mfcc=True), 256(mfcc=False)
	frame_step = 320 #Default 320 (mfcc=True), 128(mfcc=False)
	num_mel_bins = 40 #Default 40 (only mfcc=True)
	num_coefficients = 10 #Default 10 (only mfcc=True)
	image_size = 32 #Default 32 (only mfcc=False)
	if(mfcc):
		sp = SignalPreprocessor(labels=LABELS, sampling_rate=16000, frame_length=frame_length, frame_step=frame_step,
			num_mel_bins=num_mel_bins, lower_frequency=20, upper_frequency=4000, num_coefficients=num_coefficients, mfcc=mfcc)
	else:
		sp = SignalPreprocessor(labels=LABELS, sampling_rate=16000, frame_length=frame_length, frame_step=frame_step, image_size=image_size)

	return sp, test_files, LABELS

class Model:
	def __init__(self, model_path):
		self.model_path = model_path

		if(model_path.find('zip')>0):
			raise KeyError('YOU CAN\'T TEST A .zip MODEL. (Use zipping=False in Optimize() method)')
		self.interpreter = tf.lite.Interpreter(model_path=model_path)
		self.interpreter.allocate_tensors()

		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()

	def Evaluate(self,data):
		#print(self.input_details[0]['index'])
		#print(self.output_details[0]['index'])
		#print(data.shape)
		self.interpreter.set_tensor(self.input_details[0]['index'], data)
		self.interpreter.invoke()
		output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

		return output

class Receiver(DoSomething):
	def __init__(self,clientID):
		super().__init__(clientID)
		#received
	def notify(self, topic, msg):
		r = msg.decode('utf-8')
		r = json.loads(r)
		events = r['e']
		if(len(events)==1):
			prediction = events[0]['v']
			#Then how to pass the result to main???


if __name__ == "__main__":

	#client_rpi = Receiver("ClientRpi")
	#client_rpi.run()
	#idtopic = '/Group14_ML4IoT/'
	#client_rpi.myMqttClient.mySubscribe(idtopic+'prediction/')

	sp, test_files, LABELS = LoadData()

	model_compressed = 'models/' + 'KS_DSCNNTruespars0.9.tflite_W.zlib'
	model_path = Decompress(model_compressed)
	little_model = Model(model_path)


	ip = 'http://169.254.37.210/'
	accuracy = 0
	communication_cost = 0
	threshold_accuracy = 0.1
	for t in test_files:
		data, true_label, audio = sp.PreprocessAudio(t)

		data = tf.expand_dims(data, axis=0)
		output_layer_prediction = little_model.Evaluate(data)
		print(output_layer_prediction)
		output_layer_prediction = tf.nn.softmax(output_layer_prediction).numpy() #[0]
		print(output_layer_prediction)
		#best_predictions = tf.math.top_k(output_layer_prediction, k=2, sorted=True).values.numpy()
		#print(best_predictions)
		#Get the 2 top predictions
		top1 = [0,0]
		top2 = [0,0]
		for i ,v in enumerate(output_layer_prediction):
			#Update the top2. if v>top2 then maybe v could be greater than top1 too so check also top1<top2. If v<top2 it is useless to check for v>top1
			if(top2[1]<v):
				top2 = [i,v]
				if(top1[1]<top2[1]):
					top1, top2 = top2, top1

		#print(top1,top2)

		if(top1[1]-top2[1]<threshold_accuracy):
			print(f'Ask for help to the big model (diff={top1[1]-top2[1]}). Actually, the little model was: {top1[0]==true_label}')

			audio_b64_bytes = base64.b64encode(audio)
			audio_string = audio_b64_bytes.decode()

			timestamp = int(datetime.datetime.now().timestamp())
			body = {
						'bn' : ip,
						'bi' : int(timestamp),
						'e' : [{'n':'audio', 'u':'/', 't':0, 'vd': audio_string}]
					}
			body = json.dumps(body)
			communication_cost += len(body)
			#client_rpi.myMqttClient.myPublish(idtopic+"audio/" ,body ,False)

			#Web service address
			url = 'http://localhost:8080/'
			#The json.dump() is done automatically
			r = requests.put(url, json=body)

			if(r.satus_code):
				rbody = r.json()
				label = rbody['label']
				prob = rbody['probability']

				print(f'{label} ({prob}%)')
			else:
				raise KeyError(r.text)
		else:
			label = top1[0]

		if(label==true_label):
			accuracy += 1

	print(f'Accuracy: {(accuracy/len(test_files)*100):.3f}%')
	print(f'Communication Cost: {(communication_cost/1024**2):.3f}')

	client_rpi.end()
