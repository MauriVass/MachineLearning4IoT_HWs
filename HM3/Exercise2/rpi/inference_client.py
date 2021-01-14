import time
import datetime
import json
from board import D4
import adafruit_dht
import pyaudio
import wave
import base64
import tensorflow as tf
import numpy as np
import zlib
import os

import sys
sys.path.insert(0, './../../')
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
	LABELS = readFile('labels.txt')[0].split(' ')
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


class Receiver(DoSomething):
	def __init__(self,clientID):
		super().__init__(clientID)
		self.predictions = {}

	def notify(self, topic, msg):
		#To do checks on input
		r = msg.decode('utf-8')
		r = json.loads(r)
		#print(r)
		id = int(r['id'])
		prediction = r['prediction']

		#print('Predction: ', prediction)
		#print('Before: ', self.predictions[id])
		self.predictions[id].append(prediction)
		#print('After: ', self.predictions[id])
		#print()


if __name__ == "__main__":
	sp, test_files, LABELS = LoadData()

	client_rpi = Receiver("ClientRpi")
	client_rpi.run()
	idtopic = '/Group14_ML4IoT/'
	client_rpi.myMqttClient.mySubscribe(idtopic+'prediction/')


	ip = 'http://169.254.37.210/'
	total_test_size = len(test_files)
	for i,t in enumerate(test_files):
		print(f'Progress: {i}/{total_test_size}',end='\r')
		data, label, _ = sp.PreprocessAudio(t)

		audio_b64_bytes = base64.b64encode(data)
		audio_string = audio_b64_bytes.decode()

		'''
		print(data)
		print(type(audio_b64_bytes), len(audio_b64_bytes), audio_b64_bytes[:100])
		print(type(audio_string),audio_string[:100])

		after = audio_string.encode()
		#print(type(after),len(after), after[:100])
		ops = base64.b64decode(after)
		#print(type(ops),len(ops), ops[:100])
		hope = tf.io.decode_raw(ops,tf.float32) #encode_base64(ops)
		#print(hope)
		asok1 = tf.reshape(hope,[32,32,1])
		print(asok1)
		#exit()
		'''
		#print(data.shape)
		timestamp = int(datetime.datetime.now().timestamp())
		body = {
					'bn' : ip,
					'bi' : int(timestamp),
					'e' : [{'n':'audio', 'u':'/', 't':0, 'vd': audio_string, 'dims': list(data.shape), 'id': i }]
				}
		#exit()
		body = json.dumps(body)
		#Avoid printing on screen the msg published (Changed MyMQTT.py file)
		client_rpi.myMqttClient.myPublish(idtopic+"audio/" ,body, print_msg=False)
		#Convert label to string since the other clients send the prediction as string
		client_rpi.predictions[i] = [str(label.numpy())]

	#Wait to all the prediction to arrive. FIND A BETTER WAY!!!!
	time.sleep(10)

	accuracy = 0
	print('Whole pred dict: ', client_rpi.predictions)
	for k,v in client_rpi.predictions.items():
		true_label = v[0]
		#print('Ture label', type(true_label))
		predictions = v[1:]

		major_pred = {}
		#print('Predcitions: ', predictions)
		for p in predictions:
			if(p in major_pred.keys()):
				major_pred[p] = major_pred[p] + 1
			else:
				major_pred[p] = 1

		pred_votes = -1
		pred_label = ''
		for k,v in major_pred.items():
			if(v>pred_votes):
				pred_votes = v
				pred_label = k

		print(f'{major_pred}, Winning label: {pred_label} with {pred_votes} votes. (True label: {true_label}, Correct? {pred_label==true_label})')
		#print(pred_label, true_label, type(pred_label), type(true_label))

		if(pred_label==true_label):
			accuracy+=1

	print(f'Accuracy: {(accuracy/total_test_size*100):.3f}%')

	client_rpi.end()
