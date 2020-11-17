import os
import io
import time
import numpy as np
import tensorflow as tf
import pyaudio
from scipy import signal
from subprocess import Popen
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num-samples', type=int, help='number samples', required=True)
parser.add_argument('--output', type=str, help='output file', required=True)
args = parser.parse_args()

#Action to be called inside popen() function abount governor and monitor
set_powersave = ['sudo', 'sh',  '-c', "echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"]
set_performance = ['sudo', 'sh',  '-c', "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"]
reset_monitor = ['sudo', 'sh',  '-c',  "echo 1 > /sys/devices/system/cpu/cpufreq/policy0/stats/reset"]
print_monitor = ['cat', '/sys/devices/system/cpu/cpufreq/policy0/stats/time_in_state']


class Mic:
	def __init__(self,dur,rate,res):
		self.input_res = int(res)
		if(self.input_res==8):
			self.resolution = pyaudio.paInt8
		elif(self.input_res==16):
			self.resolution = pyaudio.paInt16
		elif(self.input_res==32):
			self.resolution = pyaudio.paInt32
		self.channels = 1
		self.rate = rate
		self.chunk = 4800
		self.record_seconds = dur

		self.audio = pyaudio.PyAudio()
		self.stream = self.audio.open(format=self.resolution,
				channels=self.channels,
				rate= self.rate, input=True,
				frames_per_buffer=self.chunk)
		self.counter = 0

	def Record(self):
		time_start = time.time()
		max_val = int(self.rate / self.chunk * self.record_seconds)
		buffer = io.BytesIO()
		self.stream.start_stream()
		for i in range(max_val):
			#Set freq to powersave
			if(i == 0):
				Popen(set_powersave)
			data = self.stream.read(self.chunk)
			buffer.write(data)
			#Set freq to performance
			if(i==max_val-1-1):
				Popen(set_performance)
		self.stream.stop_stream()
		buffer_bytes = np.frombuffer(buffer.getvalue(), dtype=np.int16)
		return buffer_bytes


	def CloseBuffer(self):
		self.stream.close()
		self.audio.terminate()

from scipy.io import wavfile

class Resampler:

	def Resample(self,input):
		start_time = time.time()
		rate = 48000
		sampling_freq = 16000
		ratio = rate / sampling_freq
		audio = signal.resample_poly(input,1,ratio)
		return audio

class STFT:
	def CalculateSTFT(self,input_file,frame_length,stride):
		tf_audio = tf.dtypes.cast(input_file, tf.float32)
		rate = 16000.0
		frame_len = int(rate * frame_length)
		frame_step = int(rate * stride)
		stft = tf.signal.stft(tf_audio,
					frame_length=frame_len,
					frame_step=frame_step,
					fft_length=frame_len)
		spectrogram_buffer = tf.abs(stft)

		return spectrogram_buffer


class MFCC:
	def __init__(self,mel_bins,coefficients,sampling_rate,low_freq,up_freq):
		self.num_mel_bins = mel_bins
		self.coefficients = coefficients
		self.sampling_rate = sampling_rate
		self.lower_frequency = low_freq
		self.upper_frequency = up_freq
		self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
									self.num_mel_bins,
									321,
									self.sampling_rate,
									self.lower_frequency,
									self.upper_frequency)

	def CalculateMFCC(self,input_file,output_file):
		spectrogram = tf.cast(input_file, tf.float32)
		num_spectrogram_bins = spectrogram.shape[-1]
		ltmwm = self.linear_to_mel_weight_matrix
		mel_spectrogram = tf.tensordot(spectrogram,ltmwm,1)
		mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(
					ltmwm.shape[-1:]))
		log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
		mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[:,:self.coefficients]
		#output_filee = open(output_file,'w')
		np.savetxt(output_file,mfccs.numpy())


print(args)
#Input parameter
num_samples = args.num_samples
#Create output folder if it does not exist
output_folder = args.output + '/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
#Reset monitor to have the actual number of clock cycles
Popen(reset_monitor)
#Instance objects needed for the pipeline
resampler = Resampler()
stft = STFT()
mfcc = MFCC(40,10,16000,20,4000)
mic = Mic(1,48000,16)

for i in range(num_samples):
	start_time = time.time()
	###	RECORDING	###
	audio = mic.Record()
	###	RESAMPLING	###
	resampled_audio = resampler.Resample(audio)
	###	STFT	###
	stft_audio = stft.CalculateSTFT(resampled_audio,0.04,0.02)
	###	MFCC	###
	mfcc.CalculateMFCC(stft_audio, f'{output_folder}mfccs{i}.bin')

	end_time = time.time()
	print(f'{(end_time-start_time):.3f}')
#Close the mic buffer after all recordings are ended
mic.CloseBuffer()
#Print monitor status
Popen(print_monitor)
