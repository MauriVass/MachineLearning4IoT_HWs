import tensorflow as tf
import wave
import pyaudio
import datetime

def parser_function(example):
	return tf.io.parse_single_example(example, mapping)

input_file = 'test.tfrecord'
raw_data = tf.data.TFRecordDataset(input_file)

mapping = {'datetime': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'temperature':  tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'humidity':  tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'audio':  tf.io.FixedLenFeature([], tf.string, default_value='')}

parsed_dataset = raw_data.map(parser_function)
i = 0
for p in parsed_dataset:
	date = datetime.datetime.fromtimestamp(p['datetime'].numpy())
	temp = p['temperature'].numpy()
	hum = p['humidity'].numpy()
	audio = p['audio'].numpy()
	#print(audio)

	waveFile = wave.open(f'audio{i}.wav','wb')
	i+=1
	waveFile.setnchannels(1)
	a = pyaudio.PyAudio()
	waveFile.setsampwidth(a.get_sample_size(pyaudio.paInt16))
	waveFile.setframerate(48000)
	waveFile.writeframes(audio)
	waveFile.close()

	entry = f'{date},{temp},{hum},{type(audio)}'
	print(entry)

