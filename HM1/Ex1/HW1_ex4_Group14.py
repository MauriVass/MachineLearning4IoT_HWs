import os
import time
import argparse
import datetime
import tensorflow as tf

parser = argparse.ArgumentParser()
# Get input directory via '--input' argument
parser.add_argument('--input', type=str)
# Get output file via '--output' argument, the file must be .csv
parser.add_argument('--output', type=str)

args = parser.parse_args()

# Get inputs from args
input_path = args.input + '/'
output_file = args.output

# Read all files in the input dir
input_files = os.listdir(input_path)
# Get .csv file
csv_file = [f for f in input_files if f.find('.csv') > 0][0]
# Open .csv file
file = open(input_path + csv_file, 'r')

with tf.io.TFRecordWriter(output_file) as writer:
	for f in file:
		content = f.rstrip().split(',')
		# Read a line. Format: date(dd/mm/yyyy),time(hh:mm:ss),temp,humi,wav_file
		entry_date = [int(x) for x in content[0].split('/')]
		entry_time = [int(x) for x in content[1].split(':')]
		timestamp = datetime.datetime(entry_date[2],entry_date[1],entry_date[0],entry_time[0],entry_time[1],entry_time[2])
		timestamp_posix = int(time.mktime(timestamp.timetuple()))
		temperature = float(content[2])
		humidity = float(content[3])
		audio_file = content[4]

		# Read .wav file
		audio = tf.io.read_file(input_path + audio_file)
		audio = audio.numpy()

		date_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[timestamp_posix]))
		temp_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[temperature]))
		humi_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[humidity]))
		audio_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio]))

		mapping = {'datetime': date_feature,
		       'temperature': temp_feature,
		       'humidity': humi_feature,
		       'audio': audio_feature}
		example = tf.train.Example(features=tf.train.Features(feature=mapping))

		writer.write(example.SerializeToString())


