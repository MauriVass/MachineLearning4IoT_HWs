import tensorflow as tf
import os
import time
import argparse


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
# This variable is used to 'test' the performance of the tfrecord
# Set the initial size equal to the .csv file
initial_size = os.path.getsize(input_path + csv_file)

with tf.io.TFRecordWriter(output_file) as writer:
    for f in file:
        content = f.rstrip().split(',')
        # Read a line. Format: date,time,temp,humi,wav_file
        datetime = time.time()
        temperature = int(content[2])
        humidity = int(content[3])
        audio_file = content[4]
        # Read .wav file
        audio = tf.io.read_file(input_path + audio_file)
        audio = audio.numpy()
        # Add the .wav size
        initial_size += os.path.getsize(input_path + audio_file)

        # print(int(datetime),type(datetime),type(temperature),type(humidity),type(audio))

        date_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(datetime)]))
        temp_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[temperature]))
        humi_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[humidity]))
        audio_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio]))

        mapping = {'datetime': date_feature,
                   'temperature': temp_feature,
                   'humidity': humi_feature,
                   'audio': audio_feature}
        example = tf.train.Example(features=tf.train.Features(feature=mapping))

        writer.write(example.SerializeToString())

file_size = os.path.getsize(output_file)
print(f'Initial size {initial_size}, final size {file_size} Ratio {(file_size / initial_size):.4f}')
