import tensorflow as tf
import os
import time
import argparse
import pyaudio

parser = argparse.ArgumentParser()
# Get input directory via '--input' argument
parser.add_argument('--num-samples', type=str)
# Get output file via '--output' argument, the file must be .tfrecord
parser.add_argument('--output', type=str)

args = parser.parse_args()


num_sample = args.num_samples
output = args.output

# for ns in num_sample:
