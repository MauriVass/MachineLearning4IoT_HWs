#Only colab
# !pip install paho-mqtt

import time
import sys
import json
import datetime
import tensorflow as tf
import numpy as np
import base64

#Only colab
# import os
# if(os.path.exists('Temp')==False):
#   !git clone https://github.com/MauriVass/Temp.git

from DoSomething import DoSomething

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='Model version: 1.tflite, 2.tflite, 3.tflite')
args = parser.parse_args()

class Receiver(DoSomething):
	def __init__(self,clientID, model):
		super().__init__(clientID)
		self.model = model

	def notify(self, topic, msg):
		# print(topic, msg)
		r = msg.decode('utf-8')
		r = json.loads(r)
		if('e' not in r):
			print("ANSWER PROBLEMS!! REQUIRED AN EVENT. RECEIVED:", r, ' CLOSING APPLICATION!')
			exit()
		events = r['e']
		if('vd' not in events[0]):
			print("ANSWER PROBLEMS!! REQUIRED AN AUDIO FILE. RECEIVED:", events[0], ' CLOSING APPLICATION!')
			exit()
		audio = events[0]['vd']

		audio_bytes = audio.encode()
		audio_bytes = base64.b64decode(audio_bytes)
		data = tf.io.decode_raw(audio_bytes,tf.float32)
		dims =  [49,10,1]
		data = tf.reshape(data,dims)
		data = tf.expand_dims(data,0)

		prediction = self.model.Evaluate(data)
		prediction = tf.nn.softmax(prediction)
		prediction = np.argmax(prediction)

		if('id' not in events[0]):
			print("Message PROBLEMS!! REQUIRED AN ID. RECEIVED:", events[0], ' CLOSING APPLICATION!')
			exit()
		id = str(events[0]['id'])
		body = { 'id': id, 'prediction':str(prediction)  }
		#print(body)
		body = json.dumps(body)
		self.myMqttClient.myPublish(idtopic+self.clientID+"/prediction/" ,body ,False)

class Model:
	def __init__(self, model_path):
		self.model_path = model_path

		if(model_path.find('zlib')>0):
			raise KeyError('YOU CAN\'T TEST A .zlib MODEL. (Use zipping=False in Optimize() method)')
		self.interpreter = tf.lite.Interpreter(model_path=model_path)
		self.interpreter.allocate_tensors()

		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()

	def Evaluate(self,input_data):
		self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
		self.interpreter.invoke()
		output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
		return output

if __name__ == "__main__":
	model_version = args.version
	model = Model(model_version)

	#Add the number of the inference client here: 1,2,3
	name_inf_client = f"InferClient{model_version[0]}" 
	coop_client = Receiver(name_inf_client, model)
	coop_client.run()
	idtopic = '/Group14_ML4IoT/'
	coop_client.myMqttClient.mySubscribe(idtopic+'audio/')

	while (True): 
		time.sleep(1)

	coop_client.end()

