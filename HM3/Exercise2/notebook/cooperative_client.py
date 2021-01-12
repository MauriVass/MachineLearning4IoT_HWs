import time
import sys
import json
import pyaudio
import datetime
import tensorflow as tf
import numpy as np
import base64

# sys.path.insert(0, './../../')
from DoSomething import DoSomething

class Receiver(DoSomething):
  def __init__(self,clientID, model):
    super().__init__(clientID)
    self.model = model

  def notify(self, topic, msg):
    # print(topic, msg)
    r = msg.decode('utf-8')
    r = json.loads(r)
    events = r['e']
    audio = events[0]['vd']

    audio_bytes = audio.encode()
    audio_bytes = base64.b64decode(audio_bytes)
    data = tf.io.decode_raw(audio_bytes,tf.float32)
    dims = events[0]['dims']
    data = tf.reshape(data,dims) #or anything else
    data = tf.expand_dims(data,0)

    prediction = self.model.Evaluate(data)
    prediction = [np.argmax(prediction)]

    timestamp = int(datetime.datetime.now().timestamp())
    body = { 'timestamp': timestamp, 'prediction':0 , 'index':0  }
    print(body)
    body = json.dumps(body)
    #client_rpi.myMqttClient.
    self.myMqttClient.myPublish(idtopic+"prediction/" ,body ,False)

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
	model_path = 'KS_DSCNNTruespars0.9.tflite_W'
	model = Model(model_path)

	coop_client = Receiver("Coop Client", model)
	coop_client.run()
	idtopic = '/Group14_ML4IoT/'
	coop_client.myMqttClient.mySubscribe(idtopic+'audio/')

	a=0
	while (True): #Find a better way
		a+=1
		time.sleep(1)

	coop_client.end()
