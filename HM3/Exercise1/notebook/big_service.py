#!/usr/bin/env python
# coding: utf-8

# In[1]:

#only Colab
# import sys
# get_ipython().system('conda install --yes --prefix {sys.prefix} cherrypy')


# In[1]:


import tensorflow as tf
import tensorflow.lite as tflite
import cherrypy
import json
import base64
from cherrypy.process.wspbus import ChannelFailures
import numpy as np
import tensorflow as tf
import sys


class SignalPreprocessor:
    def __init__(self, sampling_rate, frame_length, frame_step, num_mel_bins=None, lower_frequency=None,
                 upper_frequency=None, num_coefficients=None, mfcc=False, image_size=32):
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        self.mfccs = mfcc
        self.image_size = image_size

        if (mfcc):
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

    def pad(self, audio):
        zero_padding = tf.zeros(self.sampling_rate - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])
        return audio

    def get_spectrogram(self, audio):
        # Calculate the STFT of the signal given frame_length and frame_step
        stft = tf.signal.stft(audio,
                              frame_length=self.frame_length,
                              frame_step=self.frame_step,
                              fft_length=self.frame_length)
        # Transform the complex number in real number
        spectrogram = tf.abs(stft)
        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                                       self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[:, :self.num_coefficients]
        return mfccs

    def preprocess_with_stft(self, audio):
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [self.image_size, self.image_size])
        return spectrogram

    def preprocess_with_mfcc(self, audio):
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)
        return mfccs

    def load_audio_from_base_64(self, audio):
        audio = base64.b64decode(audio)
        return audio

    def audio_preprocessing(self, audio):
        try:
            audio, _ = tf.audio.decode_wav(audio)
            audio = tf.squeeze(audio, axis=1)
        except Exception as e:
            print(e)
        processed_audio = self.preprocess(audio)
        return processed_audio


class Model:
    def __init__(self, model_path):
        self.model_path = model_path

        if (model_path.find('zip') > 0):
            raise KeyError('YOU CAN\'T TEST A .zip MODEL. (Use zipping=False in Optimize() method)')
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def Evaluate(self, data):
        self.interpreter.set_tensor(self.input_details[0]['index'], data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        return output


class Server(object):
    exposed = True
    
    def __init__(self):
        # It can be: True, False
        mfcc = True
        # It can be (0,1]
        alpha = 1
        # Sparcity increases latency(may be a problem for KS) due to cache misses
        # it can be (0.3,1) or None(if you don't to use sparsity)
        sparsity = None

        # Here you can change:
        # STFT(mfcc=False): frame_length, frame_step
        # MFCC(mfcc=True): frame_length, frame_step, num_mel_bins, num_coefficients
        frame_length = 640  # Default 640 (mfcc=True), 256(mfcc=False)
        frame_step = 320  # Default 320 (mfcc=True), 128(mfcc=False)
        num_mel_bins = 40  # Default 40 (only mfcc=True)
        num_coefficients = 10  # Default 10 (only mfcc=True)
        image_size = 32  # Default 32 (only mfcc=False)

        if (mfcc):
            self.sp = SignalPreprocessor(sampling_rate=16000, frame_length=int(frame_length),
                                       frame_step=int(frame_step),
                                       num_mel_bins=int(num_mel_bins), lower_frequency=20, upper_frequency=4000,
                                       num_coefficients=int(num_coefficients), mfcc=mfcc)
        else:
            self.sp = SignalPreprocessor(sampling_rate=16000, frame_length=frame_length, frame_step=frame_step,
                                       image_size=image_size)


    def request_checker(seld, input):
        input = json.loads(input)
        ip = None
        timestamp = None
        audio = None

        bn = input["bn"]

        if input['bn'] is None:
            raise cherrypy.HTTPError(400, "Client IP is missing")
        else:
            ip = input['bn']

        if input['bi'] is None:
            raise cherrypy.HTTPError(400, "timestamp is missing")
        else:
            timestamp = input['bi']

        if input['e'] is None:
            raise cherrypy.HTTPError(400, "audio is missing")
        else:
            e = input['e'][0]
            audioBase64 = e['vd']
            if audioBase64 is None:
                raise cherrypy.HTTPError(400, "audio base 64 format is missing")
            else:
                audio = base64.b64decode(audioBase64)

        return audio

    def PUT(self, *path, **query):
        input = cherrypy.request.body.read()
        audio = self.request_checker(input)
        
        processed_audio = self.sp.audio_preprocessing(audio)
        model_path = 'big.tflite'
        model = Model(model_path) 
        data = tf.expand_dims(processed_audio, axis=0)
        y_pred = model.Evaluate(data)
        y_pred = tf.nn.softmax(y_pred).numpy()
        y_pred_best = np.argmax(y_pred)
        #print('y_pred ', y_pred_best, y_pred[y_pred_best])
        
        #Adding the probability adds a little overhead (one string) but it may be usefull to compiute some statistics
        body = { 'label': str(y_pred_best), 'probability':f'{(y_pred[y_pred_best]):.4f}' }
        return json.dumps(body)


# In[ ]:


if __name__ == '__main__':
    conf = {
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher()
        }
    }
    cherrypy.tree.mount(Server(), '/', conf)
    
    ip_server_machine = '192.168.1.7'
    cherrypy.config.update({'server.socket_host': ip_server_machine})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()


# In[ ]:




