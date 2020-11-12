from scipy import signal
import io
import time
from scipy.io.wavfile import read, write
import numpy as np
import pyaudio
import wave
import tensorflow as tf
import os  # To Remove
import sys  # To Remove


class Mic:
    def __init__(self, dur, rate, res):
        self.input_res = int(res)
        if (self.input_res == 8):
            self.resolution = pyaudio.paInt8
        elif (self.input_res == 16):
            self.resolution = pyaudio.paInt16
        elif (self.input_res == 32):
            self.resolution = pyaudio.paInt32
        self.channels = 1
        self.rate = rate
        self.chunk = 4800
        self.record_seconds = dur

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.resolution,
                                      channels=self.channels,
                                      rate=self.rate, input=True,
                                      frames_per_buffer=self.chunk)
        self.counter = 0

    def Record(self, save=False):
        # time_start = time.time()
        if (save):
            frames = []
        # audio = pyaudio.PyAudio()
        # stream = audio.open(format=self.resolution,
        #		channels=self.channels,
        #		rate= self.rate, input=True,
        #		frames_per_buffer=self.chunk)

        print('\n---	Recording Start	---')
        max_val = int(self.rate / self.chunk * self.record_seconds)
        l = 0
        buffer = io.BytesIO()
        # buffer.write(b'RIFF$w\x01\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\xbb\x00\x00\x00w\x01\x00\x02\x00\x10\x00data\x00w\x01\x00')
        self.stream.start_stream()
        # time_reg = time.time()
        for i in range(max_val):
            data = self.stream.read(self.chunk, exception_on_overflow=True)
            buffer.write(data)
            # print(len(buffer.getvalue()))
            # frames.append(int.from_bytes(data,'little'))
            if (save):
                frames.append(data)
                l += len(data)
        # buffer.write(bytearray(data))
        self.stream.stop_stream()
        # stream.close()
        # audio.terminate()
        print('End loop Rec: ', len(buffer.getvalue()))

        # Buffer requires to be cast as uint16 but I'M NOT SURE WHY
        # with dtype=float(default) the len(buffer_bytes) is 12k while it should be 48k (smaller size, faster compu)
        # with dtype=uint16 the len(buffer_bytes) is ok (48k)
        # Next calculation requires lot more time to compiute wrt float or uint8 (actually not sure if it only depends on the output size of the 2 methods)
        # ROOM FOR IMPROVEMENT
        buffer_bytes = np.frombuffer(buffer.getvalue(), dtype=np.uint16)
        # buffer_bytes = np.ndarray(len(buffer.getvalue())-1, buffer=buffer.getvalue()) #, dtype=np.uint16)
        print(f'Buffer len {len(buffer_bytes)}')

        # time_end = time.time()
        # elapsed_time = time.time() - time_start
        # file_audio = io.BytesIO() #'right.wav'
        # time_start = time.time()
        if (save):
            # waveFile = wave.open(file_audio,'wb')
            # waveFile.setnchannels(self.channels)
            # waveFile.setsampwidth(audio.get_sample_size(self.resolution))
            # waveFile.setframerate(self.rate)
            # waveFile.writeframes(b''.join(frames))
            # waveFile.close()

            waveFile = wave.open(f'output/mic_file_{self.counter}.wav', 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.resolution))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()
            self.counter += 1
            print(f'File len {l}')
        # print(file_audio.getvalue()[:60], type(file_audio), type(file_audio.getvalue()))
        # time_end = time.time()
        # print(f'From Mic: {type(file_audio)}, {file_audio.getvalue()[:50]}')
        # print(f'---	Recording End	--- Elapsed time {elapsed_time:.3f}', '\n')
        # return waveFile
        # return file_audio
        #		return result
        return buffer_bytes

    # storage_time = time_end - time_start
    # print(f'Time to storage {storage_time:.3f}')
    def CloseBuffer(self):
        self.stream.close()
        self.audio.terminate()


from scipy.io import wavfile


class Resampler:
    def __init__(self):
        self.counter = 0

    def Resample(self, input, save=False):
        print(f'---	Resampling Start	---')
        # print(f'Before From Resamp {type(input)} {input.getvalue()[:50]}')
        # print('Res before ',input.getbuffer().nbytes)
        #		print('Res before ',len(input))
        # start_time = time.time()
        # print('Red after, ', len(audio),rate)
        # print(f'After From Resamp {type(audio)} {audio[:50]}')
        rate = 48000
        # print(type(input))
        audio = input  # .astype('float64')
        # print(type(audio))
        sampling_freq = 16000
        ratio = rate / sampling_freq

        # print(audio,type(audio))
        print('before res ', len(audio))
        audio = signal.resample_poly(audio, 1, ratio)
        print('after res ', len(audio), type(audio))
        #		print(type(audio))
        # end_time = time.time()
        # exec_time = end_time - start_time

        if (save):
            rate_file, audio_file = wavfile.read(f'output/mic_file_{self.counter}.wav')
            ratio = rate_file / sampling_freq
            print('File: before res ', len(audio_file))
            audio_file = signal.resample_poly(audio_file, 1, ratio)
            print('File: after res ', len(audio_file))
            audio_file = audio_file.astype(np.int16)
            wavfile.write(f'output/resampl_file_{self.counter}.wav', sampling_freq, audio_file)
            self.counter += 1

        # print(f'--- Resampling End	--- Elapsed time {exec_time:.3f}')
        return audio


class STFT:
    def __init__(self):
        self.counter = 0

    def CalculateSTFT(self, input_file, frame_length, stride, save=False):
        print('\n---	STFT Start	---')

        # Array of bytes
        if (save):
            audio_file = tf.io.read_file(f'output/resampl_file_{self.counter}.wav')

        # start_time = time.time()
        # Signal and frequency of the audio input
        #		tf_audio, rate = tf.audio.decode_wav(audio)
        tf_audio = input_file
        print(f'Buffer len {len(tf_audio)}')
        rate = 16000.0
        # Add a dimension to specify the number of channels

        # Hyperparameters. (Maybe) Good values: 0.04, 0.02
        frame = frame_length
        stride = stride
        frame_len = int(rate * frame)
        frame_step = int(rate * stride)
        # print(f'Frame len: {frame_len}, frame step: {frame_step}')

        # print(type(tf_audio))

        # Calculate the STFT of the signal given frame_length and frame_step
        # ts = time.time()
        stft = tf.signal.stft(tf_audio,
                              frame_length=frame_len,
                              frame_step=frame_step,
                              fft_length=frame_len)
        # te = time.time()
        # print(f'tf.signal.stft time: {(te - ts):.3f}')

        # Transform the complex number in real number
        # ts = time.time()
        # spectrogram_buffer = tf.abs(stft)
        # te = time.time()
        # print(f'tf.abs time: {(te - ts):.3f}')
        # print(f'Buffer spect {(tf_audio).shape}')
        # end_time = time.time()

        if (save):
            audio, rate = tf.audio.decode_wav(audio_file)
            audio = tf.squeeze(audio, 1)
            print(f'file len {(tf_audio).shape}')
            stft = tf.signal.stft(audio,
                                  frame_length=frame_len,
                                  frame_step=frame_step,
                                  fft_length=frame_len)
            # Transform the complex number in real number
            spectrogram = tf.abs(stft)
            print(f'file spect {(tf_audio).shape}')
            byte_string = tf.io.serialize_tensor(spectrogram)
            # Save the spectrogram and the .png image on disk
            output_file = f'output/file_spect_{self.counter}.spect'
            self.counter += 1
            tf.io.write_file(output_file, byte_string)

        elapsed_time = end_time - start_time
        print(f'---	STFT End	--- Required time: {elapsed_time:.3f}')

        return spectrogram_buffer


class MFCC:
    def __init__(self, mel_bins, coefficients, sampling_rate, low_freq, up_freq):
        self.num_mel_bins = mel_bins
        self.coefficients = coefficients
        self.sampling_rate = sampling_rate
        self.lower_frequency = low_freq
        self.upper_frequency = up_freq
        self.counter = 0

    def CalculateMFCC(self, input_file, output_file, save=False):
        # Read file
        print('\n---	MFCC Start	----')
        # in_file = 'spect'

        # spectrogram = input_file.numpy() #.astype('float32')
        # start_time = time.time()
        spectrogram = input_file
        print(
            f'Buffer Spectrogram shape: {(spectrogram).shape} (This should be (49,321)), {type(spectrogram)}, {(spectrogram.numpy()).shape}, {type(spectrogram.numpy())}')
        spectrogram = tf.cast(spectrogram, tf.float32)
        # te = time.time()
        # print(f'Casting(+ import time~0) time: {(te - start_time):.3f}')
        print(
            f'Buffer Spectrogram shape: {(spectrogram).shape}, {type(spectrogram)}, {(spectrogram.numpy()).shape}, {type(spectrogram.numpy())}')
        # spectrogram = tf.constant(input_file.numpy(), dtype=tf.float64)
        # spectrogram = tf.io.parse_tensor(input_file.numpy(), out_type=tf.float32)
        # print(f'Spectrogram shape: {(spectrogram).shape}, {type(spectrogram)}')

        num_spectrogram_bins = spectrogram.shape[-1]
        # print(num_spectrogram_bins)
        # ts = time.time()
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            self.num_mel_bins,
            num_spectrogram_bins,
            self.sampling_rate,
            self.lower_frequency,
            self.upper_frequency)
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        # te = time.time()
        # print(f'Calculation time: {(te - ts):.3f}')

        # Not all coefficients are important, so it is useful to select only some of them.
        # To find them you can use a search algorithm
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[:, :self.coefficients]
        print('Buffer MFCCS shape: ', mfccs.shape)

        # print(f'Saved {output_file}.mfccs')
        file = open(output_file + '.mfccs', 'w')
        print(mfccs.numpy(), file=file)
        file.close()
        # file_inp_size = os.path.getsize(input_file)
        # print(f'File size {file_inp_size}')
        # file_out_size = os.path.getsize(output_file+'.mfccs')
        # print(f'File size {file_out_size}')

        # end_time = time.time()

        if (save):
            spectrogram_file = tf.io.read_file(f'output/file_spect_{self.counter}.spect')
            # print(f'File Spectrogram shape: {(spectrogram).shape}, {type(spectrogram)}, {len(spectrogram.numpy())}, {type(spectrogram.numpy())}, {spectrogram.numpy()[:20]}')
            spectrogram_file = tf.io.parse_tensor(spectrogram_file, out_type=tf.float32)
            # print((spectrogram).shape)
            # print(type(spectrogram))
            # print((spectrogram.numpy()).shape)
            print(
                f'File Spectrogram shape: {(spectrogram_file).shape}, {type(spectrogram_file)}, {type(spectrogram_file.numpy())}')

            mel_spectrogram = tf.tensordot(spectrogram_file, linear_to_mel_weight_matrix, 1)
            mel_spectrogram.set_shape(spectrogram_file.shape[:-1].concatenate(
                linear_to_mel_weight_matrix.shape[-1:]))
            log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

            # Not all coefficients are important, so it is useful to select only some of them.
            # To find them you can use a search algorithm
            mfccs_file = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[:, :self.coefficients]
            print('File MFCCS shape: ', mfccs_file.shape)

            file = open(f'file_mfccs_{self.counter}.mfccs', 'w')
            print(mfccs_file.numpy(), file=file)
            file.close()

            # print(f'Saved {output_file}.mfccs')
            file_inp_size = os.path.getsize(f'file_mfccs_{self.counter}.mfccs')
            print(f'File size {file_inp_size}')
            file_out_size = os.path.getsize(output_file + '.mfccs')
            print(f'Buffer size {file_out_size}')

            # File
            image = tf.transpose(mfccs_file)
            # Add the 'channel' dimension
            image = tf.expand_dims(image, -1)
            # Normalize to have values on a range [0,255]
            min_val = tf.reduce_min(image)
            max_val = tf.reduce_max(image)
            image = (image - min_val) / (max_val - min_val)
            image = image * 255
            image = tf.cast(image, tf.uint8)

            png_image = tf.io.encode_png(image)
            tf.io.write_file(f'output/file_mfccs_{self.counter}.png', png_image)

            # Buffer
            image = tf.transpose(mfccs)
            # Add the 'channel' dimension
            image = tf.expand_dims(image, -1)
            # Normalize to have values on a range [0,255]
            min_val = tf.reduce_min(image)
            max_val = tf.reduce_max(image)
            image = (image - min_val) / (max_val - min_val)
            image = image * 255
            image = tf.cast(image, tf.uint8)

            png_image = tf.io.encode_png(image)
            tf.io.write_file(f'output/buffer_mfccs_{self.counter}.png', png_image)

        print(f'--- MFCC End	--- Execution time: {(end_time - start_time):.3f}')


# Input parameter
num_samples = 1
output_folder = 'output/'

# duration, rate, res
mic = Mic(1, 48000, 16)
resampler = Resampler()
stft = STFT()
mfcc = MFCC(40, 10, 16000, 20, 4000)

save = True if sys.argv[1] == '0' else False
# a = 44
times = []
for i in range(num_samples):
    start_time = time.time()

    ###	RECORDING	###
    audio = mic.Record(save)

    # buffer = io.BytesIO()
    # for _ in range(int(48000/4800*1)):
    #	buffer.write(stream.read(4800))
    # with open(audio.getvalue(),'rb') as w:
    #	audio_mic = w.read()
    # audio = io.BytesIO(audio)
    #	print(f'Reco: {type(audio)}, {audio[:a]}')

    # with open('right.wav','rb') as w:
    #	audioo = w.read()
    # print(f'Read: {type(audioo)} {audioo[:a]}')

    # Store the registration in a IOBuffer
    # f = io.BytesIO(b'RIFF$w\x01\x00WAVEfmt '+audio) #.getvalue() b'RIFF$w\x01\x00WAVEfmt '+
    # print('Recorded Content ',f.getvalue()[:50])
    # print(f'Buffer recorded type: {type(f)}'

    ###	RESAMPLING	###
    resampled_audio = resampler.Resample(audio, save)

    ###	STFT	###
    stft_audio = stft.CalculateSTFT(resampled_audio, 0.04, 0.02, save)

    ###	MFCC	###
    mfcc.CalculateMFCC(stft_audio, f'{output_folder}recording_{i}', save)

    end_time = time.time()
    print(f'Elapsed time {(end_time - start_time):.3f}')
    times.append(f'{(end_time - start_time):.3f}')
    print('\n\n')
# wavefile = wave.open('prova_res.wav','rb')
# print(wavefile.getnframes())

mic.CloseBuffer()

# Print times for eache recording
for i in times:
    print(i, 'Shame of you!' if float(i) > 1.08 else 'Great u awesome!!')
