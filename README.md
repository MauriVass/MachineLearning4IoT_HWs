# MachineLearning4IoT_HWs

## Introduction
Laboraties developed during the Machine Learning for IoT course at PoliTo.

## Description
The course aims to introduce the problems related to the implementation of machine learning applications and algorithms on platforms other than high-performance servers available in the cloud. The contents and the skills acquired at the end of the course include both the hardware aspects of the problem (architectures of "edge" devices) and the software aspects (programming models, protocols and related APIs). These skills will allow a correct understanding of decentralized systems in which the flow of data is processed not only on servers, but rather locally on devices with reduced computational resources and energy.

## Topics homeworks

1.  1. TFRecord Dataset for **Sensor Fusion**: marge data coming from different sensors (temperature, humidity, microphone, ...) into binary records. In this way, the data will reside in consecutive locations of memory, thus
accelerating parallel fetching operations.
    2. Low-power Data Collection and Pre-processing using **D**ynamic **V**oltage **F**requency **S**caling (**DVFS**): change CPU frequency, in order to reduce the hardware energy consumption, when the task is not too complex.

2.  1. **Multi-Step Temperature and Humidity Forecasting**: usage of a deep neural network to predict the future values of temperature and humidity. The networks have to satisfy some constrains: MAE and model size.
    2. **Keyword Spotting**: usage of a deep neural network to detect the word in an audio signal. The networks have to satisfy some constrains: accuracy, model size, inference latency and total latency.
    
3.  1. **Big/Little Inference**: usage of a Big/Little system for keyword spotting. This system enables to reduce the energy consumption: 
          * when the task is easy run the Little model (on the edge);
          * when the Little model is not enough use the Big model (on the cloud). 
        <a/>
       Constrains on: overall accuracy (Big/Little system), Little network memory size, Little network inference time, communication cost (data sent to the cloud). </br>
       Used RESTful API for communication.
    2. **Cooperative Inference**: run N different networks to increase the overall accuracy for keyword spotting. </br>
        Constrains on: overall accuracy. Used MQTT standard for communication.
