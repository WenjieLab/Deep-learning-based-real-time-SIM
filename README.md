# VDL-SIM
The code is developed for real-time deap learning based SIM reconstruction, VDL-SIM, and is related to the paper A0-23-10003. 

## Contents
- [Environment](#Environment)
- [Requirement](#Requirement)
- [File structure](#Filestructure)
- [Running guide](#Runningguide)

## Environment
GPU: NVIDIA GeForce RTX 3050Ti  
Tensorflow-gpu 2.10.0  
Keras 2.10.0  
CUDA 11.6  
Python 3.9.5  

## Requirement
graphviz==0.20.1  
h5py==3.7.0  
imageio==2.22.4  
keras==2.10.0  
matplotlib==3.6.2  
numpy==1.23.5  
onnx==1.13.0  
onnx-tf==1.10.0  
opencv-python==4.6.0.66  
pandas==1.5.3  
Pillow==9.4.0  
pyimagej==1.4.1  
QtPy==2.3.0  
scikit-image==0.19.3  
scipy==1.10.1  
tensorboard==2.10.1  
tensorboardX==2.5.1  
tensorflow-estimator==2.10.0  
tensorflow-gpu==2.10.0  
tf-slim==1.1.0  
torch==1.13.1+cu116  
torchaudio==0.13.1+cu116  
torchvision==0.14.1+cu116  
zipp==3.10.0  

## File structure
- ```./models``` includes declaration of VDL-SIM model  
- ```./test``` includes some different SNR demo images of microtubules to test VDL-SIM model  
- ```./utils``` is the tool package of VDL-SIM  
- ```./weight``` place pre-trained VDL-SIM model here for testing
- ```./models``` includes C++ interface for code

## Running guide
- Download pre-trained models of VDL-SIM and place them in ```./weight/```   
- Download test data and place them in ```./test/images/```. Also, you can prepare other testing data
- Open your terminal and run ```predict.py```
- The output SR images will be saved in ```./test/images/output_resu-SIM_weight-SIM/```
