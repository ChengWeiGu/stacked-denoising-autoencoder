# Stacked-Denoising-Convolutional-Autoencoder (SDCAE-Pytorch)
The SDCAE model is implemented for PHM data. The scripst are public and based on Pytorch.

## Model and Environment:  
1. The SDCAE model is revised from the previous wrok:  
https://github.com/ShayanPersonal/stacked-autoencoder-pytorch  

2. One can install some related packages by referring to requirements.txt  
-> Python Version: 3.6  

## Run the model:  

1. command: python train_SDAE_PHM.py -t  
    => For PHM data, there are three classes: Normal, NG1 and NG2.  


2. command: python train_SDAE_PHM.py -v -w .\dir_name\Model_Weight_Name.ph  
    => One can evaluate the mode by the cmd, where weights of model should exist.  


## DataSet:   
TBD
