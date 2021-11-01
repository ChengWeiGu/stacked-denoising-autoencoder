# Stacked-Denoising-Convolutional-Autoencoder (SDCAE-Pytorch)
The SDCAE model is implemented for PHM data. The scripst are public and based on Pytorch.

## Model and Environment:  
1. The SDCAE model and script are revised from the following previous wrok:  
https://github.com/ShayanPersonal/stacked-autoencoder-pytorch  

2. One can install some related packages by referring to requirements.txt  
-> Python Version: 3.6  

## Run the model:  

1. command: python train_SDCAE_PHM.py -t  
    => For PHM data, there are three classes: Normal, NG1 and NG2.  
    => For MNIST data, by the cmd please: python train_SDCAE_MNIST.py -t  


2. command: python train_SDCAE_PHM.py -v -w .\dir_name\Model_Weight_Name.ph  
    => One can evaluate the model by the cmd, where weights of model should exist.  


## DataSet Description:   
1. There are three classes of data: Normal, NG1 and NG2.  
2. The data were measured from motor oscillation by a three-axis accelerator.  
3. The x, y and z data were divided and saved under different folders.  
4. Each data.csv contains about 60k ea points where measured time is 30 sec.  
  

![image](https://github.com/ChengWeiGu/stacked-denoising-autoencoder/blob/main/result.jpg)  
