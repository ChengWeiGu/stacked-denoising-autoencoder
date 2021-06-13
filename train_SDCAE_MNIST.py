import os
import time

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torchvision.datasets import MNIST, CIFAR10
from torchvision.utils import save_image

from model import StackedAutoEncoder_MNIST

import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import pickle
import argparse
from scipy import fft

import matplotlib.pyplot as plt



class MNIST_dataset(Dataset):
    def __init__(self, df, rows=42000):
        self.imgnp = df.iloc[:rows, 1:].values
        self.labels = df.iloc[:rows, 0].values
        self.rows = rows
    
    def __len__(self):
        return self.rows
    
    def __getitem__(self, idx):
        image = self.imgnp[idx].reshape((28,28))
        image = np.pad(array = image, pad_width=((2,2),(2,2)), mode='constant', constant_values=0)
        image = torch.tensor(image, dtype=torch.float) / 255  # Normalize
        image = image.view(1, 32, 32)  # (channel, height, width)
        label = self.labels[idx]
        return (image, label)
        
        

if not os.path.exists('./imgs'):
    os.mkdir('./imgs')

def to_img(x):
    x = x.view(x.size(0), 1, 32, 32)
    return x

num_epochs = 200
batch_size = 2
num_class = 10
display_freq = 20
random_freq = 220


train_df = pd.read_csv('train.csv')
train_dataset = MNIST_dataset(train_df, 600)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


test_df = pd.read_csv('test.csv')
test_dataset = MNIST_dataset(train_df, 100)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = StackedAutoEncoder_MNIST().cuda()
# print(model)

print("len of train_loader = %d" %len(train_loader))
print("len of test_loader = %d" %len(test_loader))
print("len of train data = %d" %len(train_dataset))
print("len of test data = %d" %len(test_dataset))



for epoch in range(num_epochs):
    if epoch % random_freq == 0:
        # Test the quality of our features with a randomly initialzed linear classifier.
        classifier = nn.Linear(512 * 16, num_class).cuda() #MNIST
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0001)

    model.train()
    total_time = time.time()
    correct = 0
    for i, data in enumerate(train_loader):
        img, target = data
        # plt.imshow(img.numpy()[0,1,:,:].reshape((32,32)))
        # plt.show()
        target = Variable(target).cuda()
        img = Variable(img).cuda()
        features = model(img).detach()
        prediction = classifier(features.view(features.size(0), -1))
        loss = criterion(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = prediction.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    total_time = time.time() - total_time

    model.eval()
    img, _ = data
    img = Variable(img).cuda()
    features, x_reconstructed = model(img)
    reconstruction_loss = torch.mean((x_reconstructed.data - img.data)**2)
    # plt.imshow(x_reconstructed.cpu().data.numpy()[0,1,:,:].reshape((32,32)))
    # plt.show()
    
    if epoch % display_freq == 0:
        # print(img.cpu().data.shape)
        print("Saving epoch {}".format(epoch))
        orig = to_img(img.cpu().data)
        save_image(orig, './imgs/orig_{}.png'.format(epoch))
        pic = to_img(x_reconstructed.cpu().data)
        save_image(pic, './imgs/reconstruction_{}.png'.format(epoch))

    
    print("Epoch {} complete\tTime: {:.4f}s\t\tLoss: {:.4f}".format(epoch, total_time, reconstruction_loss))
    print("Feature Statistics\tMean: {:.4f}\t\tMax: {:.4f}\t\tSparsity: {:.4f}%".format(
        torch.mean(features.data), torch.max(features.data), torch.sum(features.data == 0.0)*100 / features.data.numel())
    )
    print("Linear classifier performance: {}/{} = {:.2f}%".format(correct, len(train_loader)*batch_size, 100*float(correct) / (len(train_loader)*batch_size)))
    
    
    
    # val_loss=0
    accuracy=0
    sub_total = 0
    with torch.no_grad():
        for j, data in enumerate(test_loader,1):
            images, labels = data

            if(torch.cuda.is_available()):
                images = images.cuda()
                labels = labels.cuda()

            features, x_reconstructed = model(images)
            outputs = classifier(features.view(features.size(0), -1))
            _, predicted = torch.max(outputs.data, 1)
            sub_total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
            # val_loss += criterion(outputs, labels).item()
    
    print("Summary at Epoch: {}/{}..".format(epoch,num_epochs),
          "Val_Accu: {:.3f}%".format((accuracy/sub_total)*100))
            
    print("="*80)
    
torch.save(model.state_dict(), './MNIST_CDAE.pth')
