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

from model import StackedAutoEncoder, EnsembleModel

import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import pickle
import argparse
from scipy import fft

import matplotlib.pyplot as plt



def pytorch_rolling_window(x, window_size, step_size=1):
    # unfold dimension to make our rolling window
    return x.unfold(0,window_size,step_size)


def preprocess(mode ='train', rows = 60000):
    
    axes = ['x','y','z']
    pass_dir = r"D:\Side Work Data\PHM\Demo_box\demo_box_0518\wav\pass"
    ng1_dir = r"D:\Side Work Data\PHM\Demo_box\demo_box_0518\wav\ng1"
    ng2_dir = r"D:\Side Work Data\PHM\Demo_box\demo_box_0518\wav\ng2"
    all_data = np.array([])
    all_label_data = np.array([])
    window_size = 2048
    step_size = 512
    
    for ax in axes:
        
        pass_filename = ""
        ng1_filename = ""
        ng2_filename = ""
        
        # there are three files
        if mode == 'train':
            pass_filename = pass_dir + '/' + ax + '/' + '20210518112208.csv'
            ng1_filename = ng1_dir + '/' + ax + '/' + '20210518114530.csv'
            ng2_filename = ng2_dir + '/' + ax + '/' + '20210518134847.csv'
        elif mode == 'test1':
            pass_filename = pass_dir + '/' + ax + '/' + '20210518112338.csv'
            ng1_filename = ng1_dir + '/' + ax + '/' + '20210518114602.csv'
            ng2_filename = ng2_dir + '/' + ax + '/' + '20210518134922.csv'
            step_size = window_size #no augmentation for test
        elif mode == 'test2':
            pass_filename = pass_dir + '/' + ax + '/' + '20210518112416.csv'
            ng1_filename = ng1_dir + '/' + ax + '/' + '20210518114635.csv'
            ng2_filename = ng2_dir + '/' + ax + '/' + '20210518134957.csv'
            step_size = window_size #no augmentation for test
        
        
        _temp_data = np.array([])
        _temp_lable_data = np.array([])
        
        for filename, label in zip([pass_filename, ng1_filename, ng2_filename],[0,1,2]):
            print('processing for file {}'.format(filename))
            
            data = pd.read_csv(filename).values[:rows]
            data = torch.tensor(data, dtype=torch.float)
            data = pytorch_rolling_window(data, window_size = window_size, step_size = step_size)
            data = data.numpy().reshape((-1,window_size)) #important: you must reshape it!
            n_samples = len(data)
            label_data = np.array([label]*n_samples).reshape((-1,1))
            
            if label == 0:
                _temp_data = data
                _temp_lable_data = label_data
            else:
                _temp_data = np.vstack((_temp_data,data))
                _temp_lable_data = np.vstack((_temp_lable_data,label_data))
                
        
        if ax == 'x':
            all_data = _temp_data
            all_label_data = _temp_lable_data
        else:
            all_data = np.hstack((all_data, _temp_data))
        
        
    
    # print(all_data.shape)
    # print(all_label_data.shape)
    return np.hstack((all_data,all_label_data))
    

def merge_testdata(all_data1,all_data2):
    data1, label1 = all_data1[:,:-1], all_data1[:,-1].reshape((-1,1))
    data2, label2 = all_data2[:,:-1], all_data2[:,-1].reshape((-1,1))
    
    data, label = np.vstack((data1,data2)), np.vstack((label1,label2))
    
    return np.hstack((data,label))



class PHM_dataset(Dataset):
    def __init__(self, all_data):
        
        self.rows = len(all_data)
        self.imgnp = all_data[:self.rows, :-1]
        self.labels = all_data[:self.rows, -1]
    
    def __len__(self):
        return self.rows
    
    def __getitem__(self, idx):
        xyz_data = self.imgnp[idx]
        fft_data = self.do_fft(xyz_data, normalize = True)
        # fft_data = self.do_fft(xyz_data, normalize = False)
        image = torch.tensor(fft_data, dtype=torch.float)
        image = image.view(3, 32, 32)  # (channel, height, width)
        label = torch.tensor(self.labels[idx], dtype = torch.long)
        return (image, label)
        
    def do_fft(self, xyz_data, normalize = True):
        
        n = len(xyz_data)
        x_data = fft(xyz_data[:int(n/3)])
        y_data = fft(xyz_data[int(n/3):int(2*n/3)])
        z_data = fft(xyz_data[int(2*n/3):])
        N = int(len(x_data)/2)
        x_data = np.abs(x_data[range(N)])
        y_data = np.abs(y_data[range(N)])
        z_data = np.abs(z_data[range(N)])
        
        if normalize:
            x_data /= np.max(x_data)
            y_data /= np.max(y_data)
            z_data /= np.max(z_data)
        
        
        res = np.array([])
        for _data in [x_data,y_data,z_data]:
            res = np.append(res,_data)
        
        return res
        
        

if not os.path.exists('./imgs'):
    os.mkdir('./imgs')

def to_img(x):
    x = x.view(x.size(0), 3, 32, 32)
    return x


#hyper-params
num_epochs = 5000
batch_size = 8
num_class = 3
display_freq = 250
lr=0.0001


_train_data = preprocess('train',rows=60000)
train_dataset = PHM_dataset(_train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_loader_no_shuffle = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

_test_data1 = preprocess('test1',rows=15000)
_test_data2 = preprocess('test2',rows=15000)
_test_data = merge_testdata(_test_data1, _test_data2)

test_dataset = PHM_dataset(_test_data)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("len of train_loader = %d" %len(train_loader))
print("len of test_loader = %d" %len(test_loader))
print("len of train data = %d" %len(train_dataset))
print("len of test data = %d" %len(test_dataset))



if __name__ == "__main__":
    
    parser=argparse.ArgumentParser()
    help_="train the model"
    parser.add_argument("-t","--train",help=help_,action="store_true")
    help_="verify the model"
    parser.add_argument("-v","--verify",help=help_,action="store_true")
    help_="load the previous weights"
    parser.add_argument("-w","--weights",help=help_)
    args=parser.parse_args()
    
    model = StackedAutoEncoder().cuda()
    
    classifier = nn.Linear(512 * 16, num_class).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    
    
    if args.train:
        
        if args.weights:
            model.load_state_dict(torch.load(args.weights))
        
        for epoch in range(num_epochs):
            
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
            
        torch.save(model.state_dict(), './PHM_CDAE.pth')
        
        #save self-defined model
        ensemble = EnsembleModel(model, classifier).cuda()
        torch.save(ensemble.state_dict(), './PHM_ENSEMBLE.pth')
    
    if args.weights and args.verify:
        
        ensemble = EnsembleModel(model, classifier).cuda()
        ensemble.load_state_dict(torch.load(args.weights))
        ensemble.eval()
        #------------------------confusion matrix--------------------------#
        print('Calculate confusion matrix of testing data....')
        start = time.time()
        predictions = torch.LongTensor()
        
        for i, data in enumerate(test_loader, 1):
            images, labels = data
            images = images.cuda()
            
            outputs = ensemble(images)
            
            pred = outputs.cpu().data.max(1, keepdim=True)[1]
            predictions = torch.cat((predictions, pred), dim=0)
            
        print('Test Completed in {} secs'.format(time.time() - start))
        test_matrix = confusion_matrix(test_dataset.labels , predictions.numpy())
        
        
        print('Calculate confusion matrix of training data....')
        start = time.time()
        
        predictions = torch.LongTensor()
        for i, data in enumerate(train_loader_no_shuffle, 1):
            images, labels = data
            images = images.cuda()

            outputs = ensemble(images)
            
            pred = outputs.cpu().data.max(1, keepdim=True)[1]
            predictions = torch.cat((predictions, pred), dim=0)
            
        print('Train Completed in {} secs'.format(time.time() - start))
        train_matrix = confusion_matrix(train_dataset.labels , predictions.numpy())
        
        
        for matrix , filename in zip([test_matrix,train_matrix],["test_matrix.csv", "train_matrix.csv"]):
            df_mat = pd.DataFrame(matrix)
            df_mat.to_csv(filename)
            
        print("test matrix:\n",test_matrix)
        print("train matrix:\n",train_matrix)
        
        #------------------------confusion matrix--------------------------#
