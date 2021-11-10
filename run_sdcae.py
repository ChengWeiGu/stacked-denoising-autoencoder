import time
import numpy as np
import pandas as pd
from scipy import fft

import torch
import torchvision
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import utils, models
from torchvision.utils import save_image

from model import StackedAutoEncoder, EnsembleModel



def pytorch_rolling_window(x, window_size, step_size=1):
    # unfold dimension to make our rolling window
    return x.unfold(0,window_size,step_size)



class PHM_DATASET(Dataset):
    
    def __init__(self, df, augment = True):
        
        self.preprocess(df, augment = augment, window_size = 8192, step_size = 2048)
        self.rows = len(self.all_data)
        self.imgnp = self.all_data[:self.rows, :-1]
        self.labels = self.all_data[:self.rows, -1]
        
    
    def __len__(self):
        return self.rows
    
    
    def __getitem__(self, idx):
        x_data = self.imgnp[idx]
        fft_data = self.do_fft(x_data)
        image = torch.tensor(fft_data, dtype=torch.float)
        image = image.view(1, 64, 64)  # (channel, height, width)
        label = torch.tensor(self.labels[idx], dtype = torch.long)
        return (image, label)
    
    
    def preprocess(self, df, augment, window_size, step_size):
        _temp_data = np.array([])
        _temp_lable_data = np.array([])
        filenames, labels = df['path_x'].values, df['label'].values
        for ind, filename in enumerate(filenames):
            print('processing for file {}'.format(filename))
            data = pd.read_csv(filename).values # 750000*1 size for 30 sec
            if augment:
                data = data[:int(len(data)/3),:] # 10 sec for training and testing
                data = torch.tensor(data, dtype=torch.float)
                data = pytorch_rolling_window(data, window_size = window_size, step_size = step_size)
                data = data.numpy().reshape((-1,window_size))
            else:
                data = data[:window_size].reshape((-1,window_size))
            n_samples = len(data)
            label_data = np.array([labels[ind]]*n_samples).reshape((-1,1))
            if ind == 0:
                _temp_data = data
                _temp_lable_data = label_data
            else:
                _temp_data = np.vstack((_temp_data,data))
                _temp_lable_data = np.vstack((_temp_lable_data,label_data))
        self.all_data = np.hstack((_temp_data,_temp_lable_data))
    
        
    def do_fft(self, x_data):
        x_data = fft(x_data)
        N = int(len(x_data)/2)
        x_data = np.abs(x_data[range(N)])
        
        return x_data




class SDCAE_TOOl:
    
    def __init__(self, df_train, df_valid):
        
        self.df_train = df_train
        self.df_valid = df_valid
        
    
    def set_params(self,params):
        
        self.num_epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.num_class = params['num_class']
        self.lr=params['lr']
    
    
    def build_data_loader(self):
        train_dataset = PHM_DATASET(self.df_train)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataset = PHM_DATASET(self.df_valid)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
    
    
    def build_model(self):
        self.model = StackedAutoEncoder().cuda()
        self.classifier = nn.Linear(512 * 64, self.num_class).cuda()
        self.ensemble = EnsembleModel(self.model, self.classifier).cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
    
    
    def load_weights(self, filename='./SDCAE_WEIGHTS.pth'):
        self.ensemble.load_state_dict(torch.load(filename))
        print('load pretrained model finish!')
        
    
    def train(self):
        self.build_data_loader()
        print('start training......')
        for epoch in range(1, self.num_epochs+1):
            self.model.train()
            self.classifier.train()
            total_time = time.time()
            correct = 0
            for i, data in enumerate(self.train_loader):
                img, target = data
                target = Variable(target).cuda()
                img = Variable(img).cuda()
                features = self.model(img).detach()
                prediction = self.classifier(features.view(features.size(0), -1))
                loss = self.criterion(prediction, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pred = prediction.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
            total_time = time.time() - total_time
            
            self.model.eval()
            self.classifier.eval()
            img, _ = data
            img = Variable(img).cuda()
            features, x_reconstructed = self.model(img)
            reconstruction_loss = torch.mean((x_reconstructed.data - img.data)**2)
            
            print("Epoch {} complete\tTime: {:.4f}s\t\tLoss: {:.4f}".format(epoch, total_time, reconstruction_loss))
            print("Feature Statistics\tMean: {:.4f}\t\tMax: {:.4f}\t\t".format(
                torch.mean(features.data), torch.max(features.data))
            )
            print("Linear classifier performance: {}/{} = {:.2f}%".format(correct, len(self.train_loader)*self.batch_size, 100*float(correct) / (len(self.train_loader)*self.batch_size)))
            
            
            # val_loss=0
            accuracy=0
            sub_total = 0
            with torch.no_grad():
                for j, data in enumerate(self.valid_loader,1):
                    images, labels = data

                    if(torch.cuda.is_available()):
                        images = images.cuda()
                        labels = labels.cuda()

                    features, x_reconstructed = self.model(images)
                    outputs = self.classifier(features.view(features.size(0), -1))
                    _, predicted = torch.max(outputs.data, 1)
                    sub_total += labels.size(0)
                    accuracy += (predicted == labels).sum().item()
                    # val_loss += criterion(outputs, labels).item()
            
            print("Summary at Epoch: {}/{}..".format(epoch,self.num_epochs),
                  "Val_Accu: {:.3f}%".format((accuracy/sub_total)*100))
                  
            print("="*100)
        
        self.ensemble = EnsembleModel(self.model, self.classifier).cuda()
        torch.save(self.ensemble.state_dict(), './SDCAE_WEIGHTS.pth')
        
        return True
            
    
    def predict(self,df_test):
        test_dataset = PHM_DATASET(df_test,augment=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.ensemble.eval()
        
        predictions = torch.LongTensor()
        for i, data in enumerate(self.test_loader, 1):
            images, labels = data
            images = images.cuda()
            outputs = self.ensemble(images)
            pred = outputs.cpu().data.max(1, keepdim=True)[1]
            predictions = torch.cat((predictions, pred), dim=0)
        
        return predictions.numpy().ravel()
        
        
    

if __name__ == "__main__":	
    # train_dataset in pandas
    df_train = pd.read_csv('list_train.csv',encoding = 'utf-8')
    df_train['label'] = 0 #normal
    df_train.loc[df_train['class']=='Axes_NOT_Parallel','label'] = 1 #ng1
    df_train.loc[df_train['class']=='Broken_Tooth','label'] = 2 #ng2
    
    
    # test_dataset in pandas
    df_test = pd.read_csv('list_test.csv',encoding = 'utf-8')
    df_test['label'] = 0 #normal
    df_test.loc[df_test['class']=='Axes_NOT_Parallel','label'] = 1 #ng1
    df_test.loc[df_test['class']=='Broken_Tooth','label'] = 2 #ng2
    
    
    
    # build model and set params: make vaid and test equal
    sdcae = SDCAE_TOOl(df_train=df_train, df_valid=df_test)
    sdcae.set_params(params={'epochs':15,
                            'batch_size':8,
                            'num_class':3,
                            'lr':0.0001})
    
    sdcae.build_model()
    
    
    # train sdcae and do prediction
    train_check = sdcae.train() # True means "finish"
    sdcae.load_weights('./SDCAE_WEIGHTS.pth')
    for i in range(5):
        t1 = time.time()
        result = sdcae.predict(df_test)
        print(result)
        print('time cost: {:.4f} s\n'.format(time.time()-t1))
    
    
    
    
    
    
    
    
    
    