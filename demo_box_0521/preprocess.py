import os
import time

import numpy as np
import pandas as pd
import json
import argparse

from os import listdir
from os.path import join, basename, dirname
import random
import torch
import pickle


def group_data():
    
    data_dict = {'order-1':{'class':'ng1',
                        'train':{'x':[],'y':[],'z':[]},
                        'test':{'x':[],'y':[],'z':[]},
                        'label':1
                        },
                'order-2':{'class':'ng2',
                        'train':{'x':[],'y':[],'z':[]},
                        'test':{'x':[],'y':[],'z':[]},
                        'label':2
                        },
                'order-3':{'class':'pass',
                        'train':{'x':[],'y':[],'z':[]},
                        'test':{'x':[],'y':[],'z':[]},
                        'label':0
                        }
                }
    
    # total 40pcs of data for each class
    n_test = 6
    n_train = 34
    
    
    # axis = ['x','y','z']
    ng1_dir = r"D:\Side Work Data\PHM\Demo_box\demo_box_0521\wav\ng1"
    ng2_dir = r"D:\Side Work Data\PHM\Demo_box\demo_box_0521\wav\ng2"
    pass_dir = r"D:\Side Work Data\PHM\Demo_box\demo_box_0521\wav\pass"
        
    for order, dir_name in enumerate([ng1_dir+'\\x', ng2_dir+'\\x', pass_dir+'\\x'], 1):
        base_list = listdir(dir_name)
        group_name = dirname(dir_name)
        random.shuffle(base_list)
        filenames_x = [join(group_name,'x',base) for base in base_list]
        filenames_y = [join(group_name,'y',base) for base in base_list]
        filenames_z = [join(group_name,'z',base) for base in base_list]
        
        train_filenames_x = filenames_x[:n_train]
        train_filenames_y = filenames_y[:n_train]
        train_filenames_z = filenames_z[:n_train]
        test_filenames_x = filenames_x[-n_test:]
        test_filenames_y = filenames_y[-n_test:]
        test_filenames_z = filenames_z[-n_test:]
        
        
        data_dict[f'order-{order}']['train']['x'] = train_filenames_x
        data_dict[f'order-{order}']['train']['y'] = train_filenames_y
        data_dict[f'order-{order}']['train']['z'] = train_filenames_z
        
        data_dict[f'order-{order}']['test']['x'] = test_filenames_x
        data_dict[f'order-{order}']['test']['y'] = test_filenames_y
        data_dict[f'order-{order}']['test']['z'] = test_filenames_z
        
    
    with open('file_list.json', 'w+') as f:
        json.dump(data_dict, f)
        f.close()
        
    print('finished!')


def pytorch_rolling_window(x, window_size, step_size=1):
    # unfold dimension to make our rolling window
    return x.unfold(0,window_size,step_size)
    
    

def convert_files2data():
    
    all_data_dict = {'order-1':{'class':'ng1',
                            'train':[],
                            'test':[]
                            },
                    'order-2':{'class':'ng2',
                            'train':[],
                            'test':[]
                            },
                    'order-3':{'class':'pass',
                            'train':[],
                            'test':[]
                            }
                    }
    
    
    rows = 60000
    window_size = 2048
    step_size = 2048
    
    
    
    
    json_data = {}
    with open('file_list.json', 'r') as f:
        json_data = json.load(f)
    
    
    for order in range(1,4):
        
        order_key = f"order-{order}"
        label = json_data[order_key]['label']
    
        for train_test_key in ['train','test']:
            
            print(train_test_key,'...')
            all_data = np.array([])
            all_label_data = np.array([])
            
            for ax in ['x','y','z']:
                
                _temp_data = np.array([])
                _temp_lable_data = np.array([])
                
                for ind, filename in enumerate(json_data[order_key][train_test_key][ax]):
                    print('processing for file {}'.format(filename))
                    data = pd.read_csv(filename).values[:rows] # 60000*1 size
                    data = torch.tensor(data, dtype=torch.float)
                    data = pytorch_rolling_window(data, window_size = window_size, step_size = step_size)
                    data = data.numpy().reshape((-1,window_size)) #important: you must reshape it!
                    n_samples = len(data)
                    label_data = np.array([label]*n_samples).reshape((-1,1))
                    
                    if ind == 0:
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
                
            
            all_data_dict[order_key][train_test_key] = np.hstack((all_data,all_label_data))
    
    
    with open('BigData.pkl','wb') as f:
        pickle.dump(all_data_dict,f)        
        f.close()
        
    print('finished!')
    

def load_pkl_data(dataset_dst):
    with open(dataset_dst, "rb") as the_file:
        return pickle.load(the_file)
        the_file.close()



def merge_big_data():

    data_merge_dict = {
                        'train':[],
                        'test':[]                      
                      }
    
    
    _big_data_dict = load_pkl_data('BigData.pkl')
    
    _train_data_ps = _big_data_dict['order-3']['train']
    _train_data_ng1 = _big_data_dict['order-1']['train']
    _train_data_ng2 = _big_data_dict['order-2']['train']

    _test_data_ps = _big_data_dict['order-3']['test']
    _test_data_ng1 = _big_data_dict['order-1']['test']
    _test_data_ng2 = _big_data_dict['order-2']['test']
    
    
    data_merge_dict['train'] = np.vstack((_train_data_ps,_train_data_ng1,_train_data_ng2))
    data_merge_dict['test'] = np.vstack((_test_data_ps,_test_data_ng1,_test_data_ng2))
    
    with open('BigData_asInput.pkl','wb') as f:
        pickle.dump(data_merge_dict,f)        
        f.close()
        
    print('finished!')
    


if __name__ == '__main__':
    # group_data() #step1
    # convert_files2data() #step2
    merge_big_data() #step3
    
        
        
        
        
        