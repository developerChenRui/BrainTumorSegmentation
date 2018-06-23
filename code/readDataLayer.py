import caffe
import scipy.io as scio
import os.path as osp
import h5py
import numpy as np
import random
#import read_binaryproto
#import read_lmdb
import matplotlib.pyplot as plt
import matplotlib.image as mping
from scipy import stats
from scipy.stats import mode
from sklearn.feature_extraction.image import extract_patches_2d




class input_layer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.data_dir = params['data_dir']
        self.split = params['split']
        self.train_batches = params['train_batches']
        self.test_batches = params['test_batches']
        self.train_data_name = params['train_data_name']
        self.test_data_name = params['test_data_name']
        self.crop_size_x = params['crop_size_x']
        self.crop_size_y = params['crop_size_y']
        self.train_pack_nums = params['train_pack_nums']
        self.test_pack_nums = params['test_pack_nums']

        this_dir = osp.dirname(__file__)
        self.data_path = osp.join(this_dir, '..',self.data_dir)

        

        
    def reshape(self, bottom, top):
        if self.split == 'train':
            h,w = self.crop_size_x,self.crop_size_y
            train_num = np.random.randint(1, 9+1)
            h5file = h5py.File(self.data_path + self.train_data_name + str(train_num) + '.hdf5', 'r')
            datas = h5file['data'][...]
            labelss = h5file['label'][...]            
            datas = datas.transpose(3,2,1,0)
            labelss = labelss.transpose(3,2,1,0)
            cases = datas.shape
            random_cropped_data = np.zeros((self.train_batches, 4, self.crop_size_x, self.crop_size_y), dtype = np.float32)
            random_cropped_labels = np.zeros((self.train_batches),dtype = np.int32) 
            patch_index = 0
            for i in range(0,5):
                ct = 0
                patch_num = 25
                if i==0:
                    patch_num = 28
                while ct < patch_num:                                 
                    case_num = np.random.randint(0, cases[0])
                    data = datas[case_num,:,:,:]
                    label = labelss[case_num,:,:,:]
                    label = label[0,:,:]
            # resample if class_num not in selected slice
                    if len(np.argwhere(label == i)) < 10:
                        continue
            # select centerpix (p) and patch (p_ix)            
                    p = random.choice(np.argwhere(label == i))
                    p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
                    patch = np.array([k[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for k in data])
                   
            # resample it patch is empty or too close to edge
                    while patch.shape != (4, h, w):
                        p = random.choice(np.argwhere(label == i))
                        p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
                        patch = np.array([kk[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for kk in data])
                    if patch.shape != (4, h, w) or len(np.argwhere(patch == 0)) > (h * w):
                        continue
                    # exclude background area
                    if (patch[0,16,16]+patch[1,16,16]+patch[2,16,16]+patch[3,16,16])==0:
                        continue
                    ct += 1                   
                    random_cropped_data[patch_index,:,:,:]=patch
                    random_cropped_labels[patch_index]=i
                    patch_index = patch_index+1
            for img_ix in xrange(len(random_cropped_data)):
                for slice in xrange(len(random_cropped_data[img_ix])):
                    if np.max(random_cropped_data[img_ix][slice])!=0:
                        random_cropped_data[img_ix][slice]/=np.max(random_cropped_data[img_ix][slice])
            shuffle = zip(random_cropped_data, random_cropped_labels)
            np.random.shuffle(shuffle)
            self.data = np.array([shuffle[j][0] for j in xrange(len(shuffle))])
            self.labels = np.array([shuffle[j][1] for j in xrange(len(shuffle))])
#            self.data = random_cropped_data
#            self.labels = random_cropped_labels
             

#            self.mask = random_cropped_mask
            
        elif self.split == 'test':
            self.data = np.zeros((self.test_batches, 4, self.crop_size_x, self.crop_size_y), dtype = np.float32)
            self.labels = np.zeros((self.test_batches),dtype = np.int32)
#            h5file = h5py.File('/home/wangc/ResNet/data/test_1.hdf5', 'r')
#            test_data = h5file['data'][...]
#            label = h5file['label'][...]
#            test_data = test_data.transpose(3,2,1,0)
#            test_data = np.array(test_data, dtype = np.float32)
#            data_cases,data_channels,data_height,data_width = test_data.shape
#            rand_index =70
#            imgs = test_data[rand_index, :, :, :]
#                                            
#            plist = []
#        #     create patches from an entire slice
#            for img in imgs[:]:
#                if np.max(img) != 0:
#                    img /= np.max(img)
#                p = extract_patches_2d(img[50:200,50:200], (33,33))
#                plist.append(p)
#            patches = np.array(zip(np.array(plist[0]), np.array(plist[1]), np.array(plist[2]),np.array(plist[3])))
#            self.data = patches

#    for x in range(33,data_width-33,33):
#        for y in range(33,data_height-33,33):
#            crops_x = x
#            crops_y = y
#            random_cropped_data = np.zeros((train_batches, data_channels, 33, 33), dtype = np.float32)
#            random_cropped_data = test_data[rand_index,:, crops_x : (crops_x + 33), crops_y : (crops_y + 33)]
#            for img_ix in xrange(len(random_cropped_data)):
#                for slice in xrange(len(random_cropped_data[img_ix])):
#                    if np.max(random_cropped_data[img_ix][slice])!=0:
#                        random_cropped_data[img_ix][slice]/=np.max(random_cropped_data[img_ix][slice])

            # create patches from an entire slice

#            pack_index = 1
#            h5file = h5py.File(self.data_path + self.test_data_name +'_'+ str(pack_index) + '.hdf5', 'r')
#            self.read_data = h5file['data'][...]
#            self.read_labels = h5file['label'][...]
#            
#            self.read_data = self.read_data.transpose(3,2,1,0)
#            self.read_labels = self.read_labels.transpose(3,2,1,0)
#
#            self.read_data = np.array(self.read_data, dtype = np.float32)
#            self.read_labels = np.array(self.read_labels, dtype = np.float32)
##            rand_index = [rand_index for rand_index in range(self.read_data.shape[0])]
##            self.data = self.read_data[rand_index, :, :, :]
##            self.labels = self.read_labels[rand_index, :, :, :]
#            self.data_cases, self.data_channels, self.data_height, self.data_width = self.read_data.shape 
#            self.labels_cases, self.labels_channels, self.labels_height, self.labels_width = self.read_labels.shape 
#            rand_index = np.random.random_integers(0, self.data_cases-1, size = self.train_batches)
#
#            rand_index.sort()
#            self.data = self.read_data[rand_index, :, :, :]
#            self.labels = self.read_labels[rand_index, :, :, :]
#            crops_x = np.random.random_integers(0, high = self.data_height - self.crop_size_x, size = self.test_batches)
#            crops_y = np.random.random_integers(0, high = self.data_width - self.crop_size_y, size = self.test_batches)
#            random_cropped_data = np.zeros((self.test_batches, self.data_channels, self.crop_size_x, self.crop_size_y), dtype = np.float32)
#            random_cropped_labels = np.zeros((self.test_batches), dtype = np.float32)
#
#            for i in range(self.test_batches):
#    #            i=i+1
#                tmp_data = self.data[i, :, crops_x[i] : (crops_x[i] + self.crop_size_x), crops_y[i] : (crops_y[i] + self.crop_size_y)]
#                tmp_labels = self.labels[i, :, crops_x[i] : (crops_x[i] + self.crop_size_x), crops_y[i] : (crops_y[i] + self.crop_size_y)]
##                flip1 = np.random.random_integers(0,1)
##                if flip1 == 1:
##                    tmp_data = np.fliplr(tmp_data.transpose(1,2,0))
##                    tmp_data = tmp_data.transpose(2,0,1)
##                    tmp_labels = np.fliplr(tmp_labels.transpose(1,2,0))
##                    tmp_labels = tmp_labels.transpose(2,0,1)
##                    
##                flip2 = np.random.random_integers(0,1)
##                if flip2 == 1:
##                    tmp_data = np.flipud(tmp_data.transpose(1,2,0))
##                    tmp_data = tmp_data.transpose(2,0,1)
##                    tmp_labels = np.flipud(tmp_labels.transpose(1,2,0))
##                    tmp_labels = tmp_labels.transpose(2,0,1)
##                random_cropped_mask[i, 0, :, :] = self.resd_mask[crops_x[i]:(crops_x[i]+self.crop_size_x), crops_y[i]:(crops_y[i]+self.crop_size_y)]
#                random_cropped_data[i, :, :, :] = tmp_data
#                random_cropped_labels[i] = tmp_labels[0,16,16]
#            for img_ix in xrange(len(random_cropped_data)):
#                for slice in xrange(len(random_cropped_data[img_ix])):
#                    if np.max(random_cropped_data[img_ix][slice])!=0:
#                        random_cropped_data[img_ix][slice]/=np.max(random_cropped_data[img_ix][slice])
#            
#            self.data = random_cropped_data
#            self.labels = random_cropped_labels

        
            
            
            
        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.labels.shape)
#        top[2].reshape(*self.mask.shape)
##        
#        print top[0].data.shape
#        print top[1].data.shape
#        print top[2].data.shape
#        ghufjselk
        
    def forward(self, bottom, top):
        top[0].data[...] = self.data #/ 255.0
        top[1].data[...] = self.labels #/ 255.0
#        top[2].data[...] = self.mask
        
    def backward(self, bottom, top):
        pass
    
