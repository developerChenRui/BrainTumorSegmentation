import _init_paths
import tools
import caffe
import numpy as np
import os.path as osp
import os
import sys
import h5py
import scipy
import matplotlib.pyplot as plt
from caffe import layers as L, params as P
from caffe import to_proto
import matplotlib.image as mpimg
from sklearn.feature_extraction.image import extract_patches_2d

this_dir = osp.dirname(__file__)

def conv_BN_scale_relu(split,bottom,nout,ks,stride,pad,bias_term=False):
    conv = L.Convolution(bottom,kernel_size = ks,stride = stride,num_output = nout,
                            pad = pad,bias_term = bias_term,
                            weight_filler = dict(type = 'xavier'),
                            bias_filler = dict(type = 'constant'))
    if split == 'train':
        use_global_stats = False
    else:
        use_global_stats = True
    BN = L.BatchNorm(conv, batch_norm_param = dict(use_global_stats = use_global_stats),
                     in_place = True,
                     moving_average_fraction = 0.95,
                     param = [dict(lr_mult = 0, decay_mult = 0),
                              dict(lr_mult = 0, decay_mult = 0),
                              dict(lr_mult = 0, decay_mult = 0)])
    scale = L.Scale(BN, scale_param = dict(bias_term = True), in_place = True)
    relu = L.ReLU(scale, in_place = True)
    return scale, relu


def make_net():
    with open(this_dir +'/mri_model/train.prototxt', 'w') as f:
        f.write(str(ResNet('train')))
    with open(this_dir +'/mri_model/test.prototxt', 'w') as f:
        f.write(str(ResNet('test')))
   
def ResNet(split):
    
    data, labels = L.Python(module = 'readDataLayer',
						layer = 'input_layer',
						ntop = 2,
						param_str = str(dict(split = split,
											 data_dir = this_dir + '/data/',
											 train_data_name = 'train_',
											 test_data_name = 'test',
											 train_batches = 128,
											 test_batches = 128,
											 crop_size_x = 33,
                                                  crop_size_y = 33,
                                                  train_pack_nums = 9,
                                                  test_pack_nums = 1)))
    HGG_1,_ = conv_BN_scale_relu(split, data, 64, 3, 1, 0)						
    HGG_2,_ = conv_BN_scale_relu(split, HGG_1, 64, 3, 1, 0)						
    HGG_3,_  = conv_BN_scale_relu(split, HGG_2, 64, 3, 1, 0)
    HGG_4  = L.Pooling(HGG_3, pool = P.Pooling.MAX, global_pooling = False,
	                     stride = 2,kernel_size = 3);	
			
    HGG_5,_  = conv_BN_scale_relu(split, HGG_4, 128, 3, 1, 0)
					
    HGG_6,_  = conv_BN_scale_relu(split, HGG_5, 128, 3, 1, 0)
					
    HGG_7,_  = conv_BN_scale_relu(split, HGG_6, 128, 3, 1, 0)

    HGG_8  = L.Pooling(HGG_7, pool = P.Pooling.MAX, global_pooling = False,
	                        stride = 2,kernel_size = 3)

    HGG_8a = L.Flatten(HGG_8)

    HGG_9 = L.ReLU(HGG_8a)
    HGG_9a = L.InnerProduct(L.Dropout(HGG_9,dropout_ratio = 0.1),num_output = 256,
                            weight_filler = dict(type = 'xavier'),
                            bias_filler = dict(type = 'constant'))
#    HGG_9a = L.InnerProduct(HGG_9, num_output = 256)
    
    HGG_10 = L.ReLU(HGG_9a)
    HGG_10a = L.InnerProduct(L.Dropout(HGG_10,dropout_ratio = 0.1),num_output = 256,
                             weight_filler = dict(type = 'xavier'), 
                             bias_filler = dict(type = 'constant'))
#    HGG_10a = L.InnerProduct(HGG_10,num_output = 256)
    

    HGG_11 = L.Dropout(HGG_10a,dropout_ratio = 0.1)
    HGG_11a = L.InnerProduct(HGG_11,num_output = 5,
                             weight_filler = dict(type = 'xavier'),
                             bias_filler = dict(type = 'constant'))
   
    acc = L.Accuracy(HGG_11a,labels)
    loss = L.SoftmaxWithLoss(HGG_11a,labels)
    return to_proto(loss,acc)
	
	
if __name__=='__main__':
    make_net()	
	
	

    solver_dir = this_dir +'/mri_model/MRI_model_solver.prototxt'
    model_dir = '/home/wangc/ResNet/mri_model/snapshot_norm/'
#    pre_model = os.path.join(model_dir,'snapshot_net_iter_37500.solverstate')
#    solver_prototxt = tools.CaffeSolver()
#    solver_prototxt.write(solver_dir)
    caffe.set_device(0)
    caffe.set_mode_gpu()
#    solver = caffe.SGDSolver(str(solver_dir))
#    solver.restore(pre_model)
#    for _ in range(500):
#	  solver.step(100)
   
    model_def = '/home/wangc/ResNet/mri_model/deploy.prototxt'
    model_weight = '/home/wangc/ResNet/mri_model/snapshot_orgdata_excludeBG/snapshot_net_iter_30000.caffemodel'
#    
    net = caffe.Net(model_def,
                    model_weight,
                    caffe.TEST)
#    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#    transformer.set_raw_scale('data', 255)
#    image = caffe.io.load_image('/home/wangc/ResNet/data/')
    
    h5file = h5py.File('/home/wangc/ResNet/data/test_1.hdf5', 'r')
    test_data = h5file['data'][...]
    label = h5file['label'][...]
    test_data = test_data.transpose(3,2,1,0)
    test_label = label.transpose(3,2,1,0)
    test_data = np.array(test_data, dtype = np.float32)
    test_label = np.array(test_label, dtype = np.float32)
    
    data_cases,data_channels,data_height,data_width = test_data.shape
    rand_index =75
    imgs = test_data[rand_index, :, :, :]
    output_data = []
                                    
    plist = []
    fullpred = []
#     create patches from an entire slice
    for img in imgs[:]:
        if np.max(img) != 0:
            img /= np.max(img)
        p = extract_patches_2d(img, (33,33))
        plist.append(p)
    patches = np.array(zip(np.array(plist[0]), np.array(plist[1]), np.array(plist[2]),np.array(plist[3])))
    cases = patches.shape
    num_cases = cases[0]
    num=0
    for numpatch in range(0,num_cases-1,128):
        train_patch = patches[numpatch:numpatch+128]
        transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
        net.blobs['data'].data[...] = train_patch
        output = net.forward()
        output_prob = net.blobs['InnerProduct3'].data
        pred = np.argmax(output_prob,axis=1)
        fullpred[numpatch:numpatch+128]=pred
        num = num +1
    fullpred = np.array(fullpred)

    output_data=fullpred.reshape(208,208)
    pad = np.zeros((240,240))
    pad[16:-16,16:-16]=output_data
    
    plt.subplot(1,3,1)
    plt.imshow(test_data[75,1,:,:],cmap = 'Greys_r')
    plt.title('image of one channel')
    
    plt.subplot(1,3,2)
    plt.imshow(test_label[75,0,:,:],cmap = 'Greys_r')
    plt.title('manual segmentation')
    
    plt.subplot(1,3,3)
    plt.imshow(pad,cmap = 'Greys_r')
    plt.show()
    plt.title('Processed by CNN')

    
















	
