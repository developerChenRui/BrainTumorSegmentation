#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:53:50 2017

@author: chenrui
"""

import _init_paths
import Tkinter as tk
from Tkinter import Label
from Tkinter import *
from tkFileDialog import *
import ttk
import h5py
import matplotlib
import numpy as np
import caffe
import matplotlib.pyplot as plt
from caffe import layers as L, params as P
from caffe import to_proto
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.image as mpimg
from sklearn.feature_extraction.image import extract_patches_2d
import tools
import numpy as np
import os.path as osp
import os
import sys
import scipy

global path_
global label
global pad
global net
global index_var
def SelectPath():
#     try:sampleCount=int(inputEntry.get())
#     except:
#         sampleCount=50
#         print '请输入整数'
#         inputEntry.delete(0,END)
#         inputEntry.insert(0,'50')
    # read the h5 file 
    global label
    global index_var
    global path_

    path_ = askopenfilename()
    path = StringVar()
    Entry(root, textvariable = path).grid(row = 0, column = 1, sticky = W)
    path.set(path_)
    
def drawPic():
    global path_
    global label
    global net
    global pad

    h5file = h5py.File(path_,'r')
    read_data = h5file['data'][...]
    read_labels = h5file['label'][...]
    read_data = read_data.transpose((3,2,1,0))
    read_data = np.array(read_data, dtype = np.float32)
    read_labels = read_labels.transpose((3,2,1,0))
    t1 = read_data[index_var.get(),0,:,:]
    t1c = read_data[index_var.get(),1,:,:]
    t2 = read_data[index_var.get(),2,:,:]
    flair = read_data[index_var.get(),3,:,:]
    label = read_labels[index_var.get(),0,:,:]
     #清空图像，以使得前后两次绘制的图像不会重叠
    drawPic.f.clf()
    
     
#     #在[0,100]范围内随机生成sampleCount个数据点
#    x=np.random.randint(0,100,size=sampleCount)
#    y=np.random.randint(0,100,size=sampleCount)
#    color=['b','r','y','g']
     
     #绘制这些随机点的散点图，颜色随机选取
    drawPic.a=drawPic.f.add_subplot(221)
    drawPic.a.imshow(t1, cmap ='gray')
    drawPic.a.axis('off')
    drawPic.a.set_title('T1')
    
    drawPic.a=drawPic.f.add_subplot(222)
    drawPic.a.imshow(t1c, cmap ='gray')
    drawPic.a.axis('off')
    drawPic.a.set_title('T1c')
    
    drawPic.a=drawPic.f.add_subplot(223)
    drawPic.a.imshow(t2, cmap ='gray')
    drawPic.a.axis('off')
    drawPic.a.set_title('T2')
    
    drawPic.a=drawPic.f.add_subplot(224)
    drawPic.a.imshow(flair, cmap ='gray')
    drawPic.a.axis('off')
    drawPic.a.set_title('Flair')
    
#    drawPic.a=drawPic.f.add_subplot(233)
#    drawPic.a.imshow(label, cmap ='gray')
#    drawPic.a.set_title('manual segmentation')
#    
#    drawPic.a=drawPic.f.add_subplot(236)
#    drawPic.a.imshow(label, cmap ='gray')
#    drawPic.a.set_title('manual segmentation')
    drawPic.canvas.show()    
    
    output_data = []                                   
    plist = []
    fullpred = []
    imgs = read_data[index_var.get(),:,:,:]
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


#def ReadFile():
#    # read the h5 file 
#    path_ = askopenfilename()
#    path.set(path_)
#    h5file = h5py.File(path_,'r')
#    read_data = h5file['data'][...]
#    read_labels = h5file['label'][...]
#    read_data = read_data.transpose((3,2,1,0))
#    read_labels = read_labels.transpose((3,2,1,0))
#    return read_data,read_labels
def SegImage():
    global label
    global index_var
    SegImage.f.clf()
    SegImage.a=SegImage.f.add_subplot(111)
    SegImage.a.imshow(label, cmap ='gray')
    SegImage.a.set_title('manual segmentation')
    SegImage.canvas.show()

def MacImage():
    global pad
    data = pad>0
    dst=morphology.remove_small_objects(data,min_size=300,connectivity=1)
    final_seg = np.multiply(dst,pad)
    SegImage.f.clf()
    SegImage.a=SegImage.f.add_subplot(111)
    SegImage.a.imshow(final_seg, cmap ='gray')
    SegImage.a.set_title('intelligent segmentation')
    SegImage.canvas.show()
    
    
    

if __name__ == '__main__':
    global label
    global index_var
    global net
    caffe.set_device(0)
    caffe.set_mode_gpu()
    model_def = '/home/wangc/ResNet/mri_model/deploy.prototxt'
    model_weight = '/home/wangc/ResNet/mri_model/snapshot_orgdata_excludeBG/snapshot_net_iter_30000.caffemodel'
#    
    net = caffe.Net(model_def,
                    model_weight,
                    caffe.TEST)

    root = tk.Tk()   
     # title
    root.title("Brain Tumor Segmentation")
     # create frame
    mainframe = ttk.Frame(root,padding = "3 3 12 12")
    mainframe.columnconfigure(0,weight=1)
    mainframe.rowconfigure(0,weight=1)
     # drawmat
#    matplotlib.use('TkAgg')
    drawPic.f = Figure(figsize=(4,4), dpi=100)
    drawPic.canvas = FigureCanvasTkAgg(drawPic.f, master=root)
    drawPic.canvas.show()
    drawPic.canvas.get_tk_widget().grid(row=1, columnspan=3)
    Label(root,text='输入路径：').grid(row=0,column=0)
    inputEntry=Entry(root)
    inputEntry.grid(row=0,column=1, sticky = W)
#    inputEntry.insert(1,'50')
    Button(root,text='路径选择',command=SelectPath).grid(row=0,column=2,columnspan=3, sticky = W)    
    path = tk.StringVar()
    
    SegImage.f = Figure(figsize=(4,4), dpi=100)
    SegImage.canvas = FigureCanvasTkAgg(SegImage.f, master=root)
    SegImage.canvas.show()
    SegImage.canvas.get_tk_widget().grid(row=1, column = 5,columnspan=3)
    
    Radiobutton(root,text='人工分割',command=SegImage).grid(row=0,column=5, sticky = E)       
    Radiobutton(root,text='智能分割',command=MacImage).grid(row=0,column=6, sticky = E)
    Label(root,text='选择图片(1~3410):').grid(row=2,column=0,sticky = W)
    
    index_var = IntVar()
    inputEntry=Entry(root,textvariable=index_var)
    inputEntry.grid(row=2,column=1, sticky = W)
    Button(root,text='确定选择',command=drawPic).grid(row=2,column=2, sticky = W)

    
    root.mainloop()