# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 22:02:44 2021
@author: HuangHongxiang
"""
import os
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras import layers,activations
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn
from sklearn.model_selection import train_test_split
from utils import get_vgg16, data_PathWrapper, DataGenerator

image_data_root = "DATASET/xiangya_selected_10/"
mask_data_root = "DATASET/xiangya_selected_10_mask/"
seed = 520

def main():
    # ========================================
    # test if gpu is available
    gpu_device_name = tf.test.gpu_device_name()
    print(gpu_device_name)
    
    # ========================================
    # define vgg16 model
    model = get_vgg16()
    model.summary()
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
    
    # Load pretrained weights
    model.load_weights("weights0911.h5")
    
    # ========================================
    # prepare our dataset
    df = data_PathWrapper(image_data_root, mask_data_root)
    
    img_path_list = df["image_path"].values
    mask_path_list = df["mask_path"].values
    label_list = df["label"].values
    
    # ========================================
    # train test split
    img_path_list_train,img_path_list_test,label_list_train,label_list_test = train_test_split(img_path_list,label_list,test_size=0.25,random_state=seed,stratify=label_list)
    mask_path_list_train,mask_path_list_test,label_list_train,label_list_test = train_test_split(mask_path_list,label_list,test_size=0.25,random_state=seed,stratify=label_list)
    
    train_data = DataGenerator(img_path_list_train, mask_path_list_train, label_list_train)
    test_data = DataGenerator(img_path_list_test, mask_path_list_test, label_list_test)
    
    
    
    #Training.
    history = model.fit(train_data, validation_data = test_data, epochs = 30)
    model.save_weights('weights0911.h5')
    
    #visualization
    plt.style.use("ggplot")
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs=range(len(acc)) 
    plt.title('acc for train and val')
    plt.plot(epochs, acc, 'r', "TrainAcc")
    plt.plot(epochs, val_acc, 'b  ', "ValAcc")
    plt.figure()
    plt.title('loss for train and val')
    plt.plot(epochs, loss, 'r', "TrainLoss")
    plt.plot(epochs, val_loss, 'b', "ValLoss")
    plt.figure()
    
    
    # Extract Features:
    #https://blog.csdn.net/u011692048/article/details/77686208
    
    
    
    basemodel = model;
    
    Raw_dataIn = DataGenerator(img_path_list, mask_path_list, label_list, infer = True, shuffle = False)
     
    backbone = tf.keras.Model(inputs=basemodel.input,outputs=basemodel.get_layer('feature_output').output)
    
    backbone_out = backbone.predict(Raw_dataIn)
    
    print(backbone_out.shape)
    
    label_list = np.asarray(label_list,dtype=np.float64)
    
    np.save("RESULT/features_raw.npy", backbone_out)
    np.save("RESULT/corspdn_labels.npy", label_list)
    
    print('This is backbone_out',backbone_out.shape)


if __name__ == '__main__':
    main()






