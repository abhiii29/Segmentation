#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:40:58 2020
@author: Luciferden
"""
import matplotlib
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from functools import partial
from PIL import Image, ImageTk
import os, glob, shutil
import tensorflow as tf
import platform
import tables
import h5py
import csbdeep
import tensorflow as tf

mfw,mfh=600,450
nbfw,nbfh=550,40

class App(tk.Frame):
    def __init__(self,master):
        tk.Frame.__init__(self,master)
        self.master.title("Segmentation with neuronal networks")
        self.master.resizable(False,False)
        self.master.tk_setPalette(background='#ececec')
        x=int((self.master.winfo_screenwidth()-self.master.winfo_reqwidth())/1.5)
        y=int((self.master.winfo_screenheight()-self.master.winfo_reqheight())/2)
        self.master.geometry('{}x{}'.format(x,y))
        
        ##Create the common frame
		##For Root_path
        #self.bfiles=tk.Frame(master, width = mfw, height=mfh*1)
        #self.fpath=tk.Label(self.bfiles,text=fieldsprep[0][0])
        #self.fp=tk.StringVar(self.bfiles)
        #self.fpath_val=tk.Entry(self.bfiles,textvariable=self.fp)
        #self.browse_button = tk.Button(self.bfiles,text="Browse", fg="green",command=self.browseSt)
        #
        #self.fpath.grid(row=0,column=0)
        #self.fpath_val.grid(row=0,column=1)
        #self.browse_button.grid(row=0,column=2)       
        #
        #self.fext=tk.Label(self.bfiles,text=fieldsprep[1][0])
        #self.fe=tk.StringVar(self.bfiles,value=fieldsprep[1][1])
        #self.fext_val=tk.Entry(self.bfiles,textvariable=self.fe)
        #self.fext.grid(row=2,column=0)
        #self.fext_val.grid(row=2,column=1)
        #
        #
        #self.bfiles.pack(side="top")
		#
		##For Config_path
        #self.bfiles1=tk.Frame(master, width = mfw, height=mfh*1)
        ##self.fpath1=tk.Label(self.bfiles1,text=fieldsprep[0][0])
        #self.fp1=tk.StringVar(self.bfiles1)
        #self.fpath_val1=tk.Entry(self.bfiles1,textvariable=self.fp1)
        #self.browse_button1 = tk.Button(self.bfiles1,text="Browse", fg="green",command=self.browseSt1)
        #
        ##self.fpath1.grid(row=0,column=0)
        #self.fpath_val1.grid(row=0,column=1)
        #self.browse_button1.grid(row=0,column=2)       
        #
        ##self.fext1=tk.Label(self.bfiles,text=fieldsprep[1][0])
        ##self.fe1=tk.StringVar(self.bfiles,value=fieldsprep[1][1])
        ##self.fext_val1=tk.Entry(self.bfiles,textvariable=self.fe1)
        ##self.fext1.grid(row=1,column=2)
        ##self.fext_val1.grid(row=2,column=1)
        
        
        #self.bfiles1.pack(side="top")
		
        #Create Notebooks
        self.nbfr=tk.Frame(master, width = mfw, height=mfh)
        self.nbfr.pack(side="top")
        self.n=ttk.Notebook(self.nbfr)
        self.prep_frame=tk.Frame(self.n, width = mfw, height=mfh-40)   
        self.train_frame=tk.Frame(self.n, width = mfw, height=mfh-40)
        self.n.add(self.prep_frame, text='Weka Segmentation')
        self.n.add(self.train_frame, text='Deep Learning')
        self.n.pack()
        
        # Create the main containers to pre-process the data
        tk.Label(self.prep_frame,text="Weka Segmentation").grid(row=0)
        self.cen_frame_prep=tk.Frame(self.prep_frame, width = nbfw, height=nbfh*len(prepfields[1:]), pady=3)
        self.btm_frame_prep=tk.Frame(self.train_frame, width = nbfw, height=nbfh, pady=3)
        
        # Layout the containers
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.cen_frame_prep.grid(row=1, sticky="ew")
        self.btm_frame_prep.grid(row=2, sticky="ew")

        self.entsprepr=self.makeFrame(self.cen_frame_prep,fieldsprep)
        
        
        # Widgets of the bottom frame
        self.quit_button = tk.Button(self.cen_frame_prep, text='Quit', command=self.cancel_b)
        self.quit_button.grid(row=5,column=4)

        root.bind('<Return>', (lambda event: self.fetch(self.entsprepr))) 

        # Create the main containers for the FT destriping notebook
        tk.Label(self.train_frame,text="Train Model").grid(row=0)
        self.cen_frame_train=tk.Frame(self.train_frame, width = nbfw, height=nbfh*len(trfields[1:]), pady=3)
        self.trmodel_frame=tk.Frame(self.train_frame, width = nbfw, height=nbfh, pady=3)
        self.btm_frame_train=tk.Frame(self.train_frame, width = nbfw, height=nbfh, pady=3)
        
        # Layout the containers
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.cen_frame_train.grid(row=1, sticky="ew")
        self.trmodel_frame.grid(row=2, sticky="ew")
        self.btm_frame_train.grid(row=3, sticky="ew")
		
        self.entstrain=self.makeFrame(self.cen_frame_train,fieldstrain)
        
        # Widgets of the bottom frame
        #self.augm_button = tk.Button(self.btm_frame_train,text="Augment", fg='Red')
        self.loaddata_button = tk.Button(self.btm_frame_train,text="TRAIN", fg='Red', command=partial(self.loading_data))
        #self.train_button = tk.Button(self.btm_frame_train,text="Train", fg="Red", command=partial(self.training,self.entstrain))
        self.quit_button = tk.Button(self.btm_frame_train, text='Quit', command=self.cancel_b)
        #self.predict_button = tk.Button(self.btm_frame_train,text="Predict", fg="Red",command=partial(self.predict,self.entstrain))
        #self.augm_button = tk.Button(self.btm_frame_train,text="Augment", fg="Red")
        self.loaddata_button.grid(row=0,column=3)
        #self.augm_button.grid(row=0,column=1)
        #self.train_button.grid(row=0,column=0)
        self.quit_button.grid(row=0,column=2)
        #self.predict_button.grid(row=0,column=5)
        #self.augm_button.grid(row=0,column=4)
        
        root.bind('<Tab>', (lambda event: self.fetch(self.entstrain))) 




    def cancel_b(self):
        self.quit()
        self.master.destroy()
        

    def browseSt(self):
        idir='/'
        if 'Win' in platform.system():
            idir = 'W:/'
        if 'Darwin' in platform.system():
            idir = "/Volumes/Data/Luca_Work/MPI/Science/Coding/Python/Segm"
        if 'Linux' in platform.system():
            idir = '/usr/people/home/bertinetti/Data/test_ML'
        dirname = tk.filedialog.askdirectory(initialdir = idir ,title = 'Select root path for images')
        if dirname:
            self.fp.set(dirname)
			
			
    def browseSt1(self):
        idir='/'
        if 'Win' in platform.system():
            idir = 'W:/'
        if 'Darwin' in platform.system():
            idir = "/Volumes/Data/Luca_Work/MPI/Science/Coding/Python/Segm"
        if 'Linux' in platform.system():
            idir = '/usr/people/home/bertinetti/Data/test_ML'	
        dirname1 = tk.filedialog.askdirectory(initialdir = idir ,title = 'Select config path for images')
        if dirname1:
            self.fp1.set(dirname1)
            
    
    def makeFrame(self,parent,fieldz):
        entries=[]
        #entries.append((fieldz[0][0],self.fpath_val))
        #entries.append((fieldz[1][0],self.fext_val))        
        #if len(fieldz)>2 and fieldz[2][0]==fieldstrain[2][0]:
            #entries.append(('Train Model',self.trmodelstr))
            #entries.append(('Train backbone',self.trbbstr))
        for i in range(2,len(fieldz)):
           lab = tk.Label(parent, width=25, text=fieldz[i][0], anchor='w')
           ent_txt=tk.StringVar(parent,value=fieldz[i][1])
           ent = tk.Entry(parent,textvariable=ent_txt)
           ent.config(justify=tk.RIGHT)
           lab.grid(row=i,column=0)
           ent.grid(row=i,column=1)
           entries.append((fieldz[i][0], ent))
        return entries
		
		
    def loading_data(self):
        #Loading Files and Plot examples
        basepath='data/side/'
        training_original_dir='training/original/'
        training_ground_truth_dir='training/ground_truth/'
        validation_original_dir='validation/original/'
        validation_ground_truth_dir='validation/ground_truth/'
        import glob
        from skimage import io
        from matplotlib import pyplot as plt
        training_original_files=sorted(glob.glob(basepath+training_original_dir+'*.tif'))
        training_original_file=io.imread(training_original_files[0])
        training_ground_truth_files=sorted(glob.glob(basepath+training_ground_truth_dir+'*.tif'))
        training_ground_truth_file=io.imread(training_ground_truth_files[0])
        print("Training dataset's number of files and dimensions: ",len(training_original_files), training_original_file.shape, len(training_ground_truth_files), training_ground_truth_file.shape)
        training_size=len(training_original_file)
        
        validation_original_files=sorted(glob.glob(basepath+validation_original_dir+'*.tif'))
        validation_original_file=io.imread(validation_original_files[0])
        validation_ground_truth_files=sorted(glob.glob(basepath+validation_ground_truth_dir+'*.tif'))
        validation_ground_truth_file=io.imread(validation_ground_truth_files[0])
        print("Validation dataset's number of files and dimensions: ",len(validation_original_files), validation_original_file.shape, len(validation_ground_truth_files), validation_ground_truth_file.shape)
        validation_size=len(validation_original_file)
        
        if training_size==validation_size:
            size=training_size
        else:
            print('Training and validation images should be of the same dimensions!')
        
        plt.figure(figsize=(16,4))
        plt.subplot(141)
        plt.imshow(training_original_file)
        plt.subplot(142)
        plt.imshow(training_ground_truth_file)
        plt.subplot(143)
        plt.imshow(validation_original_file)
        plt.subplot(144)
        plt.imshow(validation_ground_truth_file) 
		
		#preparing inputs for NN  from pairs of 32bit TIFF image files with intensities in range 0..1. . Run it only once for each new dataset
        from csbdeep.data import RawData, create_patches, no_background_patches
        training_data = RawData.from_folder (
        basepath    = basepath,
        source_dirs = [training_original_dir],
        target_dir  = training_ground_truth_dir,
        axes        = 'YX',
        )
        
        validation_data = RawData.from_folder (
        basepath    = basepath,
        source_dirs = [validation_original_dir],
        target_dir  = validation_ground_truth_dir,
        axes        = 'YX',
        )
        
        # pathces will be created further in "data augmentation" step, 
        # that's why patch size here is the dimensions of images and number of pathes per image is 1
        size1=64
        X, Y, XY_axes = create_patches (
        raw_data            = training_data,
        patch_size          = (size1,size1),
        patch_filter        = no_background_patches(0),
        n_patches_per_image = 1,
        save_file           = basepath+'training.npz',
        )
        
        X_val, Y_val, XY_axes = create_patches (
        raw_data            = validation_data,
        patch_size          = (size1,size1),
        patch_filter        = no_background_patches(0),
        n_patches_per_image = 1,
        save_file           = basepath+'validation.npz',
        )
		
		#Loading training and validation data into memory
        from csbdeep.io import load_training_data
        (X,Y), _, axes = load_training_data(basepath+'training.npz', verbose=False)
        (X_val,Y_val), _, axes = load_training_data(basepath+'validation.npz', verbose=False)
        X.shape, Y.shape, X_val.shape, Y_val.shape
        from csbdeep.utils import axes_dict
        c = axes_dict(axes)['C']
        n_channel_in, n_channel_out = X.shape[c], Y.shape[c]
		
        batch=len(X) # You should define number of batches according to the available memory
        #batch=1
        seed = 1
        from keras.preprocessing.image import ImageDataGenerator
        data_gen_args = dict(samplewise_center=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            fill_mode='reflect',
            rotation_range=30,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
            #width_shift_range=0.2,
            #height_shift_range=0.2,
            )
        
        # training
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
        image_datagen.fit(X, augment=True, seed=seed)
        mask_datagen.fit(Y, augment=True, seed=seed)
        image_generator = image_datagen.flow(X,batch_size=batch,seed=seed)
        mask_generator = mask_datagen.flow(Y,batch_size=batch,seed=seed)
        generator = zip(image_generator, mask_generator)
        
        # validation
        image_datagen_val = ImageDataGenerator(**data_gen_args)
        mask_datagen_val = ImageDataGenerator(**data_gen_args)
        image_datagen_val.fit(X_val, augment=True, seed=seed)
        mask_datagen_val.fit(Y_val, augment=True, seed=seed)
        image_generator_val = image_datagen_val.flow(X_val,batch_size=batch,seed=seed)
        mask_generator_val = mask_datagen_val.flow(Y_val,batch_size=batch,seed=seed)
        generator_val = zip(image_generator_val, mask_generator_val)
        
        # plot examples
        x,y = generator.__next__()
        x_val,y_val = generator_val.__next__()
        
        plt.figure(figsize=(16,4))
        plt.subplot(141)
        plt.imshow(x[0,:,:,0])
        plt.subplot(142)
        plt.imshow(y[0,:,:,0])
        plt.subplot(143)
        plt.imshow(x_val[0,:,:,0])
        plt.subplot(144)
        plt.imshow(y_val[0,:,:,0])	

        import os
        blocks=2
        channels=16
        learning_rate=0.0004
        learning_rate_decay_factor=0.95
        epoch_size_multiplicator=20 # basically, how often do we decrease learning rate
        epochs=10
        comment='_side' # adds to model_name
        import datetime
        model_path = f'models/{datetime.date.today().isoformat()}_{blocks}_{channels}_{learning_rate}{comment}.h5'
        if os.path.isfile(model_path):
            print('Your model will be overwritten in the next cell')
        kernel_size=3
		
        from csbdeep.models import Config, CARE
        from keras import backend as K
        best_mae=1
        steps_per_epoch=len(X)*epoch_size_multiplicator 
        validation_steps=len(X_val)*epoch_size_multiplicator
        if 'model' in globals():
            del model
        if os.path.isfile(model_path):
            os.remove(model_path)
        for i in range(epochs):
            print('Epoch:', i+1)
            learning_rate=learning_rate*learning_rate_decay_factor
            config = Config(axes, n_channel_in, n_channel_out, unet_kern_size=kernel_size,
            train_learning_rate=learning_rate, unet_n_depth=blocks, unet_n_first=channels)
            model = CARE(config, '.', basedir='models')
            os.remove('models/config.json')
            if i>0:
                model.keras_model.load_weights(model_path)
            model.prepare_for_training()
            history = model.keras_model.fit_generator(generator, validation_data=generator_val, validation_steps=validation_steps, epochs=1,
            verbose=0, shuffle=True, steps_per_epoch=steps_per_epoch)
            if history.history['val_mae'][0]<best_mae:
                best_mae= history.history['val_mae'][0]
                if not os.path.exists('models/'):
                    os.makedirs('models/')
                model.keras_model.save(model_path)
                print(f'Validation MAE:{best_mae:.3E}')
            del model
            K.clear_session()
			
			
			
        # specify path to model
        model_path = 'models/2020-10-23_4_128_0.0004_side.h5'
        # specify directories with images that have to be segmented 
        # (32bit TIFF image files with intensities in range 0..1)
        #to_segment_dir=basepath+'to_segment/'
        to_segment_dir=basepath+'to_segment/'
        to_segment_images_paths=sorted(glob.glob(to_segment_dir+'*.tif'))
        # here you can specify which images to segment
        # e. g. first, sevenths and twentieth in the directory would be [0,8,20]
        # by default it is a range of all images
        to_segment_images_numbers=range(len(to_segment_images_paths))
        # directory where segmented images should be saved
        segmented_dir=basepath+'segmented/'
        if not os.path.exists(segmented_dir):
            os.makedirs(segmented_dir)
        # dimensions of input will be cropped by this number. it can be any number divisible by 2^blocks
        # for surface view
        #input_dims = (1408,1408)
        # for cross-section view
        input_dims = (960,1424)
		
		
        time.sleep(20)
        
        from csbdeep.models import Config, CARE
        config = Config(axes, n_channel_in, n_channel_out, unet_kern_size=kernel_size,
        train_learning_rate=learning_rate, unet_n_depth=blocks, unet_n_first=channels)
        model = CARE(config, '.', basedir='models')
        os.remove('models/config.json')
        model.keras_model.load_weights(model_path)
        
        for i in to_segment_images_numbers:
            image_path = to_segment_images_paths[i]
            file_name = os.path.basename(image_path)
            #
            topred = io.imread(image_path)[:input_dims[0],:input_dims[1]]
            pred = model.keras_model.predict(topred.reshape(1,topred.shape[0],topred.shape[1],1))
            io.imsave(segmented_dir+file_name, pred[0,:,:,0])
	
	
	
	
	
    #def augmentor(self):
        batch=len(X) # You should define number of batches according to the available memory
        seed = 1
        from keras.preprocessing.image import ImageDataGenerator
        data_gen_args = dict(samplewise_center=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        fill_mode='reflect',
        rotation_range=30,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        )
        
        # training
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
        image_datagen.fit(X, augment=True, seed=seed)
        mask_datagen.fit(Y, augment=True, seed=seed)
        image_generator = image_datagen.flow(X,batch_size=batch,seed=seed)
        mask_generator = mask_datagen.flow(Y,batch_size=batch,seed=seed)
        generator = zip(image_generator, mask_generator)
        
        # validation
        image_datagen_val = ImageDataGenerator(**data_gen_args)
        mask_datagen_val = ImageDataGenerator(**data_gen_args)
        image_datagen_val.fit(X_val, augment=True, seed=seed)
        mask_datagen_val.fit(Y_val, augment=True, seed=seed)
        image_generator_val = image_datagen_val.flow(X_val,batch_size=batch,seed=seed)
        mask_generator_val = mask_datagen_val.flow(Y_val,batch_size=batch,seed=seed)
        generator_val = zip(image_generator_val, mask_generator_val)
        
        # plot examples
        x,y = generator.__next__()
        x_val,y_val = generator_val.__next__()
        
        plt.figure(figsize=(16,4))
        plt.subplot(141)
        plt.imshow(x[0,:,:,0])
        plt.subplot(142)
        plt.imshow(y[0,:,:,0])
        plt.subplot(143)
        plt.imshow(x_val[0,:,:,0])
        plt.subplot(144)
        plt.imshow(y_val[0,:,:,0]) 
		
    #def training():
        import os
        blocks=4
        channels=128
        learning_rate=0.0004
        learning_rate_decay_factor=0.95
        epoch_size_multiplicator=20 # basically, how often do we decrease learning rate
        epochs=40
        
        comment='_side' # adds to model_name
        import datetime
        model_path = f'models/{datetime.date.today().isoformat()}_{blocks}_{channels}_{learning_rate}{comment}.h5'
        if os.path.isfile(model_path):
            print('Your model will be overwritten in the next cell')
        kernel_size=3
		
        from csbdeep.models import Config, CARE
        from keras import backend as K
        best_mae=1
        steps_per_epoch=len(X)*epoch_size_multiplicator 
        validation_steps=len(X_val)*epoch_size_multiplicator
        if 'model' in globals():
            del model
        if os.path.isfile(model_path):
            os.remove(model_path)
        for i in range(epochs):
            print('Epoch:', i+1)
            learning_rate=learning_rate*learning_rate_decay_factor
            config = Config(axes, n_channel_in, n_channel_out, unet_kern_size=kernel_size,
                            train_learning_rate=learning_rate, unet_n_depth=blocks, unet_n_first=channels)
            model = CARE(config, '.', basedir='models')
            os.remove('models/config.json')
            if i>0:
                model.keras_model.load_weights(model_path)
            model.prepare_for_training()
            history = model.keras_model.fit_generator(generator, validation_data=generator_val, validation_steps=validation_steps, epochs=1,
                                                 verbose=0, shuffle=True, steps_per_epoch=steps_per_epoch)
            if history.history['val_mae'][0]<best_mae:
                best_mae= history.history['val_mae'][0]
                if not os.path.exists('models/'):
                    os.makedirs('models/')
                model.keras_model.save(model_path)
                print(f'Validation MAE:{best_mae:.3E}')
            del model
            K.clear_session()


			
    #def predict():
	    # specify path to model
        model_path = 'models/2019-10-23_4_128_0.0004_side.h5'
        # specify directories with images that have to be segmented 
        # (32bit TIFF image files with intensities in range 0..1)
        to_segment_dir=basepath+'to_segment/'
        to_segment_images_paths=sorted(glob.glob(to_segment_dir+'*.tif'))
        # here you can specify which images to segment
        # e. g. first, sevenths and twentieth in the directory would be [0,8,20]
        # by default it is a range of all images
        to_segment_images_numbers=range(len(to_segment_images_paths))
        # directory where segmented images should be saved
        segmented_dir=basepath+'segmented/'
        if not os.path.exists(segmented_dir):
            os.makedirs(segmented_dir)
        # dimensions of input will be cropped by this number. it can be any number divisible by 2^blocks
        # for surface view
        #input_dims = (1408,1408)
        # for cross-section view
        input_dims = (960,1424)
        
        from csbdeep.models import Config, CARE
        config = Config(axes, n_channel_in, n_channel_out, unet_kern_size=kernel_size,
        train_learning_rate=learning_rate, unet_n_depth=blocks, unet_n_first=channels)
        model = CARE(config, '.', basedir='models')
        os.remove('models/config.json')
        model.keras_model.load_weights(model_path)
        
        for i in to_segment_images_numbers:
            image_path = to_segment_images_paths[i]
            file_name = os.path.basename(image_path)
            #
            topred = io.imread(image_path)[:input_dims[0],:input_dims[1]]
            pred = model.keras_model.predict(topred.reshape(1,topred.shape[0],topred.shape[1],1))
            io.imsave(segmented_dir+file_name, pred[0,:,:,0])

smooth = 1.

def check_gpu():
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
    with tf.Session() as sess:
        print (sess.run(c))


if __name__ == '__main__':
    initdir="/Volumes/Data/Luca_Work/MPI/Science/Coding/Python/Segm "
    mainfields=('Root path','Config Path')
    #mainfields=()
    maindeftexts=('','')
    #maindeftexts=()
	#prepfields=('Replica per image','Crop Ratio')
    prepfields=()
    #prepdeftxt=(3,0.8)
    prepdeftxt=()
    #trfields=('N epochs','Validation Split','N. of labels','Learning rate','Batch size')
    trfields=()
    #trdeftxt=(500,0.15,1,0.00001,16)
    trdeftxt=()
	
    fieldsprep=[]
    fieldstrain=[]
	
    if len(mainfields)==len(maindeftexts) and len(prepfields)==len(prepdeftxt):
        for i in range(len(mainfields)):
            tmp=(mainfields[i],maindeftexts[i])
            fieldsprep.append(tmp)
        for i in range(len(mainfields)):
            tmp=(mainfields[i],maindeftexts[i])
            fieldsprep.append(tmp)	
    
    #if len(mainfields)==len(maindeftexts) and len(trfields)==len(trdeftxt): 
    #    for i in range(len(mainfields)):
    #        tmp=(mainfields[i],maindeftexts[i])
    #        fieldstrain.append(tmp)
    #    for i in range(len(trfields)):
    #        tmp=(trfields[i],trdeftxt[i])
    #        fieldstrain.append(tmp)
    #    for i in range(len(mainfields)):
    #        tmp=(mainfields[i],maindeftexts[i])
    #        fieldstrain.append(tmp)

    root=tk.Tk()
    app=App(root)
    app.mainloop()

#    create_train_data(data_path,train_p,label_p,fext)
#    augment_DS(load_train_data(data_path,train_npy,labels_npy)[0],load_train_data(data_path,train_npy,labels_npy)[1],data_path,5)
#    create_train_data(data_path,train_p,label_p,fext)
#    create_test_data(data_path,test_p,fext)
#    train(data_path,train_npy,labels_npy)
#    predict(data_path,test_npy,test_id_npy)

#    tmp=load_train_data(data_path,train_npy,labels_npy)
#    tmp2=load_test_data(data_path,test_npy,test_id_npy)
