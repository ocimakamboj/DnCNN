import tensorflow as tf
import scipy.io as sio
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.initializers import RandomNormal
from keras import backend as Ks
from keras import optimizers
from keras import callbacks
from keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

def scaled_mse(y_true, y_pred):
    return 1000000*keras.losses.mean_squared_error(y_true,y_pred)

class DnCnn_Class_Train:
    def __init__(self):
        print('Constructor Called')
        self.IMAGE_WIDTH = 60
        self.IMAGE_HEIGHT = 60
        self.CHANNELS = 3
        self.N_SAMPLES = 1105920
        self.N_TRAIN_SAMPLES = 1024000
        self.N_EVALUATE_SAMPLES = 81920
        self.N_LAYERS = 20
        self.Filters = 64
        self.X_TRAIN = np.zeros((self.N_TRAIN_SAMPLES,self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.CHANNELS))
        self.Y_TRAIN = np.zeros((self.N_TRAIN_SAMPLES,self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.CHANNELS))
        self.X_EVALUATE = np.zeros((self.N_EVALUATE_SAMPLES,self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.CHANNELS))
        self.Y_EVALUATE = np.zeros((self.N_EVALUATE_SAMPLES,self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.CHANNELS))
        
        
        print('train data loading : start')
        path = './Data/'
    
        xpath_matfile = path + 'inputData' + '.mat'
        xname_matfile = 'inputData'
        x = sio.loadmat(xpath_matfile)
        self.X_TRAIN[:,:,:,:] = x[xname_matfile]
        
        ypath_matfile = path + 'labels' + '.mat'
        yname_matfile = 'labels' 
        y = sio.loadmat(ypath_matfile)
        self.Y_TRAIN[:,:,:,:] = y[yname_matfile]
        print('train data loading : end')
        
        print('validation data loading : start')
        x = sio.loadmat(path + 'inputDataVal.mat')
        self.X_EVALUATE[:,:,:,:] = x['inputDataVal']
        y = sio.loadmat(path + 'labelsVal.mat')
        self.Y_EVALUATE[:,:,:,:] = y['labelsVal']
        print('validation data loading : end')
        
    def ModelMaker(self, optim):
        self.myModel = Sequential()
        input = Input(shape=(self.IMAGE_WIDTH,self.IMAGE_HEIGHT,self.CHANNELS))
        firstLayer = Convolution2D(filters=self.Filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer = RandomNormal(mean=0.0, stddev=0.001, seed=None), padding='same', input_shape=(self.IMAGE_WIDTH,self.IMAGE_HEIGHT,self.CHANNELS), use_bias=True, bias_initializer='zeros')
        self.myModel.add(firstLayer)
        self.myModel.add(Activation('relu'))
        for i in range(self.N_LAYERS-2):
            Clayer = Convolution2D(filters=self.Filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer = RandomNormal(mean=0.0, stddev=0.001, seed=None), padding='same', input_shape=(self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.Filters), use_bias=True, bias_initializer='zeros')
            self.myModel.add(Clayer)
            Blayer = BatchNormalization(axis=-1, epsilon=1e-3)
            self.myModel.add(Blayer)
            self.myModel.add(Activation('relu'))
        lastLayer = Convolution2D(filters=self.CHANNELS, kernel_size=(3, 3), strides=(1, 1), kernel_initializer = RandomNormal(mean=0.0, stddev=0.001, seed=None), padding='same', input_shape=(self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.Filters), use_bias=True, bias_initializer='zeros')
        self.myModel.add(lastLayer)    
        self.myModel.compile(loss='mean_squared_error',metrics=[scaled_mse],optimizer=optim)
        print("Model Created")
        self.myModel.summary()
            
    def loadPrevModel(self,modelFileToLoad):
        self.savedModel = keras.models.load_model(modelFileToLoad,custom_objects={'scaled_mse': scaled_mse})
        self.savedModel.summary()
        self.myModel.set_weights(self.savedModel.get_weights());
        self.myModel.summary()
    
    def trainModelAndSaveBest(self, BATCH_SIZE, EPOCHS, modelFileToSave, logFileToSave):
        csv_logger = CSVLogger(logFileToSave)
        myCallback = callbacks.ModelCheckpoint(modelFileToSave, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        trainHistory = self.myModel.fit(x=self.X_TRAIN, y=self.Y_TRAIN, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[csv_logger,myCallback], validation_data=(self.X_EVALUATE,self.Y_EVALUATE))
        return trainHistory
        
    def reCompileModel(self,optim):
        self.myModel.compile(loss='mean_squared_error', metrics=[scaled_mse],optimizer=optim)
        
        
#os.environ["CUDA_VISIBLE_DEVICES"]="2"            
DnCNN = DnCnn_Class_Train();
myOptimizer = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
DnCNN.ModelMaker(myOptimizer)
#DnCNN.loadPrevModel('CDnCNN-B_V1_dash.h5')
#DnCNN.reCompileModel(myOptimizer)
myModelHistory = DnCNN.trainModelAndSaveBest(100,50,'CDnCNN-3_V1.h5','CDnCNN-3_V1.log')

