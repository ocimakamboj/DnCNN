import scipy.io as sio
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.initializers import RandomNormal
from keras import backend as Ks
from keras import optimizers
from keras import callbacks
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio
import os

def scaled_mse(y_true, y_pred):
    return 1000000*keras.losses.mean_squared_error(y_true,y_pred)

class DnCnn_Class_Test:
    
    def __init__(self,width,height,colorChannels,destFolderName,X_TEST):
        print('Constructor Called')
#        self.IMAGE_WIDTH = width
#        self.IMAGE_HEIGHT = height
        self.CHANNELS = colorChannels
        self.N_TEST_SAMPLES = 1
        self.destFolderName = destFolderName
        self.X_TEST = X_TEST
        self.N_LAYERS = 20
        self.Filters = 64
        
    def loadModelwithChangedInput(self,modelFileToLoad,width,height,numberTestFiles,X_TEST):
        self.IMAGE_WIDTH = width
        self.IMAGE_HEIGHT = height
        self.N_TEST_SAMPLES = numberTestFiles
        self.X_TEST = X_TEST
        self.savedModel = keras.models.load_model(modelFileToLoad, custom_objects={'scaled_mse': scaled_mse})
        #self.savedModel.summary()
        self.myModel = Sequential()
        firstLayer = Convolution2D(self.Filters, (3, 3), strides=(1, 1), kernel_initializer = RandomNormal(mean=0.0, stddev=0.001, seed=None), padding='same', input_shape=(self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.CHANNELS), use_bias=True, bias_initializer='zeros')
        self.myModel.add(firstLayer)
        self.myModel.add(Activation('relu'))
        for i in range(self.N_LAYERS-2):
            Clayer = Convolution2D(self.Filters, (3, 3), strides=(1, 1), kernel_initializer = RandomNormal(mean=0.0, stddev=0.001, seed=None), padding='same', input_shape=(self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.Filters), use_bias=True, bias_initializer='zeros')
            self.myModel.add(Clayer)
            Blayer = BatchNormalization(axis=-1, epsilon=1e-3)
            self.myModel.add(Blayer)
            self.myModel.add(Activation('relu'))
        lastLayer = Convolution2D(self.CHANNELS, (3, 3), strides=(1, 1), kernel_initializer = RandomNormal(mean=0.0, stddev=0.001, seed=None), padding='same', input_shape=(self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.Filters), use_bias=True, bias_initializer='zeros')
        self.myModel.add(lastLayer)    
        self.myModel.set_weights(self.savedModel.get_weights())
        print("Fresh model with changed size created")
        #self.myModel.summary()
        
    def runModelAndSaveImages(self,indexStart):
        #myOptimizer = optimizers.SGD(lr=0.002)
        if(os.path.exists(self.destFolderName)==0):
            os.makedirs(self.destFolderName)
        myOptimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.myModel.compile(loss='mean_squared_error', metrics=[scaled_mse],optimizer=myOptimizer)
        self.Y_TEST = self.myModel.predict(self.X_TEST,batch_size=1, verbose=1)
        print('output predicted')
        self.Z_TEST = self.X_TEST - self.Y_TEST
        self.Z_TEST = np.clip(self.Z_TEST,0.0,1.0)
        for i in range(self.N_TEST_SAMPLES):
            index = i + indexStart;
            if(index<10):
                patht = self.destFolderName+'predicted_0'+str(index)+'.jpg'
            else:
                patht = self.destFolderName+'predicted_'+str(index)+'.jpg'
            I = self.Z_TEST[i,:,:,:]
            I = I*255
            I = I.astype(np.uint8)
            imageio.imsave(patht, I)
        

baseFile = './Results/SISR/Urban100/Scale-2/'
folderName = baseFile + 'TestData/'
destFolderName = baseFile + 'Predicted/'
numberTestFilesMat = 37
colorChannels = 1
width = 60
height = 60
X_TEST = np.zeros((1,height,width,colorChannels))
DnCNNTest = DnCnn_Class_Test(width,height,colorChannels,destFolderName,X_TEST);
indexStart = 1;
for i in range(numberTestFilesMat):
    if(i<9):
        testFileName = 'testDataCollective0'+str(i+1)+'.mat'
    else:
        testFileName = 'testDataCollective'+str(i+1)+'.mat'
    pathr = folderName + testFileName
    x = sio.loadmat(pathr)
    X = x['testData']
    [numberTestFiles,height,width] = X.shape
    X_TEST = np.zeros((numberTestFiles,height,width,colorChannels))
    X_TEST[:,:,:,0] = X
    DnCNNTest.loadModelwithChangedInput('DnCNN-3_V2.h5',width,height,numberTestFiles,X_TEST)
    Y = DnCNNTest.runModelAndSaveImages(indexStart)
    indexStart = indexStart + numberTestFiles;

            
            
            
        
    
        
        
        
    
    

