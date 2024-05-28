
import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from skimage.segmentation import mark_boundaries
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from skimage.exposure import rescale_intensity
from keras.callbacks import History
from skimage import io
from matplotlib import pyplot as plt
from scipy.spatial.distance import directed_hausdorff
import cv2
from sklearn.model_selection import KFold

class UNet:
  def __init__(self,input_shape):
    self.input_shape = input_shape


  def dice_coef(self,y_true,y_pred,smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

  def dice_coeff_loss(self,y_true,y_pred):

    return 1 - self.dice_coef(y_true, y_pred)
  def evaluate_haussdorf(self,base,pred):
    distances=[]
    for i in range(base.shape[0]):
      base_image = cv2.resize(base[i], (base.shape[1], base.shape[0]))  # resize to original image size
      pred_image = cv2.resize(pred[i], (base.shape[1], base.shape[0]))  # resize to original image size
      distances.append(directed_hausdorff(base_image, pred_image)[0])
    return sum(distances)/len(distances)
    
  def get_unet(self):

    inputs = Input((self.input_shape[0], self.input_shape[1], 1))
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-3), loss=self.dice_coeff_loss, metrics=[self.dice_coef])

    return model

  def train_and_predict(self,imgs_train,imgs_mask_train,kfold=None):
    print('-'*30)
    print('Started pipeline with preprocessing train data...')
    print('-'*30)

    hauss_distances=[]
    hist_score=[]
    if kfold == None:
      imgs_train = imgs_train.astype('float32')
      mean = np.mean(imgs_train)  # mean for data centering
      std = np.std(imgs_train)  # std for data normalization

      imgs_train -= mean
      imgs_train /= std
      #Normalization of the train set
      imgs_mask_train = imgs_mask_train.astype('float32')

      print('-'*30)
      print('Creating and compiling model...')
      print('-'*30)
      model = self.get_unet()
      model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
      #Saving the weights and the loss of the best predictions we obtained

      print('-'*30)
      print('Fitting model...')
      print('-'*30)
      history=model.fit(imgs_train, imgs_mask_train, batch_size=8, epochs=50, verbose=1, shuffle=True,
                validation_split=0.2,
                callbacks=[model_checkpoint])
      return history

      #Normalization of the test set

      print('-'*30)
      print('Loading saved weights...')
      print('-'*30)
      model.load_weights('weights.h5')

      pred_dir = 'preds'
      if not os.path.exists(pred_dir):
          os.mkdir(pred_dir)

      #Saving our predictions in the directory 'preds'
      plt.plot(history.history['dice_coef'])
      plt.plot(history.history['val_dice_coef'])
      plt.title('Model dice coeff')
      plt.ylabel('Dice coeff')
      plt.xlabel('Epoch')
      plt.legend(['Train', 'Test'], loc='upper left')
      plt.savefig('resultplots.jpg')
      return history
    else:

      kf = KFold(n_splits=kfold, shuffle=True)
      
      for train_index, test_index in kf.split(imgs_train):
        X_train, X_test = imgs_train[train_index], imgs_train[test_index]
        Y_train, Y_test = imgs_mask_train[train_index], imgs_mask_train[test_index]

        X_train = X_train.astype('float32')
        mean = np.mean(X_train)  # mean for data centering
        std = np.std(X_train)  # std for data normalization

        X_train -= mean
        X_train /= std
        #Normalization of the train set
        Y_train = Y_train.astype('float32')

        print('-'*30)
        print('Creating and compiling model...')
        print('-'*30)
        model = self.get_unet()
        #Saving the weights and the loss of the best predictions we obtained
        model_checkpoint = ModelCheckpoint('weights{}.h5'.format(len(hauss_distances)+1), monitor='val_loss', save_best_only=True)

        print('-'*30)
        print('Fitting model...')
        print('-'*30)
        history=model.fit(X_train, Y_train, batch_size=8, epochs=50, verbose=1, shuffle=True,callbacks=[model_checkpoint],validation_split=0.2)

        predict = model.predict(X_test)
        hauss_distances.append(self.evaluate_haussdorf(Y_test,predict))
        hist_score.append(history.history)

        plt.plot(history.history['dice_coef'])
        plt.plot(history.history['loss'])
        plt.title('Model dice coeff / Loss')
        plt.ylabel('Dice coeff')
        plt.xlabel('Epoch')

        plt.savefig('resultplotsfold{}.jpg'.format(len(hauss_distances)))
        
      return hauss_distances,hist_score






