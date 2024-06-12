import keras
import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from skimage.segmentation import mark_boundaries
from keras.models import Model
from keras.layers import Add,ReLU,Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,BatchNormalization,Dropout,AveragePooling2D 
from keras.optimizers import Adam, SGD,AdamW
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from skimage.exposure import rescale_intensity
from keras.callbacks import History
from skimage import io
from matplotlib import pyplot as plt
from scipy.spatial.distance import directed_hausdorff
import cv2
import tensorflow.keras.backend as K

from sklearn.model_selection import KFold
import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D
from keras.layers import add
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
from tensorflow.keras import layers, models

class UNet:
    def __init__(self,input_shape):
        self.input_shape = input_shape
    def focal_loss(self,y_true, y_pred, gamma=2.0, alpha=0.25):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        y_pred_logit = K.log(y_pred / (1 - y_pred))
        y_pred_prob = tf.nn.sigmoid(y_pred_logit)

        cross_entropy = -y_true * K.log(y_pred_prob) - (1 - y_true) * K.log(1 - y_pred_prob)
        weight = alpha * y_true * K.pow((1 - y_pred_prob), gamma) + (1 - alpha) * (1 - y_true) * K.pow(y_pred_prob, gamma)

        loss = weight * cross_entropy
        return K.sum(loss, axis=-1) 


    def dice_coef(self,y_true, y_pred,smooth=0.1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)

        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


    def dice_coef_loss(self,y_true, y_pred):
    
        return 1-self.dice_coef(y_true, y_pred) * 0.15 + self.focal_loss(y_true, y_pred) 
    
    def evaluate_haussdorf(self,base,pred):
        distances=[]
        for i in range(base.shape[0]):
            base_image = cv2.resize(base[i], (base.shape[1], base.shape[0]))  # resize to original image size
            pred_image = cv2.resize(pred[i], (base.shape[1], base.shape[0]))  # resize to original image size
            distances.append(directed_hausdorff(base_image, pred_image)[0])
        return sum(distances)/len(distances)


    def get_unet(self):
        inputs = Input((self.input_shape[0], self.input_shape[1], 1))


        conv = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(inputs)
        conv = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)

        conv0 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(pool)
        conv0 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(conv0)
        pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)


        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(pool0)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv15 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(pool1)
        conv15 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(conv15)
        pool15 = MaxPooling2D(pool_size=(2, 2))(conv15)

        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(pool15)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(conv7)

        up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(up8)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(conv8)

        up85 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv15], axis=3)
        conv85 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(up85)
        conv85 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(conv85)

        up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv85), conv1], axis=3)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(up9)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(conv9)

        up95 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv9), conv0], axis=3)
        conv95 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(up95)
        conv95 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(conv95)

        up99 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv95), conv], axis=3)
        conv99 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(up99)
        conv99 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer="he_normal")(conv99)


        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv99)

        model = Model(inputs=[inputs], outputs=[conv10])

        model.compile(optimizer=Adam(learning_rate=0.00083), loss=self.dice_coef_loss, metrics=[self.dice_coef])

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
            model_checkpoint = ModelCheckpoint('model_ckpt{}.weights.h5'.format(len(hauss_distances)+1), monitor='val_loss', save_best_only=True,save_weights_only=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                                  patience=8, min_lr=0.00000001)      #Saving the weights and the loss of the best predictions we obtained

            history=model.fit(imgs_train, imgs_mask_train, batch_size=8, epochs=100, verbose=1, shuffle=True,callbacks=[model_checkpoint,reduce_lr],validation_split=0.25)


            plt.plot(history.history['dice_coef'])
            plt.plot(history.history['loss'])
            plt.title('Model dice coeff / Loss')
            plt.ylabel('Dice coeff')
            plt.xlabel('Epoch')

            return hauss_distances,model
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
                model_checkpoint = ModelCheckpoint('model_ckpt{}.weights.h5'.format(len(hauss_distances)+1), monitor='val_loss', save_best_only=True,save_weights_only=True)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75,
                                  patience=15, min_lr=0.00000001)   
                print('-'*30)
                print('Fitting model...')
                print('-'*30)
                history=model.fit(X_train, Y_train, batch_size=8, epochs=100, verbose=1, shuffle=True,callbacks=[model_checkpoint,reduce_lr],validation_split=0.25)

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
