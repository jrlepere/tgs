"""
Implementation for TGS Kaggle Competition.

Author: Jake Lepere
Date: 08/06/2018
"""

import cv2
import csv
import numpy as np
from glob import glob
from random import shuffle
from os.path import basename, join
from keras import backend as K
from keras.models import Model, load_model
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, ZeroPadding2D, Cropping2D, concatenate, BatchNormalization, Dropout, Lambda, add, Activation, RepeatVector, Reshape, UpSampling2D
from skimage.transform import warp, AffineTransform, resize, rotate, rescale
from skimage.util import img_as_ubyte, random_noise
import tensorflow as tf
from random import randint
from keras.preprocessing.image import ImageDataGenerator
from skimage.filters import gaussian
from skimage.util import random_noise
from iou import iou, get_best_threshold
from random import random
from model_input import organize_val_data, organize_test_data, DataGenerator
from rle import rle
from skimage.exposure import equalize_hist
from training_equalization import order_input
import keras.backend as K


class TGS:
    """
    Class for TGS Kaggle competition.
    """

    def __init__(self, num_models, model_dir):
        """
        Initializes an object for the TGS Kaggle competition.

        Args:
          num_models: the number of models to fit and predict
          model_dir: the directory to save/load the models
        """

        # image parameters
        self.image_height = 96
        self.image_width = 96

        # num models
        self.num_models = num_models

        # model filepath
        self.model_dir = model_dir

        # filename to save the models
        self.model_weights_filename = 'model-tgs-%s-weights.hdf5'

        # thesholds for each model
        self.thresholds = []


    def load_depths(self, depth_csv_path):
        """
        Load the csv containing the depths for each training example

        Args:
          depth_csv_path: the path to the csv file
        """

        # feedback
        print('Loading Depths ---> ', end='', flush=True)

        # open the csv file
        with open(depth_csv_path, 'r') as depth_csv:
            reader = csv.reader(depth_csv)
            next(reader, None)    # skip header
            self.depth_dict = dict((rows[0], int(rows[1])) for rows in reader)
            self.depth_avg = np.mean([i for i in self.depth_dict.values()])
            self.depth_std = np.std([i for i in self.depth_dict.values()])

        # feedback
        print('DONE', flush=True)


    def load_models(self):
        """
        Loads the model
        """

        def DoubleConv2D(layer_in, filters, kernel, norm=False, drop=None, pool=None, activation='relu'):
            conv = Conv2D(filters=filters, kernel_size=kernel, padding='same') (layer_in if drop is None else Dropout(drop) (layer_in))
            conv = BatchNormalization() (conv) if norm else conv
            conv = Activation(activation) (conv)
            conv = Conv2D(filters=filters, kernel_size=kernel, padding='same') (conv)
            conv = BatchNormalization() (conv) if norm else conv
            conv = Activation(activation) (conv)
            if pool is None:
                return conv
            else:
                return conv, MaxPooling2D(pool) (conv)

        def SingleConv2DTranspose(layer_in, filters, kernel, padding, norm=False):
            conv = Conv2DTranspose(filters=filters, kernel_size=kernel, strides=2, padding=padding) (layer_in)
            conv = BatchNormalization() (conv) if norm else conv
            conv = Activation('relu') (conv)
            return conv


        # feedback
        print('Loading Models ---> ', end='', flush=True)

        # array of models
        self.models = np.empty(self.num_models, dtype=Model)
        
        for i in range(self.num_models):

            images_in = Input((self.image_height, self.image_width, 1))
            depths_in = Input((1,))
            rows_in = Input((self.image_height, self.image_width, 1))
            depths_norm = RepeatVector(self.image_height * self.image_width) (depths_in)
            depths_norm = Reshape((self.image_height, self.image_width, 1)) (depths_norm)
            depths_norm = Lambda(lambda x : (x - self.depth_avg) / self.depth_std)(depths_norm)
            depths_norm = concatenate([depths_norm, rows_in])
            depths_norm = Conv2D(1, 1) (depths_norm)
            depths_norm = BatchNormalization() (depths_norm)
            depths_norm = Activation('sigmoid') (depths_norm)

            c1, p1 = DoubleConv2D(layer_in=images_in, filters=16, kernel=3, norm=True, drop=None, pool=2)
            c2, p2 = DoubleConv2D(layer_in=p1, filters=32, kernel=3, norm=True, drop=None, pool=2)
            c3, p3 = DoubleConv2D(layer_in=p2, filters=64, kernel=3, norm=True, drop=None, pool=2)
            c4, p4 = DoubleConv2D(layer_in=p3, filters=128, kernel=3, norm=True, drop=None, pool=2)
            c5, p5 = DoubleConv2D(layer_in=p4, filters=256, kernel=3, norm=True, drop=None, pool=2)
            c6 = DoubleConv2D(layer_in=p5, filters=512, kernel=3, norm=True, drop=None, pool=None)
            u5 = DoubleConv2D(layer_in=concatenate([SingleConv2DTranspose(layer_in=c6, filters=256, kernel=3, padding='same', norm=True), c5]), filters=256, kernel=3, norm=True, drop=0.25, pool=None)
            u4 = DoubleConv2D(layer_in=concatenate([SingleConv2DTranspose(layer_in=u5, filters=128, kernel=3, padding='same', norm=True), c4]), filters=128, kernel=3, norm=True, drop=0.25, pool=None)
            u3 = DoubleConv2D(layer_in=concatenate([SingleConv2DTranspose(layer_in=u4, filters=64, kernel=3, padding='same', norm=True), c3]), filters=64, kernel=3, norm=True, drop=0.25, pool=None)
            u2 = DoubleConv2D(layer_in=concatenate([SingleConv2DTranspose(layer_in=u3, filters=32, kernel=3, padding='same', norm=True), c2]), filters=32, kernel=3, norm=True, drop=0.25, pool=None)
            u1 = DoubleConv2D(layer_in=concatenate([SingleConv2DTranspose(layer_in=u2, filters=16, kernel=3, padding='same', norm=True), c1]), filters=16, kernel=3, norm=True, drop=0.25, pool=None)
            u1 = concatenate([u1, depths_norm])
            outputs = Conv2D(1, (1, 1)) (u1)
            outputs = BatchNormalization() (outputs)
            outputs = Activation('sigmoid') (outputs)
        
            self.models[i] = Model(inputs=[images_in, depths_in, rows_in], outputs=outputs)

        # feedback
        print('DONE', flush=True)

        self.models[i].summary()
        

    def compile_models(self, optimizer, loss):
        """
        Compile each model.

        Args:
          optimizer: the optimizer
          loss: the loss
        """
        
        # feedback
        print('Compiling Models ---> ', end='', flush=True)

        # compile each model
        for i in range(self.num_models):
            self.models[i].compile(optimizer=optimizer, loss=loss, metrics=[iou])
        
        # feedback
        print('DONE', flush=True)


    def supply_model_weights(self, filenames):
        """
        Loads hfd5 weights to the models.

        Args:
          filenames: a list of weight filenames in the model dir passed to the constructor
        """

        # verify correct number of files passed
        if self.num_models != len(filenames):
            raise Exception('# models != # filenames')

        # feedback
        print('Loading Weights ---> ', end='', flush=True)

        # load each weight
        for i in range(self.num_models):
            self.models[i].load_weights(join(self.model_dir, filenames[i]))

        # feedback
        print('DONE', flush=True)


    def save_thresholds(self, threshold_filename):
        """
        Saves the already computed thresholds. To be called after model fitting, where the thresholds are computed.

        Args:
          threshold_filename: the name of the file to save/load the per model thresholds to maximize iou
        """
        print('Saving Thresholds ---> ', end='', flush=True)
        with open(threshold_filename, 'w') as f:
            f.write('Model Thresholds')
            for threshold in self.thresholds:
                f.write('\n%.5f'%threshold)
        print('DONE', flush=True)


    def load_thresholds(self, threshold_filename):
        """
        Loads pre computed thresholds. To be called if supplying model weights.

        Args:
          threshold_filename: the name of the file to save/load the per model thresholds to maximize iou
        """
        print('Loading Thresholds ---> ', end='', flush=True)
        with open(threshold_filename, 'r') as f:
            next(f)
            for threshold in f:
                self.thresholds.append(float(threshold))
        print('DONE', flush=True)


    def fit_model(self, train_data_paths, epochs=1, batch_size=32, verbose=1, validation_split=0.2):
        """
        Fits the model with the training data.

        Args:
          train_data_paths: a list of tuples representing (image, mask) for each training example
          epochs: model.fit epochs
          batch_size: model.fit batch_size
          verbose: model.fit verbose
          validation_split: percentage of data for validation
        """

        # randomly shuffle the training files, twice
        shuffle(train_data_paths)
        shuffle(train_data_paths)

        # get number of examples
        num_examples = len(train_data_paths)

        # create data buffers for images, masks and depths
        images = np.zeros((num_examples, self.image_height, self.image_width, 1))
        masks = np.zeros((num_examples, self.image_height, self.image_width, 1))
        depths = np.zeros((num_examples,1), dtype=np.uint16)

        # feedback
        print('Loading Train Data ---> ', end='', flush=True)

        # initialize buffer for each training example
        for i in range(num_examples):
            images[i] = resize(cv2.imread(train_data_paths[i][0], cv2.IMREAD_GRAYSCALE), (self.image_height, self.image_width, 1), mode='constant')
            masks[i] = resize(cv2.imread(train_data_paths[i][1], cv2.IMREAD_GRAYSCALE), (self.image_height, self.image_width, 1), mode='constant')
            depths[i] = self.depth_dict[basename(train_data_paths[i][0])[:-4]]

        # evenly spread the training data for each fold by salt coverage
        images, masks, depths = order_input(images, masks, depths, num_folds=self.num_models, by='mask_count')

        # feedback
        print('DONE', flush=True)

        # get number validation examples
        num_validation = num_examples * validation_split
        
        # the length to shift the validation window
        validation_shift_length = 0.0
        if self.num_models > 1:
            validation_shift_length = (num_examples * (1.0 - validation_split)) / (self.num_models - 1)

        # fit each model
        for i in range(self.num_models):

            # left and right index of the validation chunk
            vil = i * validation_shift_length
            vir = vil + num_validation

            # convert indeces to integers
            vil = int(vil)
            vir = int(vir)

            # feedback
            print('\nTraining Model %d/%d w/ Validation Split [%d:%d]' % (i+1, self.num_models, vil, vir))

            train_gen = DataGenerator(images=np.concatenate((images[:vil],images[vir:]), axis=0),
                    masks=np.concatenate((masks[:vil],masks[vir:]), axis=0),
                    depths=np.concatenate((depths[:vil],depths[vir:]), axis=0),
                    batch_size=batch_size,
                    shuffle=True
                    )
            
            val_data = organize_val_data(images=images[vil:vir],
                    masks=masks[vil:vir],
                    depths=depths[vil:vir]
                    )

            self.models[i].fit_generator(generator=train_gen,
                    epochs=epochs,
                    verbose=verbose,
                    validation_data=val_data,
                    callbacks=[EarlyStopping(patience=20, verbose=1, monitor='val_iou', mode='max'),
                        ReduceLROnPlateau(factor=0.5, patience=8, verbose=1, monitor='val_iou', mode='max'),
                        ModelCheckpoint(join(self.model_dir, self.model_weights_filename % i), verbose=1, save_best_only=True, save_weights_only=True, monitor='val_iou', mode='max')])

            # reload the best saved model weights
            self.models[i].load_weights(join(self.model_dir, self.model_weights_filename % i))

            # get the best threshold
            print('Getting Best Threshold ---> ', end='', flush=True)
            best_threshold = get_best_threshold(
                y_true=val_data[1],
                y_pred=self.models[i].predict(x=val_data[0], batch_size=batch_size, verbose=0),
                thresholds=np.linspace(0, 1, 101),
                )
            print('%.5f' % best_threshold)
            self.thresholds.append(best_threshold)


    def predict_and_save(self, test_data_paths, filename, min_acceptable=0.5):
        """
        Predicts masks on the testing data.

        Args:
          test_data_paths: a list of paths to the testing images
          filename: the name of the file to save the predictions
          min_acceptable: the minimum acceptable pixel value to be considered in the final mask [0,1]
        """
        
        # get number of examples
        num_examples = len(test_data_paths)

        # create data buffers for images and masks
        images = np.zeros((num_examples, self.image_height, self.image_width, 1))
        depths = np.zeros((num_examples,1), dtype=np.uint16)

        # initialize buffer for each training example
        for i in range(num_examples):
            print('Loading Test Image %d/%d\r' % (i, num_examples), end='', flush=True)
            images[i] = resize(cv2.imread(test_data_paths[i], cv2.IMREAD_GRAYSCALE), (self.image_height, self.image_width, 1), mode='constant')
            depths[i] = self.depth_dict[basename(test_data_paths[i])[:-4]]
        print('Loading Test Image %d/%d' % (num_examples, num_examples), flush=True)

        # buffer for predictions on each model
        predictions = np.zeros((self.num_models, num_examples, self.image_height, self.image_width, 1))

        # organize model input
        model_input = organize_test_data(images=images, depths=depths) 

        # predict on each model
        for i in range(self.num_models):
            print('Predicting Model #%d ---> ' % (i+1))
            predictions[i] = self.models[i].predict(
                    x=model_input, 
                    batch_size=16,
                    verbose=1)

        # compute the final predictions
        final_predictions = np.mean(predictions, axis=0)

        # write output file
        with open(filename, 'w') as output_file:
           
            # write csv header
            output_file.write('id,rle_mask')

            # calculate and write run length encoding for each mask
            for i in range(num_examples):

                # feedback
                print('Writing Test Mask %d/%d\r' % (i, num_examples), end='')

                # write new line
                output_file.write('\n')

                # write id
                output_file.write(basename(test_data_paths[i])[:-4])

                # write comma
                output_file.write(',')

                # get and write run length encoding
                output_file.write(rle(gaussian(resize(final_predictions[i], (101, 101), preserve_range=True, order=3), min_acceptable)))
            
            # print final
            print('Writing Test Mask %d/%d' % (num_examples, num_examples))


if __name__ == '__main__':
    tgs = TGS(num_models=5, model_dir='./models')
    tgs.load_depths('./depths.csv')
    tgs.load_models()
    tgs.compile_models(optimizer='adam', loss='binary_crossentropy')
    tgs.fit_model([(image, image.replace('images','masks')) for image in glob('./train/images/*png')], epochs=200, batch_size=16, verbose=1, validation_split=0.2)
    tgs.save_thresholds(threshold_filename='./thresholds.txt')
    #tgs.supply_model_weights([basename(f) for f in glob('./models/*.hdf5')])
    #tgs.load_thresholds(threshold_filename='./thresholds.txt')
    tgs.predict_and_save(glob('./test/images/*png'), filename='predictions.csv', min_acceptable=0.5)

