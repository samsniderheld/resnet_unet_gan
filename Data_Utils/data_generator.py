"""defines the data generators required for traininng"""
import glob
import os
import cv2
import tensorflow as tf
import numpy as np

class DataGeneratorGen(tf.keras.utils.Sequence):
    """data generator for generator pretraining and normal training"""
    def __init__(self, shuffle=True,img_dims=512, batch_size = 64, data_size = -1):
        self.shuffle = shuffle
        self.input_dir = os.path.join("unet_data/X/*")
        self.output_dir = os.path.join("unet_data/Y/*")
        self.batch_size = batch_size
        self.input_files  = sorted(glob.glob(self.input_dir))
        self.output_dir  = sorted(glob.glob(self.output_dir))
        self.all_files = list(zip(self.input_files,self.output_dir))[:data_size]
        self.img_dims = img_dims
        self.on_epoch_end()
        self.count = self.__len__()
        print("number of all samples = ", len(self.all_files))


    def __len__(self):
        """Denotes the number of batches per epoch"""
        self.num_batches = int(np.floor(len(self.all_files) / self.batch_size))
        return self.num_batches

    def __getitem__(self, index):
        """gets batch of x and y samples"""
        x_data,y_data = self.__data_generation(index)

        return x_data,y_data

    def on_epoch_end(self):
        """called at the end of an epoch"""
        if self.shuffle is True:
            np.random.shuffle(self.all_files)

    def resize(self, input_image, real_image, height, width):
        """does resizing for batches"""
        input_image = tf.image.resize(input_image, [height, width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return input_image, real_image


    def random_crop(self,input_image, real_image):
        """randomly crops image for batches"""
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[2, self.img_dims, self.img_dims, 3])

        return cropped_image[0], cropped_image[1]

    def random_jitter(self, input_image, real_image):
        """does resizing and random crop"""
        input_image, real_image = self.resize(input_image, real_image, self.img_dims+30, self.img_dims+30)

        # Random cropping back to 256x256
        input_image, real_image = self.random_crop(input_image, real_image)

        if tf.random.uniform(()) > 0.5:
            # Random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image


    def __data_generation(self, idx):
        """Generates data containing batch_size samples"""

        batch_files = self.all_files[idx*self.batch_size:idx*self.batch_size+self.batch_size]
   
        x_data = np.empty((self.batch_size,self.img_dims,self.img_dims,3))
        y_data = np.empty((self.batch_size,self.img_dims,self.img_dims,3))

        for i, batch_file in enumerate(batch_files):

            img_x = cv2.imread(batch_file[0])
            img_y = cv2.imread(batch_file[1])

            img_x = tf.keras.applications.resnet50.preprocess_input(img_x)
            img_y = tf.keras.applications.resnet50.preprocess_input(img_y)

            x_data[i],y_data[i] = self.random_jitter(img_x, img_y)

        return x_data,y_data

class DataGeneratorDisc(tf.keras.utils.Sequence):
    """generator for discriminator pre training"""
    def __init__(self, shuffle=True,img_dims=512, batch_size = 64, ):
        self.shuffle = shuffle
        self.input_dir = os.path.join("unet_data/Y/*")
        self.output_dir = os.path.join("fakes_dir/*")
        self.batch_size = batch_size
        self.input_files  = sorted(glob.glob(self.input_dir))
        self.output_dir  = sorted(glob.glob(self.output_dir))
        self.all_files = list(zip(self.input_files,self.output_dir))
        self.img_dims = img_dims
        self.on_epoch_end()
        self.count = self.__len__()
        print("number of all samples = ", len(self.all_files))


    def __len__(self):
        """Denotes the number of batches per epoch"""
        self.num_batches = int(np.floor(len(self.all_files) / self.batch_size))
        return self.num_batches

    def __getitem__(self, index):
        """gets samples for batchg"""
        x_data,y_data = self.__data_generation(index)

        return x_data,y_data

    def on_epoch_end(self):
        """called when epoch ends"""
        if self.shuffle is True:
            np.random.shuffle(self.all_files)

    def resize(self, input_image, real_image, height, width):
        """resize image so we can do a random crop"""
        input_image = tf.image.resize(input_image, [height, width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return input_image, real_image


    def random_crop(self,input_image, real_image):
        """random crop functionality"""
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(stacked_image,
            size=[2, self.img_dims, self.img_dims, 3])

        return cropped_image[0], cropped_image[1]

    def random_jitter(self, input_image, real_image):
        """random movement of cropping box"""
        # Resizing to 286x286
        input_image, real_image = self.resize(input_image, real_image,
            self.img_dims+30, self.img_dims+30)

        # Random cropping back to 256x256
        input_image, real_image = self.random_crop(input_image, real_image)

        if tf.random.uniform(()) > 0.5:
            # Random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image


    def __data_generation(self, idx):
        """Generates data containing batch_size samples"""
        batch_files = self.all_files[idx*self.batch_size:idx*self.batch_size+self.batch_size]

        x_data = np.empty((self.batch_size,self.img_dims,self.img_dims,3))
        y_data = np.empty((self.batch_size,self.img_dims,self.img_dims,3))
        # read image
        for i, batch_file in enumerate(batch_files):

            img_x = cv2.imread(batch_file[0])
            img_y = cv2.imread(batch_file[1])

            img_x = tf.keras.applications.resnet50.preprocess_input(img_x)
            img_y = tf.keras.applications.resnet50.preprocess_input(img_y)

            x_data[i],y_data[i] = self.random_jitter(img_x, img_y)

        return x_data,y_data
