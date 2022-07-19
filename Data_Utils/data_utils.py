"""file for misc data util functions"""
import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from Model.unet import build_resnet50_unet

def resize_single(input_image, height, width):
    """resize image"""
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image


def random_crop_single(input_image, width, height):
    """random crop img"""
    cropped_image = tf.image.random_crop(
        input_image, size=[1, width, height, 3])

    return cropped_image[0]

def random_jitter_single(input_image, width, height):
    """resize and randomly recrop"""
    # Resizing to 286x286
    input_image = resize_single(input_image, width+30,height+30)

    # Random cropping back to 256x256
    input_image = random_crop_single(input_image, width, height)

    if tf.random.uniform(()) > 0.5:
      # Random mirroring
      input_image = tf.image.flip_left_right(input_image)

    return input_image

def generate_pre_training_data(args):
    """function to generate all disc pretrain data"""
    input_shape = (args.img_dim,args.img_dim, 3)

    generator = build_resnet50_unet(input_shape)


    generator.load_weights(args.gen_pre_train_model_path_unfrozen)



    input_dir = os.path.join(args.base_data_dir,args.input_data_dir,"*")

    input_files  = sorted(glob.glob(input_dir))[:args.data_size]

    for i, file in tqdm(enumerate(input_files)):
        img_x = cv2.imread(file)

        img_x = tf.keras.applications.resnet50.preprocess_input(img_x)

        img_x = np.expand_dims(img_x,0)

        img_x = random_jitter_single(img_x,args.img_dim,args.img_dim)

        img_x = np.expand_dims(img_x,0)

        prediction = generator(img_x)

        out = prediction[0].numpy()

        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB )

        output_path = os.path.join(args.pre_train_fakes_dir,f"{i:04d}_image.jpg")

        cv2.imwrite(output_path,out)
