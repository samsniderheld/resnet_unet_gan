import argparse
import os
import shutil
from Training.training import *
from Data_Utils.normalize import normalize_data, normalize_image_data
from Model.pix_2_pix_gan import *
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import cv2
import tensorflow as tf
import numpy as np


def parse_args():
    desc = "An autoencoder for pose similarity detection"

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--base_data_dir', type=str, default="datasets/", help='The directory that holds the image data')
    parser.add_argument('--input_data_dir', type=str, default="full_frame/", help='The directory for CSV input data')
    parser.add_argument('--base_results_dir', type=str, default="/", help='The base directory to hold the results')
    parser.add_argument('--saved_model_dir', type=str, default="Saved_Models/", help='The directory for input data')
    parser.add_argument('--saved_model_name', type=str, default="399999_pix_2pix_model.h5", help='The directory for input data')
    parser.add_argument('--img_dim', type=int, default=256, help='The image dimensions')
    parser.add_argument('--output_dim_x', type=int, default=1280, help='Output Img dimensions')
    parser.add_argument('--output_dim_y', type=int, default=720, help='Output Img dimensions')
    
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = os.path.join(args.base_data_dir,args.input_data_dir + "*")
    
    target_imgs = sorted(glob.glob(input_dir),key=natural_keys)

    generator = Generator()

    saved_model_path = os.path.join(args.saved_model_dir,args.saved_model_name)

    generator.load_weights(saved_model_path)

    generate_video(args,generator,target_imgs)

    print("video generated")

def generate_video(args,model,img_paths):

  output_frames = []

  for i,file_name in tqdm(enumerate(img_paths)):
    X = tf.io.read_file(file_name)
    X = tf.io.decode_jpeg(X)
    X = tf.image.resize(X,(256,256))
    X = X/255
    X = tf.cast(X,tf.float32)
    model_input = np.expand_dims(X,0)
    output = model(model_input, training=True)
    output_img = output[0].numpy() * 255
    # cv2.imwrite(f"{i:04d}_image.jpg", output_img.astype(np.uint8))
    # output_frames.append(output_img.astype(np.uint8))
    # final_out = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    
    output_resized = cv2.resize(output_img,(args.output_dim_x,args.output_dim_y),interpolation = cv2.INTER_NEAREST)
    output_frames.append(output_resized.astype(np.uint8))

  fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

  out_path = os.path.join(args.base_results_dir, f"video.mp4")
  writer = cv2.VideoWriter(out_path, fourcc, 24, (args.output_dim_x,args.output_dim_y))
  # writer = cv2.VideoWriter(out_path, fourcc, 24, (256,256))


  for frame in output_frames:
      writer.write(frame)

  writer.release() 

# os.system(f"cp -r {out_path} {video_save_path}")


if __name__ == '__main__':
    main()
