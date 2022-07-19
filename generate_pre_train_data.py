"""file for misc data util functions"""
import os
import argparse
from Data_Utils.data_utils import generate_pre_training_data

def parse_args():
    """the base argument parser"""
    desc = "script for generating disc pretraining data"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--base_data_dir', type=str,
        default="unet_data/", help='The directory that holds the image data')
    parser.add_argument('--input_data_dir',
        type=str, default="X/", help='The directory for CSV input data')
    parser.add_argument('--output_data_dir',
        type=str, default="Y/", help='The directory for CSV input data')
    parser.add_argument('--gen_pre_train_model_path_unfrozen',
        type=str, default="gen_pre_unfrozen.h5", help='Saved unfrozen pretrainedunet  model')
    parser.add_argument('--img_dim', type=int,
        default=256, help='The image dimensions')
    parser.add_argument('--data_size',
        type=int, default=-1, help='how much of the data are we using')
    parser.add_argument('--pre_train_fakes_dir',
        type=str, default="fakes_dir", help='saved final model')

    return parser.parse_args()


def main():
    """the main loop"""
    args = parse_args()
    fakes_dir = args.pre_train_fakes_dir

    if not os.path.exists(fakes_dir):
        os.makedirs(fakes_dir)

    generate_pre_training_data(args)


if __name__ == '__main__':
    main()
