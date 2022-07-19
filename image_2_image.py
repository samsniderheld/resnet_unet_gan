"""
Base script for definining parameters,setting up directories
and running experiments
"""
import argparse
import os
from datetime import datetime
from Training.training import pre_train_unet, pre_train_discriminator, train_unet_gan





def parse_args():
    """the base argument parser"""
    desc = "A multi step pretraining gan for image 2 image translation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--base_data_dir', type=str,
        default="unet_data/", help='The directory that holds the image data')
    parser.add_argument('--input_data_dir',
        type=str, default="X/", help='The directory for CSV input data')
    parser.add_argument('--output_data_dir',
        type=str, default="Y/", help='The directory for CSV input data')
    parser.add_argument('--base_results_dir',
        type=str, default="/", help='The base directory to hold the results')
    parser.add_argument('--saved_model_dir',
        type=str, default="Saved_Models/", help='The directory for input data')
    parser.add_argument('--history_dir',
        type=str, default="History/", help='The directory for input data')
    parser.add_argument('--gan', action='store_true',
        help='train gan or not')
    parser.add_argument('--img_dim', type=int,
        default=256, help='The image dimensions')
    parser.add_argument('--num_epochs', type=int,
        default=100, help='The number of epochs')
    parser.add_argument('--batch_size',
        type=int, default=128, help='The size of batch')
    parser.add_argument('--print_freq',
        type=int, default=5, help='How often is the status printed')
    parser.add_argument('--save_freq', type=int,
        default=10, help='How often is the model saved')
    parser.add_argument('--save_best_only',
        action='store_true')
    parser.add_argument('--continue_training',
        action='store_true')
    parser.add_argument('--notes', type=str,
     default="N/A", help='A description of the experiment')
    parser.add_argument('--experiment_name',
        type=str, default="", help='A name for the experiment')
    parser.add_argument('--data_size',
        type=int, default=-1, help='how much of the data are we using')
    parser.add_argument('--gen_pre_train_model_path_frozen',
        type=str, default="gen_pre_frozen.h5", help='Saved frozen pretrained unet model')
    parser.add_argument('--gen_pre_train_model_path_unfrozen',
        type=str, default="gen_pre_unfrozen.h5", help='Saved unfrozen pretrainedunet  model')
    parser.add_argument('--disc_pre_train_model_path',
        type=str, default="disc_pre.h5", help='saved disc pretrained model')
    parser.add_argument('--final_model_path',
        type=str, default="gen_final.h5", help='saved final model')
    parser.add_argument('--critic_thresh',
        type=float, default=.65, help='How much do we let the disc catch up')
    parser.add_argument('--lr',
        type=float, default=1e-4, help='The learning rate')
    parser.add_argument('--save_img_path',
        type=str, default="Images", help='saved final model')
    parser.add_argument('--disc_pre_train_epochs',
        type=int, default=10, help='num epochs to pretrain discriminator')
    parser.add_argument('--gen_pre_train_epochs',
        type=int, default=4, help='num epochs to pretrain generator')
    parser.add_argument('--gan_epochs',
        type=int, default=10, help='num epochs to gan')
    parser.add_argument('--mode', type=str, default="gan",
        choices=['disc', 'gen', 'gan'], help="which training mode")

    return parser.parse_args()


def main():
    """the main loop"""
    args = parse_args()

    args.experiment_name = datetime.now().strftime("%Y_%m_%d_%H_%M") + "_" + args.experiment_name

    args.base_results_dir = os.path.join(args.base_results_dir,args.experiment_name)

    if not os.path.exists(args.base_results_dir):
        os.makedirs(args.base_results_dir)
        os.makedirs(os.path.join(args.base_results_dir,"Images"))
        os.makedirs(os.path.join(args.base_results_dir,"History"))
        os.makedirs(os.path.join(args.base_results_dir,"Saved_Models"))


    if args.mode == "gen":
        pre_train_unet(args)
    elif args.mode == "disc":
        pre_train_discriminator(args)
    elif args.mode == "gan":
        train_unet_gan(args)



if __name__ == '__main__':
    main()
