import os
import argparse

from utils import *
from SalienGAN import SalienGAN
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',type=str,default='train',help="train or test?")
    parser.add_argument('--model_name',type=str,default='SalienGAN',help="model name")
    parser.add_argument('--dataset',type=str,default='Hayao',help="dataset_name")
    parser.add_argument("--batch_size", type = int, default = 2,help = "Size of each batches (Default: 128)")
    parser.add_argument('--data_mean',type=list,default=[7.6287,-3.3273,-4.3014],help='data_mean(bgr) from data_mean.py')
    parser.add_argument('--epoch', type=int, default=101, help='The number of epochs to run')
    parser.add_argument('--init_epoch', type=int, default=8, help='The number of epochs for weight initialization')
    parser.add_argument('--save_freq', type=int, default=1, help='The number of ckpt_save_freq')

    parser.add_argument('--init_lr', type=float, default=2e-4, help='The learning rate')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='The learning rate')
    parser.add_argument('--d_lr', type=float, default=0.0004, help='The learning rate')
    parser.add_argument("--beta1", type = float, default = 0.5,help = "Coefficients used for computing running averages of gradient and its square")
    parser.add_argument("--beta2", type = float, default = 0.99,help = "Coefficients used for computing running averages of gradient and its square")

    parser.add_argument('--adv_g_weight', type=float, default=300.0, help='Weight of generator about GAN')
    parser.add_argument('--adv_d_weight', type=float, default=300.0, help='Weight of generator about GAN')  #
    parser.add_argument('--content_weight', type=float, default=1.5, help='Weight about VGG19') # 1.1 for Shinkai
    parser.add_argument('--texture_weight', type=float, default=1.0, help='Weight about texture')
    parser.add_argument('--shading_weight', type=float, default=10.0, help='Weight about picture color and shadiing')

    parser.add_argument('--img_size', type=list, default=[256,256], help='The size of image: H and W')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_dis', type=int, default=3, help='The number of discriminator layer')
    parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')

    parser.add_argument('--training_rate', type=int, default=1, help='training rate about G & D') #？？
    parser.add_argument('--gan_type', type=str, default='lsgan', help='[gan / lsgan / wgan-gp / wgan-lp / dragan / hinge]')

    parser.add_argument("--gpu_id", type = int, default = 0 ,help = "Select the specific gpu to training")

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',help='Directory name to save the samples on training')
    return check_args(parser.parse_args())

def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)
    if args.phase == 'test':
        # --result_dir
        check_folder(args.result_dir)

    # --log_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


if __name__ == '__main__':

    args = parse_args()

    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Generator and Discriminator
    myGAN = SalienGAN(args, device)

    # Start Training
    myGAN.train(device)
