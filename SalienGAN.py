
import time
import models

from glob import glob
from ops import *
from utils import *
from vgg19 import Vgg19

from tools.data_loader import *

import torch
import torchvision.transforms as T


class SalienGAN(object):
    def __init__(self, args, device):
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.data_mean = args.data_mean

        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.epoch = args.epoch
        self.init_epoch = args.init_epoch  #args.epoch // 20
        self.gan_type = args.gan_type
        self.training_rate = args.training_rate

        self.init_lr = args.init_lr
        self.d_lr = args.d_lr
        self.g_lr = args.g_lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        """ Weight """
        self.adv_g_weight = args.adv_g_weight
        self.adv_d_weight = args.adv_d_weight
        self.content_weight = args.content_weight
        self.texture_weight = args.texture_weight
        self.shading_weight = args.shading_weight

        """ Discriminator """
        self.n_dis = args.n_dis
        self.ch = args.ch
        self.sn = args.sn


        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        self.save_freq = args.save_freq

        check_folder(self.sample_dir)

        self.picture_loader = get_saliency_dataloader(
            '../dataset/train_photo','../dataset/train_saliency', self.img_size, self.batch_size)  #返回的是一个dataloader的迭代器
        self.cartoon_loader = get_saliency_dataloader(
            '../dataset/{}'.format(self.dataset + '/style'),'../dataset/{}'.format(self.dataset + '/saliency'), self.img_size, self.batch_size)
        self.cartoon_gray_loader = get_gray_dataloader(
            '../dataset/{}'.format(self.dataset + '/style'), self.img_size, self.batch_size)
        self.cartoon_blur_loader = get_gray_dataloader(
            '../dataset/{}'.format(self.dataset + '/blur'), self.img_size, self.batch_size)
        self.data_picture_num = len(get_image_paths_list('../dataset/train_photo'))          
        self.data_style_num = len(get_image_paths_list('../dataset/{}'.format(self.dataset + '/style')))
        self.data_num = max(self.data_picture_num, self.data_style_num)

        # Initialize Generator and Discriminator
        self.generator = models.Generator().to(device)
        self.discriminator = models.Discriminator(self.ch, self.n_dis, self.sn).to(device)

        self.vgg = Vgg19().to(device)
        self.vgg.eval()

        # Optimizer and lr_scheduler
        self.optimizer_init = torch.optim.Adam(
            self.generator.parameters(), self.init_lr, betas=(self.beta1, self.beta2))
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(), self.g_lr, betas=(self.beta1, self.beta2))
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), self.d_lr, betas=(self.beta1, self.beta2))

        print("##### GANsformer Initialization Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset)
        print("# picture dataset number : ", self.data_picture_num)
        print("# style dataset number : ", self.data_style_num)
        print("# max dataset number : ", self.data_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# init_epoch : ", self.init_epoch)
        print("# training image size [H, W] : ", self.img_size)
        print("# -------------------------------------------------")
        print("# adv_g_weight:", self.adv_g_weight)
        print("# adv_g_weight: ", self.adv_g_weight)
        print("# content_weight: ", self.content_weight)
        print("# texture_weight: ", self.texture_weight)
        print("# shading_weight: ", self.shading_weight)
        # print("# attentive_weight: ", self.attentive_weight)
        print("# -------------------------------------------------")
        print("# init_lr, g_lr, d_lr : ", self.init_lr,self.g_lr, self.d_lr)
        print(f"# training_rate G -- D: {self.training_rate} : 1")
        # print('---------- Networks Structure -----------------')
        # print_network(self.generator)
        # print_network(self.discriminator)
        # print_network(self.vgg)
        # print('-----------------------------------------------')

    def train(self, device):
        # restore check-point if it exits
        load_success, last_epoch = self.load(self.checkpoint_dir)
        if load_success:
            start_epoch = last_epoch + 1
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            print(" [!] Load failed...")

        # loop for epoch
        init_mean_loss = []
        g_mean_loss = []
        d_mean_loss = []

        j = self.training_rate

        self.generator.train()  # 开启训练模式
        self.discriminator.train()

        for epoch in range(start_epoch, self.epoch):

            for index in range(int(self.data_num / self.batch_size)):

                try:
                    picture, picture_saliency = next(self.picture_loader)
                    picture = picture.to(device)
                    picture_saliency = picture_saliency.to(device)
                except StopIteration:
                    self.picture_loader = get_saliency_dataloader('../dataset/train_photo', '../dataset/train_saliency',  self.img_size, self.batch_size)
                    picture, picture_saliency = next(self.picture_loader)
                    picture = picture.to(device)
                    picture_saliency = picture_saliency.to(device)

                try:
                    cartoon, cartoon_saliency = next(self.cartoon_loader)
                    cartoon = cartoon.to(device)
                    cartoon_saliency = cartoon_saliency.to(device)
                except StopIteration:
                    self.cartoon_loader = get_saliency_dataloader('../dataset/{}'.format(self.dataset + '/style'), '../dataset/{}'.format(self.dataset + '/saliency'), self.img_size, self.batch_size)
                    cartoon, cartoon_saliency = next(self.cartoon_loader)
                    cartoon = cartoon.to(device)
                    cartoon_saliency = cartoon_saliency.to(device)

                try:
                    cartoon_gray = next(self.cartoon_gray_loader).to(device)
                except StopIteration:
                    self.cartoon_gray_loader = get_gray_dataloader('../dataset/{}'.format(self.dataset + '/style'), self.img_size, self.batch_size) 
                    cartoon_gray = next(self.cartoon_gray_loader).to(device)
                
                try:
                    cartoon_blur = next(self.cartoon_blur_loader).to(device)
                except StopIteration:
                    self.cartoon_blur_loader = get_gray_dataloader('../dataset/{}'.format(self.dataset + '/blur'), self.img_size, self.batch_size)
                    cartoon_blur = next(self.cartoon_blur_loader).to(device)

                generated = self.generator(picture, picture_saliency)  #picture_saliency[batch, 2, h ,w]
                generated_detach = generated.clone().detach()

                if epoch < self.init_epoch:
                    start_time = time.time()
                    # Init G
                    self.optimizer_init.zero_grad()
                    init_content_loss = get_content_loss(self.vgg, picture, generated)
                    init_loss = self.content_weight * init_content_loss
                    init_loss.backward()
                    self.optimizer_init.step()

                    init_loss_cpu = init_loss.detach().cpu().numpy()
                    init_mean_loss.append(init_loss_cpu)
                    print("Epoch: %3d Step: %5d / %5d  time: %f s init_loss: %.8f  init_mean_loss: %.8f" % (epoch, index,
                          int(self.data_num / self.batch_size), time.time() - start_time, init_loss, np.mean(init_mean_loss)))
                    if (index+1) % 200 == 0:
                        init_mean_loss.clear()

                else:
                    # update D
                    if j == self.training_rate:
                        start_time = time.time()
                        self.optimizer_d.zero_grad()

                        cartoon_logit = self.discriminator(cartoon)
                        cartoon_gray_logit = self.discriminator(cartoon_gray)
                        cartoon_blur_logit = self.discriminator(cartoon_blur)
                        generated_logit_detach = self.discriminator(generated_detach)

                        cartoon_label = torch.ones(self.batch_size, 1, self.img_size[0] // 4, self.img_size[1] // 4).to(device)
                        fake_label = torch.zeros(self.batch_size, 1, self.img_size[0] // 4, self.img_size[1] // 4).to(device)

                        cartoon_loss, fake_loss, adv_color_loss, adv_edge_loss = discriminator_loss(
                            self.gan_type, cartoon_logit, cartoon_gray_logit,cartoon_blur_logit, generated_logit_detach, cartoon_label, fake_label)
                        d_loss = self.adv_d_weight * (1.7 * cartoon_loss + 1.7 * fake_loss + 1.7 * adv_color_loss  +  0.8 * adv_edge_loss)

                        d_loss.backward()
                        self.optimizer_d.step()

                        d_loss_cpu = d_loss.detach().cpu().numpy()
                        d_mean_loss.append(d_loss_cpu)
                        print("Epoch: %3d Step: %5d / %5d time: %f s, cartoon_loss: %.5f  fake_loss: %.5f  color_loss: %.8f  edge_loss: %.5f \n"
                              "d_loss: %.5f  d_mean_loss: %.5f\n" % (
                            epoch, index, int(self.data_num / self.batch_size), time.time() - start_time,cartoon_loss, fake_loss, adv_color_loss, adv_edge_loss, d_loss, np.mean(d_mean_loss)))
                        if (index + 1) % 200 == 0:
                            d_mean_loss.clear()

                    # update G
                    start_time = time.time()
                    self.optimizer_g.zero_grad()

                    content_loss = get_content_loss(self.vgg, picture, generated)
                    color_loss = get_color_loss(picture, generated, device)
                    texture_loss = 0.33*get_texture_loss_guided(self.vgg, cartoon_gray, generated, cartoon_saliency, picture_saliency) + \
                        0.67*get_texture_loss(self.vgg, cartoon_gray, generated)
                    t_loss = self.content_weight * content_loss + self.texture_weight * texture_loss+self.shading_weight*color_loss
                    
                    generated_logit = self.discriminator(generated)
                    cartoon_label = torch.ones(self.batch_size, 1, self.img_size[0] // 4, self.img_size[1] // 4).to(device)
                    g_loss = self.adv_g_weight * generator_loss(self.gan_type, generated_logit, cartoon_label) + t_loss

                    g_loss.backward()
                    self.optimizer_g.step()

                    g_loss_cpu = g_loss.detach().cpu().numpy()
                    g_mean_loss.append(g_loss_cpu)
                    print("Epoch: %3d Step: %5d / %5d time: %f s, content_loss: %.5f  color_loss: %.5f  texture_loss: %.5f                 \n"
                          "g_loss: %.5f  g_mean_loss: %.5f\n" % (
                        epoch, index, int(self.data_num / self.batch_size), time.time() - start_time, content_loss, color_loss, texture_loss, g_loss, np.mean(g_mean_loss)))
                    if (index + 1) % 200 == 0:
                        g_mean_loss.clear()

                    j = j - 1
                    if j < 1:
                        j = self.training_rate

            if (epoch + 1) >= self.init_epoch and np.mod(epoch + 1, self.save_freq) == 0:
                with torch.no_grad():
                    # Save the checkpoints.
                    self.save_checkpoint(epoch)
                    
            if (epoch + 1) >= self.init_epoch:
                with torch.no_grad():
                    self.generator.eval()  # 开启验证模式，会停止使用dropout, batch_norme等操作
                    self.save_sample(epoch, device)
                    self.generator.train()  # 回到训练模式

    # Python内置的@property装饰器负责把一个方法变成属性调用

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}_{}_{}_{}".format(self.model_name,self.dataset,
                                                   self.gan_type,
                                                   float(self.adv_g_weight), float(self.adv_d_weight), float(self.content_weight), float(self.texture_weight), float(self.shading_weight))

    def save_sample(self, epoch, device):
        val_files = glob('../dataset/{}/*.*'.format('val'))
        save_path = './{}/{:03d}/'.format(self.sample_dir, epoch)
        check_folder(save_path)

        for i, image_path in enumerate(val_files):
            print('val: ' + str(i) + image_path)
            image_name = image_path.split('/')[-1]
            sliency_name = image_name.split('.')[0] + '.png'
            saliency_path = '../dataset/{}/'.format('val_saliency')+sliency_name

            sample_image, sample_saliency = load_test_data(image_path, saliency_path, self.img_size)
            sample_image = sample_image.to(device)
            sample_saliency = sample_saliency.to(device)
            test_generated = self.generator(sample_image, sample_saliency).detach()

            sample_image_cpu = sample_image.cpu()
            test_generated_cpu = test_generated.cpu()

            sample_image = save_transform(sample_image_cpu)
            test_generated=save_transform(test_generated_cpu)

            sample_image.save(save_path+'/{:03d}_picture.jpg'.format(i))
            test_generated.save(save_path+'/{:03d}_generated.jpg'.format(i))
            print("Save sample" + str(i)+" to "+save_path)

    def save_sample_test(self, device):
        val_files = glob('../dataset/{}/*.*'.format('val'))
        save_path = './samples/test'
        check_folder(save_path)

        for i, sample_path in enumerate(val_files):
            print('val: ' + str(i) + sample_path)
            sample_image = load_test_data(sample_path, self.img_size).to(device)

            sample_image = save_transform(sample_image)

            sample_image.save(save_path+'/picture{:03d}.jpg'.format(i))

            print("Save sample" + str(i)+" to "+save_path)

    def save_checkpoint(self, epoch):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir, 'epoch-{:03d}'.format(epoch))
        check_folder(checkpoint_dir)

        torch.save(self.generator.state_dict(),checkpoint_dir+'/Generator.pth')
        #torch.save(self.generator_ema.state_dict(),checkpoint_dir+'/Generator_ema.pth')
        torch.save(self.discriminator.state_dict(),checkpoint_dir+'/Discriminator.pth')
        print("Save model state of epoch " + str(epoch))

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        if os.path.exists(checkpoint_dir):
            dirs_list = os.listdir(checkpoint_dir)
            if len(dirs_list) > 0:
                last_checkpoint = dirs_list[-1]
                epoch = int(last_checkpoint.split('-')[-1])
                last_checkpoint_dir = os.path.join(
                    checkpoint_dir, last_checkpoint)
                self.generator.load_state_dict(torch.load(
                    last_checkpoint_dir+'/Generator.pth'))
                #self.generator_ema.load_state_dict(torch.load(
                    #last_checkpoint_dir+'/Generator_ema.pth'))
                self.discriminator.load_state_dict(torch.load(
                    last_checkpoint_dir+'/Discriminator.pth'))
                print(" [*] Success to read {}".format(last_checkpoint_dir))
                return True, epoch

        print(" [*] Failed to find a checkpoint")
        return False, 0

    def test_vgg(self,device):
        picture = next(self.picture_loader).to(device)
        fm = self.vgg(picture)
        max = torch.max(fm)
        min = torch.min(fm)
        mean = torch.mean(fm)
        print(fm)
        print(max)
        print(min)
        print(mean)


