import torch
import torch.nn as nn
import torch.nn.functional as F
from vgg19 import *

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    m = nn.LeakyReLU(alpha)
    return m(x)

def relu(x):
    m = nn.ReLU()
    return m(x)

def tanh(x):
    m = nn.Tanh()
    return m(x)

def sigmoid(x) :
    m = nn.Sigmoid()
    return m(x)

##################################################################################
# Loss function
##################################################################################

def L1_loss(x, y):
    L1_loss = nn.L1Loss()
    loss = L1_loss(x, y)
    return loss

def L2_loss(x,y):
    L2_loss = nn.MSELoss()
    loss = L2_loss(x, y)
    return loss

def Huber_loss(x,y):
    # h = nn.HuberLoss()
    h = nn.SmoothL1Loss()
    return h(x,y)

def discriminator_loss(loss_func, cartoon_logit, cartoon_gray_logit,cartoon_blur_logit, fake_logit, cartoon_label, fake_label):
    cartoon_loss = 0
    adv_color_loss = 0       #adv_color_loss
    fake_loss = 0
    adv_edge_loss = 0  #adv_edge_loss

    if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
        cartoon_loss = -torch.mean(cartoon_logit)
        adv_color_loss = torch.mean(cartoon_gray_logit)
        fake_loss = torch.mean(fake_logit)
        adv_edge_loss = torch.mean(cartoon_blur_logit)

    if loss_func == 'lsgan' :
        cartoon_loss = torch.mean((cartoon_logit - 1.0)*(cartoon_logit - 1.0))
        adv_color_loss = torch.mean(cartoon_gray_logit*cartoon_gray_logit)
        fake_loss = torch.mean(fake_logit*fake_logit)
        adv_edge_loss = torch.mean(cartoon_blur_logit*cartoon_blur_logit)
        # cartoon_loss = torch.mean(torch.square(cartoon_logit - 1.0))
        # adv_color_loss = torch.mean(torch.square(cartoon_gray_logit))
        # fake_loss = torch.mean(torch.square(fake_logit))
        # adv_edge_loss = torch.mean(torch.square(cartoon_blur_logit))

    if loss_func == 'gan' or loss_func == 'dragan' :
        BCELOSS = nn.BCELoss() #BCELoss() reduction默认为mean，已经会对batch_size个数做平均值
        cartoon_loss = BCELOSS(cartoon_logit, cartoon_label)
        adv_color_loss = BCELOSS(cartoon_gray_logit, fake_label)
        adv_edge_loss = BCELOSS(cartoon_blur_logit, fake_label)
        fake_loss = BCELOSS(fake_logit, fake_label)

    if loss_func == 'hinge':
        cartoon_loss = torch.mean(relu(1.0 - cartoon_label))
        fake_loss = torch.mean(relu(1.0 + fake_logit))
        adv_color_loss = torch.mean(relu(1.0 + cartoon_gray_logit))
        adv_edge_loss = torch.mean(relu(1.0 + cartoon_blur_logit))

    #loss = 1.7 * cartoon_loss + 1.7 * fake_loss + 1.7 * adv_color_loss  +  0.8 * adv_edge_loss

    return cartoon_loss,fake_loss, adv_color_loss,adv_edge_loss


def generator_loss(loss_func, fake_logit, cartoon_label):
    fake_loss = 0

    if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
        fake_loss = -torch.mean(fake_logit)

    if loss_func == 'lsgan' :
        fake_loss = torch.mean((fake_logit - 1.0)*(fake_logit - 1.0))

    if loss_func == 'gan' or loss_func == 'dragan':
        BCELOSS = nn.BCELoss()
        fake_loss = BCELOSS(fake_logit, cartoon_label)

    if loss_func == 'hinge':
        fake_loss = -torch.mean(fake_logit)

    return fake_loss


def gram_matrix(x):
    b, c, h, w = x.size()
    x = x.view(b, c, h*w)
    x_t = x.transpose(1, 2)
    gram = torch.bmm(x,x_t) / float(x.numel()//b)
    return gram


def gram_matrix_guided(x, guides):
    '''
    :param x: feature map [b, n_fm, h, w]
    :param guides: saliency [b, 2, h, w]
    '''
    assert guides.dtype == torch.float32, 'Guides should be float'
    batch, n_fm, h_fm ,w_fm = x.size()
    batch, n_guide, h_guide,w_guide = guides.size()[0], guides.size()[1], guides.size()[2], guides.size()[3]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = torch.zeros((batch, n_guide,n_fm,n_fm)).to(device)
    for b in range(batch):
        for c in range(n_guide):
            F = (x[b,:,:,:] * guides[b,c,:,:]).reshape(n_fm,-1) # n_fm * (h_fm * w_fm)
            G[b,c,:,:] = torch.mm(F,F.t()) / float(x.numel()//batch)

    return G  #[batch, n_guide, n_fm, n_fm]


def get_content_loss(vgg, picture, fake):
    picture_feature_map = vgg(picture)
    fake_feature_map = vgg(fake)

    loss = L1_loss(picture_feature_map,fake_feature_map)
    return loss


def style_loss(style, fake):
    return L1_loss(gram_matrix(style), gram_matrix(fake))


def style_loss_guided(style, fake, sal_cartoon, sal_fake):
    n_guide = sal_cartoon.size()[1]
    loss = torch.Tensor([0.0]).float().cuda()
    for i in range(n_guide):
        loss += L1_loss(gram_matrix_guided(style,sal_cartoon)[:,i,:,:], gram_matrix_guided(fake,sal_fake)[:,i,:,:])
    return loss

def style_loss_insaliency(style, fake, sal_cartoon):
    return L1_loss(gram_matrix_guided(style,sal_cartoon)[:,1,:,:], gram_matrix(fake))


def get_texture_loss(vgg, cartoon_gray, fake):
    cartoon_feature_map = vgg(cartoon_gray)
    fake_feature_map = vgg(fake)

    texture_loss = style_loss(cartoon_feature_map, fake_feature_map)
    return texture_loss


def get_texture_loss_guided(vgg, cartoon_gray, fake, cartoon_saliency, fake_saliency): #sliency中包含了显著和不显著两个通道的信息，都是[0,1]
    cartoon_feature_map = vgg(cartoon_gray)
    fake_feature_map = vgg(fake)
    cartoon_fm_size = cartoon_feature_map.size()[-2:]
    fake_fm_size = fake_feature_map.size()[-2:]
    cartoon_saliency_fm = get_fm_saliency(cartoon_saliency, cartoon_fm_size, caffe_model=vgg, th=.4)
    fake_saliency_fm = get_fm_saliency(fake_saliency, fake_fm_size, caffe_model=vgg, th=.4)

    texture_loss = style_loss_guided(cartoon_feature_map, fake_feature_map, cartoon_saliency_fm, fake_saliency_fm)
    return texture_loss

def get_texture_loss_insaliency(vgg, cartoon_gray, fake, cartoon_saliency): #sliency中包含了显著和不显著两个通道的信息，都是[0,1]
    cartoon_feature_map = vgg(cartoon_gray)
    fake_feature_map = vgg(fake)
    cartoon_fm_size = cartoon_feature_map.size()[-2:]
    cartoon_saliency_fm = get_fm_saliency(cartoon_saliency, cartoon_fm_size, caffe_model=vgg, mode=['simple','simple'], th=.4)

    texture_loss = style_loss_insaliency(cartoon_feature_map, fake_feature_map, cartoon_saliency_fm)
    return texture_loss

def get_color_loss(picture, fake, device):
    picture = rgb2yuv(picture, device)
    fake = rgb2yuv(fake, device)

    return L1_loss(picture[:,0,:,:], fake[:,0,:,:]) + Huber_loss(picture[:,1,:,:],fake[:,1,:,:]) + Huber_loss(picture[:,2,:,:],fake[:,2,:,:])
    #return L1_loss(picture[:,0,:,:], fake[:,0,:,:]) + L1_loss(picture[:,1,:,:],fake[:,1,:,:]) + L1_loss(picture[:,2,:,:],fake[:,2,:,:])


def get_shading_loss(content_adjusted, fake, device):
    content_adjusted = rgb2yuv(content_adjusted, device)
    fake = rgb2yuv(fake, device)

    return L1_loss(content_adjusted[:,0,:,:], fake[:,0,:,:]) + Huber_loss(content_adjusted[:,1,:,:],fake[:,1,:,:]) + Huber_loss(content_adjusted[:,2,:,:],fake[:,2,:,:])
    #return L1_loss(content_adjusted[:,0,:,:], fake[:,0,:,:]) + L1_loss(content_adjusted[:,1,:,:],fake[:,1,:,:]) + L1_loss(content_adjusted[:,2,:,:],fake[:,2,:,:])


def rgb2yuv(rgb, device):
    """
    Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
    """
    rgb = ((rgb + 1) / 2) * 255.0
    rgb_ = rgb.transpose(1, 3)  # input is 3*n*n   default
    A = torch.tensor([[0.299, -0.14714119, 0.61497538],
                      [0.587, -0.28886916, -0.51496512],
                      [0.114, 0.43601035, -0.10001026]]).to(device)  # from  Wikipedia
    yuv = torch.tensordot(rgb_, A, 1).transpose(1, 3)
    yuv = yuv / 255.0
    return yuv


def get_fm_saliency(saleincy, out_size, caffe_model=None, mode=['all','inside'], th=.4):
    '''
    :param saleincy: the pixel guides [b,2,h,w]
    :param caffe_model: the network model to compute the guides for
    :param layers: the layers on which to compute the guides
    :param mode: the mode of obtaining the feature map guide: simple|all|inside, downsampling|all neurons that see region| neurons that see only region
    :param th: threshold to make guides binary, only used for modes all|inside
    :param batch_size: batch_size for probing which neurons see the guide
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, n_saleincy, _, _ = saleincy.size()
    fm_saleincy = torch.zeros((batch_size, n_saleincy, out_size[0], out_size[1])).to(device)
    saleincy[saleincy < th] = 0.0
    saleincy[saleincy >= th] = 1.0
    for b in range(batch_size):
        for m in range(n_saleincy):
            guide = saleincy[b, m, :, :]

            if mode[m] == 'simple' or caffe_model is None:
                fm_saleincy[b, m, :, :] = F.interpolate(saleincy, size=out_size, mode="nearest")[b, m, :, :]

            if mode[m] == 'all':

                probe_image = torch.rand((2, 3, guide.size()[0], guide.size()[1])).to(device)*100
                probe_image *= guide
                feature_maps = caffe_model(probe_image)
                fm_saleincy[b, m, :, :] = (feature_maps.var(0).mean(0) != 0).float()

            elif mode[m] == 'inside':
                inv_guide = guide.clone()-1.0
                inv_guide *= -1.0
                probe_image_out = torch.rand((2, 3, inv_guide.size()[0], inv_guide.size()[1])).to(device)*100
                probe_image_out *= inv_guide
                feature_maps_out = caffe_model(probe_image_out)

                fm_saleincy[b, m, :, :] = (feature_maps_out.var(0).mean(0) == 0).float()

    return fm_saleincy  #[batch_size, 2, h, w]


if __name__ == '__main__':
    # vgg = Vgg19()
    # a1 = torch.zeros((1, 1, 128, 256))
    # a2 = torch.ones((1, 1, 128, 256))
    #
    # d1 = torch.ones((1,1,128,256))
    # d2 = torch.zeros((1,1,128,256))
    #
    # a = torch.cat((a1,a2),2)
    # d = torch.cat((d1,d2),2)
    # s = torch.cat((a,d),1)
    #
    # s_fm = get_fm_saliency(s,[32,32],caffe_model=vgg, mode=['all', 'inside'])
    #
    # print(s_fm)
    # print(s_fm.size())

    # a1 = torch.rand((2, 3, 256, 256))*256
    # guide = torch.cat((torch.zeros((128,256)),torch.ones((128,256))),0)
    # a1*=guide
    #
    # s_fm = vgg(a1)
    # var = s_fm.var(0).mean(0)
    # print(var)
    # print(var.size())
    # c = (s_fm.var(0).mean(0)==0.0).float()
    # print(c)
    # print(c.size())
    # print(s_fm[0,0,:,:])
    # print(s_fm[1,0,:,:])
    # # print(s_fm)
    # print(s_fm.size())


    # a1 = torch.zeros((1, 1, 1, 3))
    # a2 = torch.ones((1, 1, 2, 3))
    #
    # b1 = torch.ones((1, 1, 1, 3))*2
    # b2 = torch.zeros((1, 1, 2, 3))
    #
    # a = torch.cat((a1,a2),2)
    # b = torch.cat((b1,b2),2)
    #
    # c = torch.ones(1,1,3,3)
    #
    # x = torch.cat((a,b),1)
    # x=  torch.cat((x,c),1) # [1,3,3,3]
    #
    # e1 = torch.zeros((1, 1, 2, 3))
    # e2 = torch.ones((1, 1, 1, 3))
    #
    # d1 = torch.ones((1,1,2,3))
    # d2 = torch.zeros((1,1,1,3))
    #
    # e = torch.cat((e1,e2),2)
    # d = torch.cat((d1,d2),2)
    # guide = torch.cat((e,d),1)
    #
    # print(x)
    # print(guide)
    #
    # s_fm = gram_matrix_guided(x, guide)
    # #
    # print(s_fm)
    # print(s_fm.size())
    # a = torch.rand((2, 3, 4, 4))
    # b = a[:,0,:,:]
    # c = a[0,:,:]
    # print(a)
    # print(b)
    # print(c)
    # print(a.size())
    # print(b.size())
    # print(c.size())
    a = torch.rand((1, 2, 4, 4))
    b = torch.rand((1, 2, 4, 4))
    l1 = L1_loss(a,b)
    l2 = L1_loss(a[:,0,:,:],b[:,0,:,:])+L1_loss(a[:,1,:,:],b[:,1,:,:])
    print(l1)
    print(l2/2)

