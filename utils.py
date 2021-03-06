import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torch.nn as nn

def load_test_data(image_path, saliency_path, img_size):
    img = Image.open(image_path)
    sal = Image.open(saliency_path)

    w, h = img.size #pil 读取出来是w × h
    if h <= img_size[0]:
        h = img_size[0]
    else:
        x = h % 32
        h = h - x

    if w < img_size[1]:
        w = img_size[1]
    else:
        y = w % 32
        w = w - y

    img = img.resize((w, h), Image.BILINEAR)
    sal = sal.resize((w, h), Image.BILINEAR)
    img_to_tensor = T.Compose([
        T.ToTensor(), #会自动除以255
        T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) #从[0,1]转到[-1，1]
    ])
    sal_to_tensor = T.Compose([
        T.ToTensor(), #会自动除以255转到[0，1]
    ])
    img = img_to_tensor(img)
    sal_1channel = sal_to_tensor(sal)

    sal_2channel = get_saleincy_2channel(sal_1channel)

    img = torch.unsqueeze(img, 0) #给最高位添加一个维度，也就是batch_size的大小
    sal = torch.unsqueeze(sal_2channel, 0)
    return img, sal

def save_transform(img):
    img = torch.squeeze(img)
    img = (img + 1.) / 2 
    img = T.ToPILImage()(img)#会自动乘以255
    return img

def get_saleincy_2channel(saliency):
    insliency = (saliency-1.)*(-1)
    return torch.cat((saliency, insliency),0)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')

def get_image_paths_list(image_dir):
    image_dir = os.path.join(image_dir)
    paths = []
    for path in os.listdir(image_dir):
        #os.listdir(image_dir)返回image_dir内的文件或文件夹列表
        # Check extensions of filename
        if path.split('.')[-1] not in ['jpg', 'jpeg', 'png', 'gif']:
            continue
        # Construct complete path to cartoon image
        path_full = os.path.join(image_dir, path)
        # Validate if colorized image exists
        if not os.path.isfile(path_full):
            continue
        paths.append(path_full)
    return paths

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()

if __name__ == '__main__':
    val_file = '../dataset/val/067.jpg'
    val_sal_file = '../dataset/val_saliency/067.png'
    # sample_image, sample_sal = load_test_data(val_file,val_sal_file,[256,256])
    sample_sal = Image.open(val_sal_file)
    # sample_sal = np.array(sample_sal)
    sal_to_tensor = T.Compose([
        # T.Resize((256, 256)),
        T.ToTensor(), #会自动除以255转到[0，1]
    ])
    sample_sal = sal_to_tensor(sample_sal)
    print(sample_sal)
    # print(sample_image)
    print(sample_sal.size())
    # sample_image=torch.squeeze(sample_image)
    # test_generated = T.ToPILImage()(sample_image)
    # test_generated.save('../dataset/sample1.jpg')
