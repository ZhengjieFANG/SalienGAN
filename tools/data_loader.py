import os
import torch
import utils
import torchvision.transforms as T
from torch.utils import data
from PIL import Image

#定义自己的数据集合
class MyDataSet(data.Dataset):
    def __init__(self,root,transform):
        # 所有图片的绝对路径
        imgs=os.listdir(root)
        self.imgs=[os.path.join(root,k) for k in imgs]
        self.transform=transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        # 1. Load the image
        pil_img = Image.open(img_path)
        # 2. Resize and normalize the images using torchvision.
        img = self.transform(pil_img)
        return img

    def __len__(self):
        return len(self.imgs)

        #定义自己的数据集合
class DataSetWithSalieny(data.Dataset):
    def __init__(self,root,saliency_root,transform,transform_saliency):
        # 所有图片的绝对路径
        self.imgs=os.listdir(root)
        self.imgs=[os.path.join(root,k) for k in self.imgs]
        self.root = root
        self.saliency_root = saliency_root
        self.transform=transform
        self.transform_saliency = transform_saliency

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_name = self.imgs[index].split('/')[-1]
        saliency_name = img_name.split('.')[0]+".png"
        saliency_path = os.path.join(self.saliency_root, saliency_name)
        # 1. Load the image
        pil_img = Image.open(img_path)
        pil_saliency = Image.open(saliency_path)
        # 2. Resize and normalize the images using torchvision.
        img = self.transform(pil_img)
        saliency_1channel = self.transform_saliency(pil_saliency)
        saliency = utils.get_saleincy_2channel(saliency_1channel)
        return img, saliency

    def __len__(self):
        return len(self.imgs)


def get_saliency_dataloader(image_dir,saliency_dir, img_size, batch_size):
    compose = [
        T.Resize((img_size[0], img_size[1])),
        T.ToTensor(), #转到[0,1]
        T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) #从[0,1]转到[-1，1]
    ]
    transform = T.Compose(compose)

    compose_saliency = [
        T.Resize((img_size[0], img_size[1])),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(), #转到[0,1]
    ]
    transform_saliency = T.Compose(compose_saliency)

    dataset = DataSetWithSalieny(image_dir,saliency_dir,transform,transform_saliency)
    dataloader = iter(torch.utils.data.DataLoader(dataset,
                                                    batch_size,
                                                    num_workers = 1))
    return dataloader #返回的是一个dataloader的迭代器


def get_gray_dataloader(image_dir, img_size, batch_size):
    compose = [
        T.Resize((img_size[0], img_size[1])),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) #从[0,1]转到[-1，1]

    ]
    transform = T.Compose(compose)

    dataset = MyDataSet(image_dir,transform)
    dataloader = iter(torch.utils.data.DataLoader(dataset,
                                                    batch_size,
                                                    num_workers = 1))
    return dataloader #返回的是一个dataloader的迭代器


if __name__ == '__main__':
#     pil_saliency = Image.open(saliency_path)
#     # 2. Resize and normalize the images using torchvision.
#     img = self.transform(pil_img)
#     saliency_1channel = self.transform_saliency(pil_saliency)
#     saliency = get_saleincy_2channel(saliency_1channel)
#     return img, saliency
    # print(cartoon_gray)
    # print(cartoon_gray2)
    # print(cartoon_gray.size())
    # print(cartoon_gray.size())

    cartoon_loader = get_saliency_dataloader('../../dataset/Hayao/style', '../../dataset/Hayao/saliency', [256, 256], 1)
    cartoon_gray_loader = get_gray_dataloader('../../dataset/Hayao/style',  [256, 256], 1)

    for i in range(10):
        cartoon, cartoon_saliency = next(cartoon_loader)
        cartoon_gray = next(cartoon_gray_loader)
        cartoon_gray2 = utils.get_generated_gray(cartoon)


        cartoon = utils.save_transform(cartoon)
        cartoon_saliency = utils.save_transform(cartoon_saliency)
        cartoon_gray = utils.save_transform(cartoon_gray)
        cartoon_gray2 = utils.save_transform(cartoon_gray2)

        # cartoon = cartoon.convert('RGB')
        cartoon_saliency = cartoon_saliency.convert('RGB')
        # cartoon = cartoon.convert('RGB')
        # cartoon = cartoon.convert('RGB')

        cartoon.save( '../IOtest/cartoon{:03d}.jpg'.format(i))
        cartoon_saliency.save( '../IOtest/cartoon_saliency{:03d}.jpg'.format(i))
        cartoon_gray.save( '../IOtest/cartoon_gray{:03d}.jpg'.format(i))
        cartoon_gray2.save('../IOtest/cartoon_gray_generated{:03d}.jpg'.format(i))
        print(i)

