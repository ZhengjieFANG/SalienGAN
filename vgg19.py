import torch
import torch.nn as nn

VGG_MEAN = [103.939, 116.779, 123.68]

# class Vgg19(nn.Module):
#     def __init__(self):
#         super(Vgg19, self).__init__()
#         self.model = models.vgg19(pretrained=True)
#         self.conv4_4 = self.model.features[27]
#
#     def forward(self, rgb):
#         rgb_scaled = ((rgb + 1) / 2) * 255.0  #[-1, 1] -> [0, 255]
#
#         red, green, blue = torch.split(rgb_scaled, 1, 1) #troch.split(t, split_size, dim) 在dim维按照每份split_size大小切分t
#         bgr = torch.cat((blue - VGG_MEAN[0],
#                         green - VGG_MEAN[1],
#                         red - VGG_MEAN[2]), 1)
#         return self.conv4_4(bgr)


class Vgg19(nn.Module):
    def __init__(self, init_weights="vgg19_weight/vgg19.pth", feature_mode=True, batch_norm=False, num_classes=1000):
        super(Vgg19, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        self.init_weights = init_weights
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm
        self.num_clases = num_classes
        self.features = self.make_layers(self.cfg, batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if not init_weights == None:
            self.load_state_dict(torch.load(init_weights))


    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, rgb):
        rgb_scaled = ((rgb + 1) / 2) * 255.0  #[-1, 1] -> [0, 255]
        red, green, blue = torch.split(rgb_scaled, 1, 1) #troch.split(t, split_size, dim) 在dim维按照每份split_size大小切分t

        x = torch.cat((red - VGG_MEAN[2],
                       green - VGG_MEAN[1],
                       blue - VGG_MEAN[0]), 1)

        if self.feature_mode:
            module_list = list(self.features.modules())
            for l in module_list[1:27]:                 # conv4_4
                x = l(x)
        if not self.feature_mode:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        return x


if __name__ == '__main__':
    vgg = Vgg19()
    a = torch.ones((1, 3, 256, 256))

    # a = torch.ones((1, 3, 256, 256))*(0.59)
    # c1 = torch.ones((1, 1, 256, 256))*(0.5)
    # d1 = torch.ones((1,2,128,256))*7
    # d2 = torch.ones((1,2,128,256))*2.1
    # da = torch.cat((d1,d2),2)
    # e1 = torch.cat((c1,da),1)
    # a = a + e1
    fm = vgg(a)
    max = torch.max(fm)
    min = torch.min(fm)
    mean = torch.mean(fm)
    print(fm)
    print(fm.size())
    print(max)
    print(min)
    print(mean)
