import torch

import utils
from torch import nn
import torch.nn.functional as F

class ConvNormLReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False, sn=False):
        
        pad_layer = {
            "zero":    nn.ZeroPad2d,
            "same":    nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError
            
        if sn:
            super(ConvNormLReLU, self).__init__(
                pad_layer[pad_mode](padding),
                nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias)),
                nn.InstanceNorm2d(out_ch,affine=True), #affine=True 意味着alpha和bete也是可学习的参数
                nn.LeakyReLU(0.2, inplace=True)  #inplace=True 表示直接将激活值覆盖在原来的tensor中
            )
        else:
            super(ConvNormLReLU, self).__init__(
                pad_layer[pad_mode](padding),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
                nn.InstanceNorm2d(out_ch,affine=True), #affine=True 意味着alpha和bete也是可学习的参数
                nn.LeakyReLU(0.2, inplace=True)  #inplace=True 表示直接将激活值覆盖在原来的tensor中
            )

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3,stride=1, depth_multiplier=1, bias=False):
        super(SeparableConv2d, self).__init__()
        if kernel_size==3 and stride==1:
            self.pad =  nn.ReflectionPad2d(padding=(1, 1, 1, 1))
            self.depthwise = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels*depth_multiplier,
                    kernel_size=3,
                    stride=1,
                    # padding=(0, 0, 0, 0, 1, 1, 1, 1),
                    # padding_mode='reflect',
                    groups=in_channels,
                    bias=bias
            )
            
        if stride == 2:
            self.pad =  nn.ReflectionPad2d(padding=(1, 1, 1, 1))
            self.depthwise =  nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels*depth_multiplier,
                kernel_size=3,
                stride=2,
                # padding=(0, 0, 0, 0, 0, 1, 0, 1),
                # padding_mode='reflect',
                groups=in_channels,
                bias=bias
            )

        self.depthwiseNormLRelu = nn.Sequential(
            self.pad,
            self.depthwise,
            nn.InstanceNorm2d(in_channels,affine=True), #affine=True 意味着alpha和bete也是可学习的参数
            nn.LeakyReLU(0.2, inplace=True)  #inplace=True 表示直接将激活值覆盖在原来的tensor中
        )

        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias
        )

        self.pointwiseNormLRelu = nn.Sequential(
            self.pointwise,
            nn.InstanceNorm2d(out_channels,affine=True), #affine=True 意味着alpha和bete也是可学习的参数
            nn.LeakyReLU(0.2, inplace=True)  #inplace=True 表示直接将激活值覆盖在原来的tensor中
        )

    def forward(self, x):
        out = self.depthwiseNormLRelu(x) 
        out = self.pointwiseNormLRelu(out)
        return out


class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch*expansion_ratio))
        layers = []
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))
        
        # dw
        layers.append(ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True))
        # pw
        layers.append(nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False))
        layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))

        self.layers = nn.Sequential(*layers)
        
    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out


# class Generator(nn.Module):
#     def __init__(self, ):
#         super().__init__()
#
#         self.block_a_1 = nn.Sequential(
#             ConvNormLReLU(5, 32, kernel_size=7, padding=3),             #32*256×256
#         )
#
#         self.block_a_2 = nn.Sequential(
#             ConvNormLReLU(32, 64, stride=2, padding=(0, 1, 0, 1)),      #64*128×128
#             ConvNormLReLU(64, 64)                                       #64*128×128
#         )
#
#
#         self.block_b = nn.Sequential(
#             ConvNormLReLU(64, 128, stride=2, padding=(0, 1, 0, 1)),     #128*64×64
#             ConvNormLReLU(128, 128)                                     #128*64×64
#         )
#
#         self.block_c = nn.Sequential(
#             ConvNormLReLU(128, 128),
#             InvertedResBlock(128, 256, 2),
#             InvertedResBlock(256, 256, 2),
#             InvertedResBlock(256, 256, 2),
#             InvertedResBlock(256, 256, 2),
#             ConvNormLReLU(256, 128),
#         )
#
#         self.block_d = nn.Sequential(
#             ConvNormLReLU(128, 128),
#             ConvNormLReLU(128, 128)
#         )
#
#         self.block_e = nn.Sequential(
#             ConvNormLReLU(128, 64),
#             ConvNormLReLU(64, 64),
#             ConvNormLReLU(64, 32, kernel_size=7, padding=3)
#         )
#
#         self.out_layer = nn.Sequential(
#             nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.Tanh()
#         )
#
#     def forward(self, input, saliency, align_corners=True):
#         input = torch.cat((input, saliency), 1)
#
#         out = self.block_a_1(input)
#         out = self.block_a_2(out)
#         half_size = out.size()[-2:]
#         out = self.block_b(out)
#         out = self.block_c(out)
#
#         if align_corners:
#             out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
#         else:
#             out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
#         out = self.block_d(out)
#
#         if align_corners:
#             out = F.interpolate(out, input.size()[-2:], mode="bilinear", align_corners=True)
#         else:
#             out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
#         out = self.block_e(out)
#
#         out = self.out_layer(out)
#         return out


class Generator(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.block_a_1 = nn.Sequential(
            ConvNormLReLU(3, 32, kernel_size=7, padding=3),             #32*256×256
        )

        # self.block_saliency_1 = nn.Sequential(
        #     ConvNormLReLU(2, 32, kernel_size=7, padding=3),             #32*256×256
        # )

        self.block_a_2 = nn.Sequential(
            ConvNormLReLU(34, 64, stride=2, padding=(0, 1, 0, 1)),      #64*128×128
            ConvNormLReLU(64, 64)                                       #64*128×128
        )

        # self.block_saliency_2 = nn.Sequential(
        #     ConvNormLReLU(32, 64, stride=2, padding=(0, 1, 0, 1)),      #64*128×128
        #     ConvNormLReLU(64, 64)                                       #64*128×128
        # )


        self.block_b = nn.Sequential(
            ConvNormLReLU(66, 128, stride=2, padding=(0, 1, 0, 1)),     #128*64×64
            ConvNormLReLU(128, 128)                                     #128*64×64
        )

        # self.block_saliency_3 = nn.Sequential(
        #     ConvNormLReLU(64, 128, stride=2, padding=(0, 1, 0, 1)),     #128*64×64
        #     ConvNormLReLU(128, 128)                                     #128*64×64
        # )

        self.block_c = nn.Sequential(
            ConvNormLReLU(130, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            ConvNormLReLU(256, 128),
        )

        self.block_d = nn.Sequential(
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 128)
        )

        self.block_e = nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64, 64),
            ConvNormLReLU(64, 32, kernel_size=7, padding=3)
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, input, saliency, align_corners=True):
        out = self.block_a_1(input)
        saliency_out = F.interpolate(saliency, size=out.size()[-2:], mode="nearest")
        out = torch.cat((out, saliency_out), 1)

        out = self.block_a_2(out)
        half_size = out.size()[-2:]
        saliency_out = F.interpolate(saliency_out, half_size, mode="nearest")
        out = torch.cat((out, saliency_out), 1)

        out = self.block_b(out)
        quarter_size = out.size()[-2:]
        saliency_out = F.interpolate(saliency_out, quarter_size, mode="nearest")
        out = torch.cat((out, saliency_out), 1)

        out = self.block_c(out)

        if align_corners:
            out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_d(out)

        if align_corners:
            out = F.interpolate(out, input.size()[-2:], mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_e(out)

        out = self.out_layer(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, ch, n_dis, sn):
        super(Discriminator, self).__init__()
        self.ch = ch // 2

        if sn:
            self.ConvLReLU1 = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(3, self.ch, kernel_size=3, stride=1, padding=1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True)
                )
            self.ConvLReLU2_1 = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(self.ch, self.ch * 2, kernel_size=3, stride=2, padding=1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.ConvLReLU3_1 = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(self.ch * 4, self.ch * 4, kernel_size=3, stride=2, padding=1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.Conv5 = nn.utils.spectral_norm(nn.Conv2d(self.ch * 8, 1, kernel_size=3, stride=1, padding=1, bias=False))

        else:
            self.ConvLReLU1 = nn.Sequential(
                nn.Conv2d(3, self.ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                )
            self.ConvLReLU2_1 = nn.Sequential(
                nn.Conv2d(self.ch, self.ch*2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.ConvLReLU3_1 = nn.Sequential(
                nn.Conv2d(self.ch * 4, self.ch * 4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.Conv5 = nn.Conv2d(self.ch * 8, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.ConvNormLReLU2_1 = ConvNormLReLU(self.ch*2, self.ch*4, kernel_size=3, stride=1, padding=1, sn=sn)

        self.ConvNormLReLU3_1 = ConvNormLReLU(self.ch*4, self.ch*8, kernel_size=3, stride=1, padding=1, sn=sn)

        self.ConvNormLReLU4 = ConvNormLReLU(self.ch*8, self.ch*8, kernel_size=3, stride=1, padding=1, sn=sn)

        # self.Conv5 = nn.Sequential(
        #     nn.Conv2d(self.ch*8, 1, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.Sigmoid()
        #     )

        # utils.initialize_weights(self)

    def forward(self, input):
        out = self.ConvLReLU1(input)

        out = self.ConvLReLU2_1(out)
        out = self.ConvNormLReLU2_1(out)

        out = self.ConvLReLU3_1(out)
        out = self.ConvNormLReLU3_1(out)

        out = self.ConvNormLReLU4(out)

        out = self.Conv5(out)

        return out



