import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, dim=128):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(dim, dim)
        def dconv_bn_relu(in_dim, out_dim, kernel_size=4, stride=2, padding=1, output_padding=0):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, output_padding),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
        self.ls = nn.Sequential(
            dconv_bn_relu(z_dim + c_dim, dim * 16, 4, 1, 0, 0),  # (N, dim * 4, 4, 4)
            dconv_bn_relu(dim * 16, dim * 8),  # (N, dim * 2, 8, 8)
            dconv_bn_relu(dim * 8, dim*4),   # (N, dim, 16, 16)
            dconv_bn_relu(dim * 4, dim*2),   # (N, dim, 32, 32)
            dconv_bn_relu(dim * 2, dim),   # (N, dim, 64, 64)
            nn.ConvTranspose2d(dim, 3, 4, 2, padding=1), # (N, 3, 128, 128)
            nn.Tanh()  
        )

    def forward(self, z, c):
        # z: (N, z_dim), c: (N, c_dim)
        c=self.label_embedding(c.long())
        x = torch.cat([z, c], 1)
        x = self.ls(x.view(x.size(0), x.size(1), 1, 1))
        return x

class Discriminator(nn.Module):
    def __init__(self, label_dim,img_dim):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(label_dim, label_dim)
        img_flat_dim = int(np.prod(img_dim))

        self.model = nn.Sequential(
            nn.Linear(img_flat_dim + label_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )
            
    def forward(self, img, labels):
        c = self.label_embedding(labels.long())
        x = torch.cat([img.view(img.size(0), -1), c], 1)
        x = x + 0.05 * torch.randn_like(x) #add noise
        validity = self.model(x)
        return validity
                
# class Generator(nn.Module):
#     def __init__(self, z_dim, label_dim, img_shape, dropout_prob=0):
#         super(Generator, self).__init__()
#         self.label_embedding = nn.Embedding(label_dim, z_dim)  # 标签嵌入维度直接设为噪声向量的维度
#         self.init_size = img_shape[1] // 16 
#         self.l1 = nn.Sequential(nn.Linear(z_dim + z_dim, 512 * self.init_size ** 2))  # 增加了通道数
#         self.conv1 = self._conv_block(512, 512, 2, dropout_prob)
#         self.conv2 = self._conv_block(512, 256, 2, dropout_prob)
#         self.conv3 = self._conv_block(256, 128, 2, dropout_prob)
#         self.conv4 = self._conv_block(128, 128, 2, dropout_prob)
#         self.conv5 = self._conv_block(128, 128, 1, dropout_prob)
#         self.final_conv = nn.Conv2d(128, img_shape[0], 3, stride=1, padding=1)  
#         self.tanh = nn.Tanh()  
#         self._initialize_weights()

#     def _conv_block(self, in_channels, out_channels, scale_factor, dropout_prob):
#         layers = [
#             nn.BatchNorm2d(in_channels),
#             nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),  
#             nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(dropout_prob)   
#         ]
#         return nn.Sequential(*layers)

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                 init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
#                 if m.bias is not None:
#        # class Discriminator(nn.Module):
#     def __init__(self, label_dim,img_dim,dropout_prob=0.5):
#         super(Discriminator, self).__init__()
#         self.label_embedding = nn.Embedding(label_dim, label_dim)
#         img_flat_dim = int(np.prod(img_dim))

#         self.model = nn.Sequential(
#             nn.Linear(img_flat_dim + label_dim, 128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(128, 64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(64, 1),
#         )
            
#     def forward(self, img, labels):
#         c = self.label_embedding(labels)
#         x = torch.cat([img.view(img.size(0), -1), c], 1)
#         x = x + 0.05 * torch.randn_like(x) #add noise
#         validity = self.model(x)
#         return validity


    