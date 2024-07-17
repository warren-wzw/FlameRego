import os
import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class UpsampleLayer(nn.Module):
    def __init__(self, g):
        super(UpsampleLayer, self).__init__()
        self.g = g

    def forward(self, input):
        batch_size, channels, height, width = input.size()
        output = torch.nn.functional.interpolate(input, scale_factor=self.g, mode='nearest')
        return output
    
class DAE(nn.Module):
    def __init__(self,dropout_prob=0.1):
        super(DAE,self).__init__()
        """encode in 3,256,256"""
        self.conv1_en = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,stride=1,padding=1)
        self.conv2_en = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3,stride=1,padding=1)
        self.conv3_en = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3,stride=1,padding=1)
        self.conv4_en = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3,stride=1,padding=1)
        self.conv5_en = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3,stride=1,padding=1)
        self.conv6_en = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        """decode in 1 4 4"""
        self.upsample_layer1=UpsampleLayer(g=2) 
        self.conv1_de=nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,stride=1,padding=1)
        self.upsample_layer2=UpsampleLayer(g=2) 
        self.conv2_de=nn.Conv2d(in_channels=4,out_channels=4,kernel_size=3,stride=1,padding=1)
        self.upsample_layer3=UpsampleLayer(g=2) 
        self.conv3_de=nn.Conv2d(in_channels=4,out_channels=8,kernel_size=3,stride=1,padding=1)
        self.upsample_layer4=UpsampleLayer(g=2) 
        self.conv4_de=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.upsample_layer5=UpsampleLayer(g=2) 
        self.conv5_de=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.upsample_layer6=UpsampleLayer(g=2) 
        self.conv6_de=nn.Conv2d(in_channels=32,out_channels=3,kernel_size=3,stride=1,padding=1)
        self._initialize_weights()
        self.dropout = nn.Dropout(dropout_prob)  # 添加 dropout 层
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) :
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                    
    def encode(self, input):
        hidden1 = F.leaky_relu(self.conv1_en(input))
        hidden2 = self.pool(hidden1)  # 32 128 128
        hidden3 = F.leaky_relu(self.conv2_en(hidden2)) 
        hidden3 = self.pool(hidden3)  # 16 64 64
        hidden4 = F.leaky_relu(self.conv3_en(hidden3)) 
        hidden4 = self.pool(hidden4)  # 8 32 32
        hidden5 = F.leaky_relu(self.conv4_en(hidden4))  
        hidden5 = self.pool(hidden5)  # 4 16 16
        hidden6 = F.leaky_relu(self.conv5_en(hidden5))  
        hidden6 = self.pool(hidden6)  # 4 8 8
        output = F.leaky_relu(self.conv6_en(hidden6))  
        output = self.pool(output)  # 1 4 4
        # return output, hidden1, hidden2, hidden3, hidden4, hidden5, hidden6
        return output
        
    def decode(self, input):
        hidden = F.leaky_relu(self.conv1_de(self.upsample_layer1(input)))  # 4 8 8
        #hidden = hidden+hidden6  
        hidden = F.leaky_relu(self.conv2_de(self.upsample_layer2(hidden)))  # 4 16 16        
        #hidden = hidden + hidden5
        hidden = F.leaky_relu(self.conv3_de(self.upsample_layer3(hidden)))  # 8 32 32
        #hidden = hidden+hidden4
        hidden = F.leaky_relu(self.conv4_de(self.upsample_layer4(hidden)))  # 16 64 64        
        #hidden = hidden + hidden3  # 跳跃连接
        hidden = F.leaky_relu(self.conv5_de(self.upsample_layer5(hidden)))  # 32 128 128        
        #hidden = hidden + hidden2  # 跳跃连接
        hidden = torch.tanh(self.conv6_de(self.upsample_layer6(hidden)))  # 3 256 256
        return hidden
                
    def forward(self, input):
        # encode_output, hidden1, hidden2, hidden3, hidden4, hidden5, hidden6 = self.encode(input)
        encode_output= self.encode(input)
        encode_output = self.dropout(encode_output)  # 在编码后的输出上应用 dropout
        output = self.decode(encode_output)
        return output    
    
class REGO(nn.Module):
    def __init__(self,dropout_prob=0.1):
        super(REGO,self).__init__()
        """rego in 3 256 256 """
        self.conv1_re = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,stride=1,padding=1) 
        self.conv2_re = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3,stride=1,padding=1)
        self.conv3_re = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3,stride=1,padding=1)
        self.conv4_re = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3,stride=1,padding=1)
        self.conv5_re = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8*8*1, 1)  # fc，将8*8*1的输入变为1个输出
        self._initialize_weights()
        self.dropout=nn.Dropout(dropout_prob)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) :
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                    
    def forward(self,input):
        hidden = self.pool(F.leaky_relu(self.conv1_re(input))) #32 128 128
        hidden = self.pool(F.leaky_relu(self.conv2_re(hidden)))#16 64 64 
        hidden = self.pool(F.leaky_relu(self.conv3_re(hidden)))#8 32 32
        hidden = self.pool(F.leaky_relu(self.conv4_re(hidden)))#4 16 16
        hidden = self.pool(F.leaky_relu(self.conv5_re(hidden)))#1 8 8 
        flattened_hidden = hidden.view(input.size(0), -1)
        flattened_hidden = self.dropout(flattened_hidden)
        output=self.fc1(flattened_hidden)
        return output
        
        