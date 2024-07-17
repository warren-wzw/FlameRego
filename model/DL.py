import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = SeparableConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SeparableConv2d(out_channels, out_channels, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)  # 残差连接
        out = self.relu(out)
        return out

class DL_MUL_COVRES_DPW_FC96(nn.Module):
    def __init__(self, num_classes=63, dropout_prob=0.1, hidden=24, num_encoders=4):
        super(DL_MUL_COVRES_DPW_FC96, self).__init__()
        self.num_encoders = num_encoders
        self.hidden=int(hidden)
        self.encoders = nn.ModuleList([self._create_encoder(self.hidden) for _ in range(num_encoders)])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(num_encoders * self.hidden * 8 * 8, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self._initialize_weights()

    def _create_encoder(self, hidden):
        layers = [
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=hidden, kernel_size=1, stride=1, padding=0)
        ]
        return nn.Sequential(*layers)
      
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        encode_outputs = [encoder(input.clone()) for encoder in self.encoders]
        concatenated = torch.cat(encode_outputs, dim=1)
        flattened = concatenated.view(concatenated.size(0), -1)
        output = self.dropout(flattened)
        fc1_output = F.leaky_relu(self.fc1(output))
        return fc1_output

    def get_concatenated_features(self, input):
        encode_outputs = [encoder(input.clone()) for encoder in self.encoders]
        concatenated = torch.cat(encode_outputs, dim=1)
        return concatenated
    
class DL_3COV_RES_DPW_FC96(nn.Module):
    def __init__(self, num_classes=63, dropout_prob=0.1, hidden=32):
        super(DL_3COV_RES_DPW_FC96, self).__init__()
        self.conv1_en = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_en = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_en = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_en = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5_en = ResidualBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv6_en = ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_pw  = nn.Conv2d(in_channels=512, out_channels=hidden, kernel_size=1, stride=1, padding=0)

        self.conv12_en = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv22_en = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv32_en = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv42_en = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv52_en = ResidualBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv62_en = ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_pw2  = nn.Conv2d(in_channels=512, out_channels=hidden, kernel_size=1, stride=1, padding=0)

        self.conv13_en = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv23_en = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv33_en = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv43_en = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv53_en = ResidualBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv63_en = ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_pw3  =  nn.Conv2d(in_channels=512, out_channels=hidden, kernel_size=1, stride=1, padding=0)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(3*hidden* 8 * 8, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def encode1(self, input):
        hidden1 = self.conv1_en(input)
        hidden1 = F.leaky_relu(hidden1)
        hidden2 = self.conv2_en(hidden1)
        hidden2 = self.pool(hidden2)
        hidden3 = self.conv3_en(hidden2)
        hidden3 = self.pool(hidden3)
        hidden4 = self.conv4_en(hidden3)
        hidden4 = self.pool(hidden4)
        hidden5 = self.conv5_en(hidden4)
        hidden5 = self.pool(hidden5)
        hidden6 = self.conv6_en(hidden5)
        hidden6 = self.pool(hidden6)
        hidden7 = self.conv_pw(hidden6)
        return hidden7

    def encode2(self, input):
        hidden1 = self.conv12_en(input)
        hidden1 = F.leaky_relu(hidden1)
        hidden2 = self.conv22_en(hidden1)
        hidden2 = self.pool(hidden2)
        hidden3 = self.conv32_en(hidden2)
        hidden3 = self.pool(hidden3)
        hidden4 = self.conv42_en(hidden3)
        hidden4 = self.pool(hidden4)
        hidden5 = self.conv52_en(hidden4)
        hidden5 = self.pool(hidden5)
        hidden6 = self.conv62_en(hidden5)
        hidden6 = self.pool(hidden6)
        hidden7 = self.conv_pw2(hidden6)
        return hidden7

    def encode3(self, input):
        hidden1 = self.conv13_en(input)
        hidden1 = F.leaky_relu(hidden1)
        hidden2 = self.conv23_en(hidden1)
        hidden2 = self.pool(hidden2)
        hidden3 = self.conv33_en(hidden2)
        hidden3 = self.pool(hidden3)
        hidden4 = self.conv43_en(hidden3)
        hidden4 = self.pool(hidden4)
        hidden5 = self.conv53_en(hidden4)
        hidden5 = self.pool(hidden5)
        hidden6 = self.conv63_en(hidden5)
        hidden6 = self.pool(hidden6)
        hidden7 = self.conv_pw3(hidden6)
        return hidden7
    
    def forward(self, input):
        encode_output1 = self.encode1(input.clone())
        encode_output2 = self.encode2(input.clone())
        encode_output3 = self.encode3(input.clone())
        concatenated = torch.cat((encode_output1, encode_output2, encode_output3), dim=1)
        flattened = concatenated.view(concatenated.size(0), -1)
        output=self.dropout(flattened)
        #fc1_output = torch.nn.functional.softmax(F.leaky_relu(self.fc1(output)),dim=-1)
        fc1_output = F.leaky_relu(self.fc1(output))
        return fc1_output
    
    def get_concatenated_features(self, input):
        encode_output1 = self.encode1(input.clone())
        encode_output2 = self.encode2(input.clone())
        encode_output3 = self.encode3(input.clone())
        concatenated = torch.cat((encode_output1, encode_output2, encode_output3), dim=1)
        return concatenated

class DL_3COV_FC96(nn.Module):
    def __init__(self, num_classes=63, dropout_prob=0.1, hidden=32):
        super(DL_3COV_FC96, self).__init__()
        self.conv1_en = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1_en = nn.BatchNorm2d(32)
        self.conv2_en = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2_en = nn.BatchNorm2d(64)
        self.conv3_en = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3_en = nn.BatchNorm2d(128)
        self.conv4_en = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4_en = nn.BatchNorm2d(256)
        self.conv5_en = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn5_en = nn.BatchNorm2d(512)
        self.conv6_en = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn6_en = nn.BatchNorm2d(512)
        self.conv_pw = nn.Conv2d(in_channels=512, out_channels=hidden, kernel_size=1, stride=1, padding=0)

        self.conv12_en = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn12_en = nn.BatchNorm2d(32)
        self.conv22_en = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn22_en = nn.BatchNorm2d(64)
        self.conv32_en = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn32_en = nn.BatchNorm2d(128)
        self.conv42_en = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn42_en = nn.BatchNorm2d(256)
        self.conv52_en = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn52_en = nn.BatchNorm2d(512)
        self.conv62_en = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn62_en = nn.BatchNorm2d(512)
        self.conv_pw2 = nn.Conv2d(in_channels=512, out_channels=hidden, kernel_size=1, stride=1, padding=0)

        self.conv13_en = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn13_en = nn.BatchNorm2d(32)
        self.conv23_en = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn23_en = nn.BatchNorm2d(64)
        self.conv33_en = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn33_en = nn.BatchNorm2d(128)
        self.conv43_en = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn43_en = nn.BatchNorm2d(256)
        self.conv53_en = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn53_en = nn.BatchNorm2d(512)
        self.conv63_en = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn63_en = nn.BatchNorm2d(512)
        self.conv_pw3 = nn.Conv2d(in_channels=512, out_channels=hidden, kernel_size=1, stride=1, padding=0)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(3*hidden* 8 * 8, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def encode1(self, input):
        hidden1 = self.bn1_en(self.conv1_en(input))
        hidden1 = F.leaky_relu(hidden1)
        hidden2 = self.bn2_en(self.conv2_en(hidden1))
        hidden2 = self.pool(hidden2)
        hidden3 = self.bn3_en(self.conv3_en(hidden2))
        hidden3 = self.pool(hidden3)
        hidden4 = self.bn4_en(self.conv4_en(hidden3))
        hidden4 = self.pool(hidden4)
        hidden5 = self.bn5_en(self.conv5_en(hidden4))
        hidden5 = self.pool(hidden5)
        hidden6 = self.bn6_en(self.conv6_en(hidden5))
        hidden6 = self.pool(hidden6)
        hidden7 = self.conv_pw(hidden6)
        return hidden7

    def encode2(self, input):
        hidden1 = self.bn12_en(self.conv12_en(input))
        hidden1 = F.leaky_relu(hidden1)
        hidden2 = self.bn22_en(self.conv22_en(hidden1))
        hidden2 = self.pool(hidden2)
        hidden3 = self.bn32_en(self.conv32_en(hidden2))
        hidden3 = self.pool(hidden3)
        hidden4 = self.bn42_en(self.conv42_en(hidden3))
        hidden4 = self.pool(hidden4)
        hidden5 = self.bn52_en(self.conv52_en(hidden4))
        hidden5 = self.pool(hidden5)
        hidden6 = self.bn62_en(self.conv62_en(hidden5))
        hidden6 = self.pool(hidden6)
        hidden7 = self.conv_pw2(hidden6)
        return hidden7

    def encode3(self, input):
        hidden1 = self.bn13_en(self.conv13_en(input))
        hidden1 = F.leaky_relu(hidden1)
        hidden2 = self.bn23_en(self.conv23_en(hidden1))
        hidden2 = self.pool(hidden2)
        hidden3 = self.bn33_en(self.conv33_en(hidden2))
        hidden3 = self.pool(hidden3)
        hidden4 = self.bn43_en(self.conv43_en(hidden3))
        hidden4 = self.pool(hidden4)
        hidden5 = self.bn53_en(self.conv53_en(hidden4))
        hidden5 = self.pool(hidden5)
        hidden6 = self.bn63_en(self.conv63_en(hidden5))
        hidden6 = self.pool(hidden6)
        hidden7 = self.conv_pw3(hidden6)
        return hidden7
    
    def forward(self, input):
        encode_output1 = self.encode1(input.clone())
        encode_output2 = self.encode2(input.clone())
        encode_output3 = self.encode3(input.clone())
        concatenated = torch.cat((encode_output1, encode_output2, encode_output3), dim=1)
        flattened = concatenated.view(concatenated.size(0), -1)
        output=self.dropout(flattened)
        fc1_output = F.leaky_relu(self.fc1(output))
        return fc1_output

class DL_DSW_RES_FC100(nn.Module):
    def __init__(self, num_classes=63, dropout_prob=0.1):
        super(DL_DSW_RES_FC100, self).__init__()
        self.conv1_en = ResidualBlock(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_en = ResidualBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_en = ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_en = ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5_en = ResidualBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_pw  = nn.Conv2d(in_channels=512, out_channels=100, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(100 * 16 * 16, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        
    def encode(self, input):
        hidden1 = self.conv1_en(input)
        hidden1 = F.leaky_relu(hidden1) #32 256 256 
        hidden2 = self.conv2_en(hidden1)
        hidden2 = self.pool(hidden2)  # 64x128x128
        hidden3 = self.conv3_en(hidden2)
        hidden3 = self.pool(hidden3)  # 128x64x64
        hidden4 = self.conv4_en(hidden3)
        hidden4 = self.pool(hidden4)  # 256x32x32
        hidden5 = self.conv5_en(hidden4)
        hidden5 = self.pool(hidden5)  # 512x16x16 
        hidden6 = self.conv_pw(hidden5)
        hidden6 = self.dropout(hidden6)
        return hidden6
    
    def forward(self, input):
        encode_output = self.encode(input)
        encode_output = encode_output.view(encode_output.size(0), -1)  # 展平
        output = F.leaky_relu(self.fc1(encode_output))
        return output

class DL_FC100_FC63(nn.Module):
    def __init__(self, num_classes=63, dropout_prob=0.1):
        super(DL_FC100_FC63, self).__init__()
        self.conv1_en = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_en = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_en = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_en = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        
    def encode(self, input):
        hidden1 = F.leaky_relu(self.conv1_en(input))
        hidden1 = self.pool(hidden1)  # 64x128x128
        hidden2 = F.leaky_relu(self.conv2_en(hidden1)) 
        hidden2 = self.pool(hidden2)  # 128x64x64
        hidden3 = F.leaky_relu(self.conv3_en(hidden2))
        hidden3 = self.pool(hidden3)  # 256x32x32
        hidden4 = F.leaky_relu(self.conv4_en(hidden3))
        hidden4 = self.pool(hidden4)  # 512x16x16 
        return hidden4
    
    def forward(self, input):
        encode_output = self.encode(input)
        encode_output = encode_output.view(encode_output.size(0), -1)  # 展平
        fc1_output = F.leaky_relu(self.fc1(encode_output))
        output = self.fc2(fc1_output)
        return output

class DL_PW_FC63(nn.Module):
    def __init__(self, num_classes=63, dropout_prob=0.1):
        super(DL_PW_FC63, self).__init__()
        self.conv1_en = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_en = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_en = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_en = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_pw  = nn.Conv2d(in_channels=512, out_channels=100, kernel_size=1, stride=1, padding=0)
        #self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(100 * 16 * 16, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        
    def encode(self, input):
        hidden1 = F.leaky_relu(self.conv1_en(input))
        hidden1 = self.pool(hidden1)  # 64x128x128
        hidden2 = F.leaky_relu(self.conv2_en(hidden1)) 
        hidden2 = self.pool(hidden2)  # 128x64x64
        hidden3 = F.leaky_relu(self.conv3_en(hidden2))
        hidden3 = self.pool(hidden3)  # 256x32x32
        hidden4 = F.leaky_relu(self.conv4_en(hidden3))
        hidden4 = self.pool(hidden4)  # 512x16x16 
        hidden5 = self.conv_pw(hidden4)
        hidden5 = self.dropout(hidden5)
        return hidden5
    
    def forward(self, input):
        encode_output = self.encode(input)
        encode_output = encode_output.view(encode_output.size(0), -1)  # 展平
        output = F.leaky_relu(self.fc1(encode_output))
        return output