import enum
import os
import torch
import numpy as np
import random
import tqdm
import math
import json
import re
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import  cv2

from datetime import datetime
from torch.utils.data import  Dataset
from PIL import Image,ImageEnhance
from torch.optim.lr_scheduler import LambdaLR
from audioop import add

def min_max_normalize(image):
    np_image = np.array(image).astype(np.float32)
    np_image = (np_image - np.min(np_image)) / (np.max(np_image) - np.min(np_image))
    return torch.tensor(np_image)

def add_gaussian_noise(image, noise_ratio):
    alpha = torch.randn_like(image)
    noise = noise_ratio * alpha
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, 0, 1)
    return noisy_image

def visual_result(input,filename):
    if len(input.shape)==4:
        np_image = input[0].cpu().permute(1,2,0).numpy()  
    elif len(input.shape)==3:
        np_image = input.cpu().permute(1,2,0).numpy()  
    if np_image.min()<0:    
        np_image = np_image * 0.5 + 0.5  # 假设图像已归一化为[-1, 1]
    plt.imshow(np_image)
    plt.axis('off')
    plt.savefig(filename) 
      
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image=transform(image)
    image = min_max_normalize(image)
    return image

def preprocess_image_cgan(image_path):
    transform = transforms.Compose([
        transforms.Resize(size=(128, 128), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  
    ])
    image = Image.open(image_path).convert('RGB')
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.8)  
    image=transform(image)
    return image

def PrintModelInfo(model):
    """Print the parameter size and shape of model detail"""
    total_params = 0
    for name, param in model.named_parameters():
        num_params = torch.prod(torch.tensor(param.shape)).item() * param.element_size() / (1024 * 1024)  # 转换为MB
        print(f"{name}: {num_params:.4f} MB, Shape: {param.shape}")
        total_params += num_params
    print(f"Total number of parameters: {total_params:.4f} MB")     

def save_model(loss_now,best_loss,model,save_path,model_name):
    if loss_now < best_loss:
        best_loss = loss_now
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path,model_name))
        print("--------------------->Saving model checkpoint {} at {}".format(save_path+model_name, 
                                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        return best_loss
        
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def compute_gradient_penalty(crit, real_samples, fake_samples):
    """
    计算梯度惩罚项
    """
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = crit(interpolates)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones_like(d_interpolates),
                                    create_graph=True, retain_graph=True)[0]
    gradients_norm = gradients.view(gradients.size(0), -1).norm(p=2, dim=1)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty

def load_and_cache(data_path,cache_file,shuffle=False):
    if cache_file is not None and os.path.exists(cache_file):
        print("Loading features from cached file ", cache_file)
        features = torch.load(cache_file)
    else:
        print("Creating features from dataset at ", data_path)
        examples = []
        for img_name in os.listdir(data_path):
            img_path = os.path.join(data_path, img_name)
            examples.append(img_path)
        features=[]       
        for example_path in tqdm.tqdm(examples):
            processed_image=preprocess_image(example_path)
            features.append(processed_image)        
        if shuffle:
            random.shuffle(features)
        if not os.path.exists(cache_file):
            print("Saving features into cached file ", cache_file)
            torch.save(features, cache_file)
    return features

def sort_key(filename):
    match = re.search(r'(\d+)_(\d+)_(\d+)\.', filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    else:
        return float('inf'), float('inf'), float('inf')   

def load_and_cache_withlabel(image_path,label_path,cache_file,shuffle=False):
    if cache_file is not None and os.path.exists(cache_file):
        print("Loading features from cached file ", cache_file)
        features = torch.load(cache_file)
    else:
        print("Creating features from dataset at ", image_path,label_path)
        images,labels = [],[]
        for img_name in os.listdir(image_path):
            img_path = os.path.join(image_path, img_name)
            images.append(img_path)
        images=sorted(images,key=sort_key)
        with open(label_path,'r') as json_file:
             for i,line in enumerate(json_file):
                labels.append(json.loads(line))
        features = []
        def get_label_data(label):
            file=label["file:"]
            match = re.match(r'(\d+)_(\d+)', label["status"])
            if match:
                n = int(match.group(1))
                m = int(match.group(2))
                result = 7 * (n - 1) + m-1
            status=result
            O2=label["O2"]
            O2_CO2=label["O2_CO2"]
            CH4=label["CH4"]
            O2_CH4=label["CH4_CO2"]
            return file,status,O2,O2_CO2,CH4,O2_CH4        
              
        total_iterations = len(images)  # 设置总的迭代次数  
        for image_path,label in tqdm.tqdm(zip(images,labels),total=total_iterations):
            #processed_image=preprocess_image(image_path)
            processed_image=preprocess_image_cgan(image_path) #only cgan
            #visual_result(processed_image,"output.jpg")
            file,status,O2,O2_CO2,CH4,O2_CH4 =get_label_data(label)
            feature = {
                "image": processed_image,
                "file": file,
                "status":status,
                "O2":O2,
                "O2_CO2":O2_CO2,
                "CH4":CH4,
                "O2_CH4":O2_CH4
            }
            features.append(feature)
 
        if shuffle:
            random.shuffle(features)
        if not os.path.exists(cache_file):
            print("Saving features into cached file ", cache_file)
            torch.save(features, cache_file)
    return features

class FireDatasetGan(Dataset):
    def __init__(self,features,num_instances):
        self.feature=features
        self.num_instances=num_instances
    
    def __len__(self):
        return int(self.num_instances)
    
    def __getitem__(self, index):
        feature = self.feature[index]
        return feature 

class FireDataset(Dataset):
    def __init__(self,features,num_instances):
        self.feature=features
        self.num_instances=num_instances
    
    def __len__(self):
        return int(self.num_instances)
    
    def __getitem__(self, index):
        feature = self.feature[index]
        image=feature["image"]
        status=feature["status"]
        return image,status
    
class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.window_size = 11
        self.size_average = True
        self.channel = 1
        self.weight = torch.ones(1, 1, self.window_size, self.window_size) / (self.window_size ** 2)
    
    def gaussian_window(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img_p, img_o):
        (_, channel, _, _) = img_p.size()
        window = self.create_window(self.window_size, channel)
        if img_p.is_cuda:
            window = window.cuda(img_p.get_device())
        window = window.type_as(img_p)
        return 1 - self.ssim(img_p, img_o, window, self.window_size, channel, self.size_average)  
    
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer_module = dict([*self.model.named_modules()])[self.target_layer]
        target_layer_module.register_forward_hook(forward_hook)
        target_layer_module.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class):
        self.model.zero_grad()
        output = self.model(input_image)
        target_class = int(target_class)  # 确保 target_class 是整数
        loss = output[0, target_class]
        loss.backward()

        if self.gradients is None:
            raise ValueError("Gradients not captured. Ensure that hooks are correctly registered and model forward pass is correct.")

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2)) #计算每个通道上梯度的均值
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam
    
def apply_colormap_on_image(org_im, activation, colormap_name):
    colormap = plt.get_cmap(colormap_name)
    no_trans_heatmap = colormap(activation)
    heatmap = no_trans_heatmap[:, :, :3]
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    overlayed_img = cv2.addWeighted(org_im, 0.5, heatmap, 0.5, 0)
    return overlayed_img