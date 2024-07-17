import os
import sys
import torch
import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import (DataLoader, SequentialSampler)
os.chdir(sys.path[0])
sys.path.append('..')
from model.DAE_GAN import DAE
from model.utils import FireDatasetGan,load_and_cache,add_gaussian_noise, visual_result
BATCH_SIZE=1
MODEL_PATH="../output/output_model/DAE_GAN/DaeModel_mse_ssim_wgan.pth"
TEST_DATE="../dataset/fine_tune_images_GAN/test/"
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_cache="../dataset/cache/test_cache_fine_tune.pt"

def CreateDataloader(data_file,cached_file):
    features = load_and_cache(data_file,cached_file,shuffle=True)    
    dataset = FireDatasetGan(features=features,num_instances=len(features))
    loader = DataLoader(dataset=dataset,batch_size=BATCH_SIZE, shuffle=True)
    return loader

def main():
    model=DAE().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    test_loader = CreateDataloader(TEST_DATE,test_cache)
    model.eval()
    with torch.no_grad():
        test_iterator = tqdm.tqdm(test_loader, initial=0,desc="Iter", disable=False)
        for i, image in enumerate(test_iterator):
                visual_result(image,"../output/Ori.jpg")
                """add noise"""
                noisy_images=add_gaussian_noise(image,0.5)
                visual_result(noisy_images,"../output/AddNoise.jpg")
                """run"""
                outputs = model(noisy_images.to(DEVICE))
                generated_images = (outputs + 1) / 2  
                visual_result(generated_images.to('cpu'),"../output/Predict.jpg")
                print()
                 
if __name__=="__main__":
    main()