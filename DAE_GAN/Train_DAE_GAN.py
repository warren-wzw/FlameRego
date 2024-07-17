import os
import sys
import tqdm
import torch
import torch.nn as nn
os.chdir(sys.path[0])
from datetime import datetime
from torch.utils.data import (DataLoader)
sys.path.append('..')
from model.DAE_GAN import DAE,REGO
from model.utils import FireDatasetGan,SSIMLoss
from model.utils import load_and_cache,add_gaussian_noise,get_linear_schedule_with_warmup,compute_gradient_penalty,save_model
from torch.optim.lr_scheduler import LambdaLR
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

BATCH_SIZE=100
EPOCH=150
LR=1e-5
TensorBoardStep=500
noise_ratio=0.2
TRAIN_IMAGE_PATH="../dataset/fine_tune_images_GAN/train"
VAL_IMAGE_PATH="../dataset/fine_tune_images_GAN/val"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED_MODEL_PATH=""
SAVE_MODEL="../output/output_model/"
train_cache="../dataset/cache/train_cache_fine_tune.pt"
val_cache="../dataset/cache/val_cache_fine_tune.pt"

def CreateDataloader(data_file,cached_file):
    features = load_and_cache(data_file,cached_file,shuffle=True)    
    dataset = FireDatasetGan(features=features,num_instances=len(features))
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def main():
    global_step=0
    model_G=DAE().to(DEVICE)
    model_R=REGO().to(DEVICE)
    model_G.load_state_dict(torch.load("../output/output_model/DaeModel_mse_ssim_wgan.pth"))
    model_G.load_state_dict(torch.load("../output/output_model/R_Model_fine_tune.pth"))
    #PrintModelInfo(model_G)
    train_loader = CreateDataloader(TRAIN_IMAGE_PATH,train_cache)
    val_loader = CreateDataloader(VAL_IMAGE_PATH,val_cache)
    """Loss """
    criterion_mse = nn.MSELoss()
    criterion_ssim=SSIMLoss()
    """optimizer"""
    optimizer_G = torch.optim.RMSprop(model_G.parameters(), lr=LR)
    optimizer_R = torch.optim.RMSprop(model_R.parameters(), lr=LR)
    """ Train! """
    print("  ************************ Running training ***********************")
    print("  Num Epochs = ", EPOCH)
    print("  Batch size per node = ", BATCH_SIZE)
    print("  Num examples = ", train_loader.sampler.data_source.num_instances)
    print(f"  Pretrained Model is {PRETRAINED_MODEL_PATH}")
    print(f"  Save Model as {SAVE_MODEL}")
    print("  ****************************************************************")
    model_G.train()
    model_R.train()
    torch.cuda.empty_cache()
    total_steps = len(train_loader) * EPOCH
    best_val_loss_G=100
    best_val_loss_R=100
    scheduler_G = get_linear_schedule_with_warmup(optimizer_G, 0.1 * total_steps , total_steps)
    scheduler_R = get_linear_schedule_with_warmup(optimizer_R, 0.1 * total_steps , total_steps)
    tb_writer = SummaryWriter(log_dir='../output/tflog/') 
    for epoch_index in range(EPOCH):
        sum_loss_G=0
        sum_loss_R=0 
        sample_num=0
        train_iterator = tqdm.tqdm(train_loader, initial=0,desc="Iter (loss=X.XXX, lr=X.XXXXXXX)", disable=False)
        for i, image in enumerate(train_iterator):
            image=image.to(DEVICE)
            """train G model"""
            optimizer_G.zero_grad()
            noisy_images=add_gaussian_noise(image,noise_ratio).to(DEVICE)
            outputs = model_G(noisy_images)
            """adversarial loss"""
            fake_output = model_R(outputs)
            adversarial_loss = -torch.mean(fake_output)
            loss_mse = criterion_mse(outputs, image)
            loss_ssim = criterion_ssim(outputs, image)  
            loss_G=loss_mse+adversarial_loss+loss_ssim
            loss_G.backward()
            optimizer_G.step()
            model_G.zero_grad()
            """train R model"""
            optimizer_R.zero_grad()
            real_output = model_R(image) 
            fake_output = model_R(outputs.detach())
            gradient_penalty = compute_gradient_penalty(model_R, image, outputs.detach())
            loss_R = -torch.mean(real_output) + torch.mean(fake_output)+10 * gradient_penalty    
            loss_R.backward()
            torch.nn.utils.clip_grad_norm_(model_R.parameters(), 2.0) 
            optimizer_R.step()
            model_R.zero_grad()
            """ tensorbooard """
            if  global_step % TensorBoardStep== 0 and tb_writer is not None:
                tb_writer.add_scalar('train_G/lr', scheduler_G.get_last_lr()[0], global_step=global_step)
                tb_writer.add_scalar('train_G/loss', loss_G.item(), global_step=global_step)
            if  global_step % TensorBoardStep== 0 and tb_writer is not None:
                tb_writer.add_scalar('train_R/lr', scheduler_R.get_last_lr()[0], global_step=global_step)
                tb_writer.add_scalar('train_R/loss', loss_R.item(), global_step=global_step)
            global_step+=1
            
            if image.shape[0]==BATCH_SIZE:
                sample_num=sample_num+(i+1)*BATCH_SIZE
            else:
                sample_num=sample_num+image.shape[0]
            current_lr_G= scheduler_G.get_last_lr()[0]
            current_lr_R= scheduler_R.get_last_lr()[0]
            scheduler_G.step()
            scheduler_R.step()
            train_iterator.set_description('Epoch=%d, loss_G=%.6f, loss_R=%.6f, lr_G=%9.7f,lr_R=%9.7f' % (
                epoch_index, loss_G.item(), loss_R.item(),current_lr_G,current_lr_R))
            sum_loss_R=sum_loss_R+loss_R.item()
            sum_loss_G=sum_loss_G+loss_G.item()
        # print("G Average loss is ",sum_loss_G/(train_loader.sampler.data_source.num_instances+1e-5))
        # print("R Average loss is ",sum_loss_R/(train_loader.sampler.data_source.num_instances+1e-5))
        
        """ validation """
        val_sum_loss_G=0
        val_sum_loss_R=0
        loss_val=0
        model_G.eval()
        with torch.no_grad():
            validation_iterator = tqdm.tqdm(val_loader, initial=0,desc="Iter (loss=X.XXX, lr=X.XXXXXXX)", disable=False)
            for i, image in enumerate(validation_iterator):
                """add noise"""
                noisy_images_vl=add_gaussian_noise(image,noise_ratio).to(DEVICE)
                outputs_val = model_G(noisy_images_vl)
                fake_output = model_R(outputs_val)
                adversarial_loss = -torch.mean(fake_output)
                loss_mse = criterion_mse(outputs_val, image.to(DEVICE))
                loss_ssim = criterion_ssim(outputs_val, image.to(DEVICE))
                loss_val=loss_mse+adversarial_loss+loss_ssim
                val_sum_loss_G=val_sum_loss_G+loss_val.item()
                """R"""
                real_output = model_R(image.to(DEVICE)) 
                fake_output = model_R(outputs_val.detach())
                loss_R = -torch.mean(real_output) + torch.mean(fake_output)
                val_sum_loss_R=val_sum_loss_R+loss_R.item()
            print("G_Validation loss is ",val_sum_loss_G/(i+1),"R_Validation loss is ",val_sum_loss_R/(i+1))
        """save model"""
        if val_sum_loss_G/(i+1) < best_val_loss_G:
            best_val_loss_G = val_sum_loss_G/(i+1)
            if not os.path.exists(SAVE_MODEL):
                os.makedirs(SAVE_MODEL)
            torch.save(model_G.state_dict(), os.path.join(SAVE_MODEL,"DaeModel.pth"))
            print("--------------------->Saving model checkpoint {} at {}".format(SAVE_MODEL+"DAE_model.pth", 
                                                                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        if val_sum_loss_R/(i+1) < best_val_loss_R:
            best_val_loss_R = val_sum_loss_R/(i+1)
            if not os.path.exists(SAVE_MODEL):
                os.makedirs(SAVE_MODEL)
            torch.save(model_G.state_dict(), os.path.join(SAVE_MODEL,"R_Model.pth"))
            print("--------------------->Saving model checkpoint {} at {}".format(SAVE_MODEL+"R_model.pth", 
                                                                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

if __name__=="__main__":
    main()