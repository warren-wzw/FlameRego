import os
import sys
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
os.chdir(sys.path[0])
from torch.utils.data import (DataLoader)
from datetime import datetime
sys.path.append("..")
from model.cGAN import Generator,Discriminator
from model.utils import FireDataset,load_and_cache_withlabel,get_linear_schedule_with_warmup,PrintModelInfo,visual_result
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
TF_ENABLE_ONEDNN_OPTS=0
BATCH_SIZE=600
EPOCH=2000
LR=1e-5
PRETRAINED_MODEL_PATH="../output/output_model/cGAN/cGAN_G_v2.pth"
TensorBoardStep=500
SAVE_MODEL='../output/output_model/cGAN/'

"""dataset"""
train_type="train"
image_path_train=f"../dataset/fine_tune_images_label/images/{train_type}"
label_path_train=f"../dataset/fine_tune_images_label/label/{train_type}.json"
cached_file=f"../dataset/cache/{train_type}_cgan.pt"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gradient_penalty(critic, real_data, fake_data, labels, device):
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolates = epsilon * real_data + (1 - epsilon) * fake_data
    interpolates.requires_grad_(True)
    d_interpolates = critic(interpolates, labels)  
    grads = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(d_interpolates.size()).to(device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

    grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty

def CreateDataloader(image_path,label_path,cached_file):
    features = load_and_cache_withlabel(image_path,label_path,cached_file,shuffle=True)  
    num_features = len(features)
    num_train = int(1* num_features)
    train_features = features[:num_train]
    dataset = FireDataset(features=train_features,num_instances=num_train)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def main():
    global_step=0
    img_dim = (3, 128, 128)  
    c_dim=63
    z_dim=100
    model_G=Generator(z_dim=z_dim,c_dim=128).to(DEVICE)
    model_D=Discriminator(c_dim,img_dim).to(DEVICE)
    model_G.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    PrintModelInfo(model_G)
    print()
    PrintModelInfo(model_D)
    dataloader_train=CreateDataloader(image_path_train,label_path_train,cached_file)
    total_steps = len(dataloader_train) * EPOCH
    """optimizer"""
    optimizer_G = torch.optim.RMSprop(model_G.parameters(), lr=LR)
    optimizer_D = torch.optim.RMSprop(model_D.parameters(), lr=LR*8)
    """ Train! """
    print("  ************************ Running training ***********************")
    print("  Num Epochs = ", EPOCH)
    print("  Batch size per node = ", BATCH_SIZE)
    print("  Num examples = ", dataloader_train.sampler.data_source.num_instances)
    print(f"  Pretrained Model is {PRETRAINED_MODEL_PATH}")
    print(f"  Save Model as {SAVE_MODEL}")
    print("  ****************************************************************")
    model_G.train()
    model_D.train()
    scheduler_G = get_linear_schedule_with_warmup(optimizer_G, 0.1 * total_steps , total_steps)
    scheduler_D = get_linear_schedule_with_warmup(optimizer_D, 0.1 * total_steps , total_steps)
    tb_writer = SummaryWriter(log_dir='../output/tflog/')
    for epoch_index in range(EPOCH):
        loss_sumd=0
        loss_sumg=0
        torch.cuda.empty_cache()
        train_iterator = tqdm.tqdm(dataloader_train, initial=0,desc="Iter", disable=False)
        for step, (image,label) in enumerate(train_iterator):
            image,label= image.to(DEVICE),label.to(DEVICE)
            #visual_result(image,"output.jpg")
            z = torch.randn(len(image), z_dim, device=DEVICE)
            gen_labels = torch.randint(0, 63, (len(image),), device=DEVICE)
            """train model_D"""
            optimizer_D.zero_grad()
            fake_image=model_G(z, gen_labels).detach()
            real_validity = model_D(image, label)
            fake_validity = model_D(fake_image, gen_labels)
            gp=gradient_penalty(model_D, image, fake_image,label,DEVICE)
            d_loss = - torch.mean(real_validity)+torch.mean(fake_validity)+10*gp
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_D.parameters(), max_norm=2.0)  # 可以调整
            optimizer_D.step()
            for p in model_D.parameters():
                p.data.clamp_(-0.01, 0.01)
                
            """train G model"""
            optimizer_G.zero_grad()
            gen_imgs = model_G(z, gen_labels)
            fake_validity = model_D(gen_imgs, gen_labels)
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_G.parameters(), max_norm=2.0)  # 可以调整
            optimizer_G.step()
            """training detail"""    
            current_lr_G= scheduler_G.get_last_lr()[0]
            current_lr_R= scheduler_D.get_last_lr()[0]
            scheduler_G.step()
            scheduler_D.step()
            loss_sumg=loss_sumg+g_loss.item()
            loss_sumd=loss_sumd+d_loss.item()
            train_iterator.set_description('Epoch=%d, loss_G=%.6f, loss_D=%.6f, lr_G=%9.7f,lr_D=%9.7f' % (
                epoch_index, loss_sumg/(step+1), loss_sumd/(step+1),current_lr_G,current_lr_R))
            """ tensorboard """
            if  global_step % TensorBoardStep== 0 and tb_writer is not None:
                tb_writer.add_scalar('train_G/lr', scheduler_G.get_last_lr()[0], global_step=global_step)
                tb_writer.add_scalar('train_G/loss', g_loss.item(), global_step=global_step)
            if  global_step % TensorBoardStep== 0 and tb_writer is not None:
                tb_writer.add_scalar('train_D/lr', scheduler_D.get_last_lr()[0], global_step=global_step)
                tb_writer.add_scalar('train_D/loss', d_loss.item(), global_step=global_step)
            global_step+=1
        
        if loss_sumg/(step+1)<0:
            SIZE=3*3
            model_G.eval()
            with torch.no_grad():
                z = torch.randn(SIZE, z_dim, device=DEVICE)
                #labels = torch.arange(0, c_dim, device=DEVICE).repeat(2)[:SIZE]
                labels = torch.randint(0, 63, (len(image),), device=DEVICE)[:SIZE]
                gen_imgs = model_G(z, labels).detach().cpu()
            fig, axs = plt.subplots(int(SIZE**0.5),int(SIZE**0.5), figsize=(10, 10))
            for i, ax in enumerate(axs.flatten()):
                ax.imshow(gen_imgs[i].permute(1, 2, 0) * 0.5 + 0.5)   
                ax.axis('off')
                ax.axis('off')
                # 添加文字（如标签）到图像上
                label_text = f'Label: {labels[i].item()}'
                ax.text(0.5, -0.1, label_text, fontsize=12, ha='center', transform=ax.transAxes)
            plt.tight_layout()
            plt.savefig(f'../output/output_images/best_loss.png')   
            plt.close(fig)
                
        if ((epoch_index+1) % 10)==0:  
            SIZE=3*3
            model_G.eval()
            with torch.no_grad():
                z = torch.randn(SIZE, z_dim, device=DEVICE)
                labels = torch.randint(0, 63, (len(image),), device=DEVICE)[:SIZE]
                gen_imgs = model_G(z, labels).detach().cpu()
            fig, axs = plt.subplots(int(SIZE**0.5),int(SIZE**0.5), figsize=(10, 10))
            for i, ax in enumerate(axs.flatten()):
                ax.imshow(gen_imgs[i].permute(1, 2, 0) * 0.5 + 0.5)   
                ax.axis('off')
                # 添加文字（如标签）到图像上
                label_text = f'Label: {labels[i].item()}'
                ax.text(0.5, -0.1, label_text, fontsize=12, ha='center', transform=ax.transAxes)
            plt.tight_layout()
            plt.savefig(f'../output/output_images/generated_images_epoch_{epoch_index}.png')   
            plt.close(fig)
              
        if ((epoch_index+1) % 50)==0:
            if not os.path.exists(SAVE_MODEL):
                os.makedirs(SAVE_MODEL)
            torch.save(model_G.state_dict(), os.path.join(SAVE_MODEL,"cGAN_G.pth"))
            print("--->Saving model checkpoint {} at {}".format(SAVE_MODEL+"cGAN_G.pth", 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"))) 
        torch.cuda.empty_cache() 
        
if __name__ == "__main__":
    main()