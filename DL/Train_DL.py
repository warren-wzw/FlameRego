import os
import sys
from tracemalloc import start
import torch
import tqdm 
import torch.nn as nn
import torch.nn.functional as F
os.chdir(sys.path[0])
sys.path.append('..')
from model.DL import DL_3COV_RES_DPW_FC96,DL_MUL_COVRES_DPW_FC96
from torch.utils.data import (DataLoader)
from datetime import datetime
from model.utils import load_and_cache_withlabel,FireDataset,get_linear_schedule_with_warmup,PrintModelInfo
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
TF_ENABLE_ONEDNN_OPTS=0
Encoder_Num=6
BATCH_SIZE=30
EPOCH=200
LR=1e-5
TensorBoardStep=500
SAVE_MODEL='../output/output_model/DL/'
PRETRAINED_MODEL_PATH="../output/output_model/DL/DL_6CONV_DPW_FC96.pth"

"""dataset"""
train_type="train"
image_path_train=f"../dataset/fine_tune_images_GAN/{train_type}"
label_path_train=f"../dataset/fine_tune_images_GAN/label/{train_type}.json"
cached_file=f"../dataset/cache/{train_type}_dl.pt"
test_type="val"
image_path_val=f"../dataset/fine_tune_images_label/images/{test_type}"
label_path_val=f"../dataset/fine_tune_images_label/label/{test_type}.json"
cached_file_val=f"../dataset/cache/{test_type}_dl.pt"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    hidden_size=96/Encoder_Num
    model=DL_MUL_COVRES_DPW_FC96(hidden=hidden_size,num_encoders=Encoder_Num)
    #model=DL_3COV_RES_DPW_FC96()
    model.to(DEVICE)
    #PrintModelInfo(model)
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH),strict=False)
    dataloader_train=CreateDataloader(image_path_train,label_path_train,cached_file)
    dataloader_val=CreateDataloader(image_path_val,label_path_val,cached_file_val)
    total_steps = len(dataloader_train) * EPOCH
    """Loss """
    criterion = nn.CrossEntropyLoss()
    """optimizer"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    """ Train! """
    print("  ************************ Running training ***********************")
    print("  Num Epochs = ", EPOCH)
    print("  Batch size per node = ", BATCH_SIZE)
    print("  Num examples = ", dataloader_train.sampler.data_source.num_instances)
    print(f"  Pretrained Model is {PRETRAINED_MODEL_PATH}")
    print(f"  Save Model as {SAVE_MODEL}")
    print("  ****************************************************************")
    model.train()
    torch.cuda.empty_cache()
    scheduler = get_linear_schedule_with_warmup(optimizer, 0.1 * total_steps , total_steps)
    tb_writer = SummaryWriter(log_dir='../output/tflog/') 
    best_accuarcy=0
    start_time=datetime.now()
    for epoch_index in range(EPOCH):
        loss_sum=0
        train_iterator = tqdm.tqdm(dataloader_train, initial=0,desc="Iter", disable=False)
        sum_test_accuarcy=0
        for step, batch in enumerate(train_iterator):
            image,status= batch
            image=image.to(DEVICE)
            status=status.to(DEVICE)
            optimizer.zero_grad()
            output=model(image)
            pred=torch.argmax(output, dim=-1)
            accuracy = ((pred == status).sum().item())/pred.shape[0]
            output=F.log_softmax(output, dim=1)
            loss = criterion(output, status)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            current_lr= scheduler.get_last_lr()[0]
            global_step=global_step+1
            loss_sum=loss_sum+loss.item()
            sum_test_accuarcy=sum_test_accuarcy+accuracy
            """ tensorbooard """
            if  global_step % TensorBoardStep== 0 and tb_writer is not None:
                tb_writer.add_scalar('train/lr', current_lr, global_step=global_step)
                tb_writer.add_scalar('train/loss', loss.item(), global_step=global_step)
            scheduler.step()
            train_iterator.set_description('Epoch=%d, Acc= %3.3f %%,loss=%.6f, lr=%9.7f' 
                                           % (epoch_index,(sum_test_accuarcy/(step+1))*100, loss_sum/(step+1), current_lr))
        """ validation """
        sum_accuarcy=0
        model.eval()
        with torch.no_grad():
            validation_iterator = tqdm.tqdm(dataloader_val, initial=0,desc="Iter", disable=False)
            for i, batch in enumerate(validation_iterator):
                image,status= batch
                image=image.to(DEVICE)
                status=status.to(DEVICE) 
                output=model(image)
                pred=torch.argmax(output, dim=-1)
                accuracy = ((pred == status).sum().item())/pred.shape[0]
                sum_accuarcy=sum_accuarcy+ accuracy
                validation_iterator.set_description('ValAcc= %3.3f %%' % (sum_accuarcy*100/(i+1)))
        
        if sum_accuarcy/(i+1) > best_accuarcy:
            best_accuarcy = sum_accuarcy/(i+1)
            if not os.path.exists(SAVE_MODEL):
                os.makedirs(SAVE_MODEL)
            torch.save(model.state_dict(), os.path.join(SAVE_MODEL,f"DL_{Encoder_Num}CONV_DPW_FC96.pth"))
            print("->Saving model checkpoint {} at {}".format(SAVE_MODEL+f"DL_{Encoder_Num}CONV_DPW_FC96.pth", 
                                                                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    end_time=datetime.now()
    print("Training consume :",(end_time-start_time)/60,"minutes")
    
if __name__=="__main__":
    main()