import os
import sys
import torch
import tqdm
import numpy as np
import joblib
import time
sys.path.append('..')
from model.DAE_ML import DAE,GPC,KSVM,RandomForest,LSVM,LogisticReg,LightGBMClassifier,KNN
from torch.utils.data import (DataLoader)
from model.utils import PrintModelInfo,load_and_cache_withlabel,FireDataset
from sklearnex import patch_sklearn
patch_sklearn()
os.chdir(sys.path[0])

BATCH_SIZE=200
MODEL="RandomForest"
SAVE_MODEL='../output/output_model/gpc_model.pkl'
"""dataset"""
train_type="train"
image_path_train=f"../dataset/fine_tune_images_GPC/images/{train_type}"
label_path_train=f"../dataset/fine_tune_images_GPC/label/{train_type}.json"
cached_file=f"../dataset/cache/{train_type}_gpc.pt"
test_type="val"
image_path_test=f"../dataset/fine_tune_images_GPC/images/{test_type}"
label_path_test=f"../dataset/fine_tune_images_GPC/label/{test_type}.json"
cached_file_test=f"../dataset/cache/{test_type}_gpc.pt"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def CreateDataloader(image_path,label_path,cached_file):
    features = load_and_cache_withlabel(image_path,label_path,cached_file,shuffle=True)  
    num_features = len(features)
    num_train = int(0.1* num_features)
    train_features = features[:num_train]
    dataset = FireDataset(features=train_features,num_instances=num_train)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def main():
    model=DAE()
    model.load_state_dict(torch.load("../output/output_model/DAE_GAN/DaeModel_mse_ssim_wgan.pth"),strict=False)
    model.to(DEVICE)
    #PrintModelInfo(model)
    dataloader_train=CreateDataloader(image_path_train,label_path_train,cached_file)
    dataloader_val=CreateDataloader(image_path_test,label_path_test,cached_file_test)
    model.eval()
    train_iterator = tqdm.tqdm(dataloader_train, initial=0,desc="Iter", disable=False)
    feature_status=[]
    with torch.no_grad():
        for step, batch in enumerate(train_iterator):
            image,status= batch
            image=image.to(DEVICE)
            status=status.to(DEVICE)
            output=model(image)
            for i in range(output.shape[0]):
                feature={
                    "feature": output[i].reshape(-1).cpu().numpy(),  
                    "status": status[i].cpu().numpy()    
                }
                feature_status.append(feature)
    "ML"
    features = [item["feature"] for item in feature_status]
    statuses = [item["status"] for item in feature_status]
    features = np.array(features)
    print(f"----->train {MODEL}")
    #classifier = GPC()
    #classifier=KSVM()
    #classifier=LogisticReg()
    classifier=RandomForest()
    #classifier=LightGBMClassifier()
    #classifier=KNN()
    start_time = time.time() 
    classifier.fit(features, statuses)
    end_time = time.time()
    print("----->Training Consume", (end_time-start_time), "seconds")
    joblib.dump(classifier, f'../output/output_model/CLASSIFIER/{MODEL}_model.pkl')
    
    "validate model"
    test_iterator = tqdm.tqdm(dataloader_val, initial=0,desc="Iter", disable=False)
    feature_status_test=[]
    with torch.no_grad():
        for step, batch in enumerate(test_iterator):
            image,status= batch
            image=image.to(DEVICE)
            status=status.to(DEVICE)
            output=model(image)
            for i in range(output.shape[0]):
                feature={
                    "feature": output[i].reshape(-1).cpu().numpy(),  
                    "status": status[i].cpu().numpy()    
                }
                feature_status_test.append(feature)
    features_test = [item["feature"] for item in feature_status_test]
    statuses_test = [item["status"] for item in feature_status_test]
    start_time = time.time() 
    test_pred = classifier.predict(features_test)
    end_time = time.time()
    print("----->Validating Consume", (end_time-start_time), "seconds")
    accuracy = (test_pred == statuses_test).mean()
    print(f'Testing Accuracy: {accuracy * 100:.2f}%')
        
if __name__=="__main__":
    main()