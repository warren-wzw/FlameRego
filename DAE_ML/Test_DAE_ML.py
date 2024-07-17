import os
import sys
import torch
import tqdm
import time
import numpy as np
os.chdir(sys.path[0])
sys.path.append('..')
from model.DAE_ML import DAE
from torch.utils.data import (DataLoader)
sys.path.append('..')
from model.utils import PrintModelInfo,load_and_cache_withlabel,FireDataset
from sklearnex import patch_sklearn
from joblib import load
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
patch_sklearn()
os.chdir(sys.path[0])

BATCH_SIZE=200
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path_test="../dataset/fine_tune_images_GPC/images/test"
label_path_test="../dataset/fine_tune_images_GPC/label/test.json"
cached_file_test="../dataset/cache/test_gpc.pt"
CLASSIFIER_MODEL="../output/output_model/CLASSIFIER/RandomForest_model.pkl"

def CreateDataloader(image_path,label_path,cached_file):
    features = load_and_cache_withlabel(image_path,label_path,cached_file,shuffle=True)  
    num_features = len(features)
    num_train = int(1 * num_features)
    train_features = features[:num_train]
    dataset = FireDataset(features=train_features,num_instances=num_train)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def main():
    model=DAE()
    model.load_state_dict(torch.load("../output/output_model/DAE_GAN/DaeModel_mse_ssim_wgan.pth"),strict=False)
    model.to(DEVICE)
    dataloader_test=CreateDataloader(image_path_test,label_path_test,cached_file_test)
    test_iterator = tqdm.tqdm(dataloader_test, initial=0,desc="Iter", disable=False)
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
    classifier=load(CLASSIFIER_MODEL)
    start_time = time.time() 
    test_pred = classifier.predict_proba(features_test)
    #test_pred = classifier.predict(features_test)
    end_time = time.time()
    
    all_labels = np.array(statuses_test)
    all_probs = np.array(test_pred)
    num_classes=60
    all_labels_one_hot = label_binarize(all_labels, classes=np.arange(num_classes))
    ap_per_class = []
    for i in range(num_classes):
        ap = average_precision_score(all_labels_one_hot[:, i], all_probs[:, i])
        ap_per_class.append(ap)
    mAP = np.mean(ap_per_class)
    print("mAP is",mAP)
    
    print("----->Testing Consume", (end_time-start_time), "seconds")
    accuracy = (test_pred == statuses_test).mean()
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
if __name__=="__main__":
    main()