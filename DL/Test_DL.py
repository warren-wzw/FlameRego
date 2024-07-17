import os
from pickle import TRUE
import sys
import torch
import tqdm 
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
os.chdir(sys.path[0])
sys.path.append('..')
from model.DL import DL_3COV_FC96,DL_PW_FC63,DL_DSW_RES_FC100,DL_3COV_RES_DPW_FC96,DL_MUL_COVRES_DPW_FC96
from torch.utils.data import (DataLoader)
from datetime import  datetime
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from model.utils import load_and_cache_withlabel,FireDataset,PrintModelInfo
from sklearnex import patch_sklearn
from sklearn.metrics import recall_score,precision_score
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
patch_sklearn()

Encoder_Num=6
BATCH_SIZE=20
TEST_TYPE="TEST" #TEST、CLUSTER、MIXMATRIX、mAP
VISUAL=2
"""dataset"""
test_type="test"
image_path_val=f"../dataset/fine_tune_images_label/images/{test_type}"
label_path_val=f"../dataset/fine_tune_images_label/label/{test_type}.json"
cached_file_val=f"../dataset/cache/{test_type}_dl.pt"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#MODEL_PATH=f"../output/output_model/DL/DL_{Encoder_Num}CONV_DPW_FC96.pth"
MODEL_PATH=f"../output/output_model/DL/DL_3COV_FC96.pth"

def CreateDataloader(image_path,label_path,cached_file):
    features = load_and_cache_withlabel(image_path,label_path,cached_file,shuffle=True)  
    num_features = len(features)
    num_train = int(1* num_features)
    train_features = features[:num_train]
    dataset = FireDataset(features=train_features,num_instances=num_train)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def cluser_visualize(dataloader, model, device):
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        validation_iterator = tqdm.tqdm(dataloader, initial=0,desc="Iter", disable=False)
        for i, batch in enumerate(validation_iterator):
            images, statuses = batch
            images = images.to(device)
            statuses = statuses.to(device)
            # Get intermediate layer output (hidden4)
            intermediate_output = model.get_concatenated_features(images)
            features_list.append(intermediate_output.cpu().numpy())
            labels_list.append(statuses.cpu().numpy())

    features = np.concatenate(features_list)
    labels = np.concatenate(labels_list)
    
    features_flat = features.reshape(-1, 6144) 
    tsne = TSNE(n_components=VISUAL)
    reduced_features = tsne.fit_transform(features_flat)
    kmeans = KMeans(n_clusters=63)  
    cluster_labels = kmeans.fit_predict(reduced_features) 
    """2D visiually"""
    if VISUAL==2:
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='tab20c', alpha=0.7)
        plt.title('Visualization of Intermediate Features')
        plt.colorbar()
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 1')
        plt.show()
    elif VISUAL==3:
        """3D visiually"""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], c=cluster_labels, cmap='coolwarm')
        plt.title('t-SNE Visualization of Train Features')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_zlabel('t-SNE Dimension 3')
        plt.colorbar(sc, label='Status')
        plt.grid(True)
        plt.show()
    
def CaculateAcc(pred,status):
    pred=torch.argmax(pred, dim=-1)
    accuracy = ((pred == status).sum().item())/pred.shape[0]
    recall_per_class = recall_score(status.to('cpu'), pred.to('cpu'), average=None)
    precision_per_class = precision_score(status.to('cpu'), pred.to('cpu'), average=None)
    return accuracy,recall_per_class,precision_per_class

def model_test(model,dataloader_test):
    sum_accuracy=0
    total_samples=0
    model.eval()
    start_time=datetime.now()
    with torch.no_grad():
        test_iterator = tqdm.tqdm(dataloader_test, initial=0,desc="Iter", disable=False)
        for i, batch in enumerate(test_iterator):
            image,status= batch
            image=image.to(DEVICE)
            status=status.to(DEVICE) 
            output=model(image)
            accuracy,_,_=CaculateAcc(output,status)
            sum_accuracy += accuracy * image.size(0)  # accumulate correct predictions
            total_samples += image.size(0)  # accumulate total number of samples
    end_time=datetime.now()
    print("---->average accuracy is ",(sum_accuracy/total_samples)*100," %","Consum time :",end_time-start_time,"sencods")

def calculate_mAP(model,dataloader_test):
    all_labels = []
    all_probs = []
    model.eval()
    with torch.no_grad():
        test_iterator = tqdm.tqdm(dataloader_test, initial=0,desc="Iter", disable=False)
        for i, batch in enumerate(test_iterator):
            image, label = batch
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            probs = torch.softmax(output, dim=-1)
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    # Convert lists to numpy arrays for easier manipulation
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    num_classes=63
    all_labels_one_hot = label_binarize(all_labels, classes=np.arange(num_classes))
    ap_per_class = []
    for i in range(num_classes):
        ap = average_precision_score(all_labels_one_hot[:, i], all_probs[:, i])
        ap_per_class.append(ap)
    mAP = np.mean(ap_per_class)
    print("mAP is",mAP)

def mix_matrix(model,dataloader_test):
    all_preds = []
    all_labels = []
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader_test):
            image, label = batch
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            pred = torch.argmax(output, dim=-1)
            probs = torch.softmax(output, dim=-1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    # Convert lists to numpy arrays for easier manipulation
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    sum_per_class = cm.sum(axis=1, keepdims=True)
    cm_normalized = cm / sum_per_class.astype(float) * 100
    plt.figure(figsize=(24, 18))
    sns.heatmap(cm_normalized, annot=True, fmt='.0f', cmap='coolwarm', 
                xticklabels=np.unique(all_labels), yticklabels=np.unique(all_labels))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Normalized Confusion Matrix (%)')
    plt.show()
            
def main():
    hidden_size=96/Encoder_Num
    #model=DL_MUL_COVRES_DPW_FC96(hidden=hidden_size,num_encoders=Encoder_Num)
    model=DL_3COV_FC96()
    model.to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    #PrintModelInfo(model)
    dataloader_test=CreateDataloader(image_path_val,label_path_val,cached_file_val)
    if TEST_TYPE=="CLUSTER":
        cluser_visualize(dataloader_test,model,DEVICE)
        
    elif TEST_TYPE=="TEST":
        model_test(model,dataloader_test)
        
    elif TEST_TYPE=="MIXMATRIX":
        mix_matrix(model,dataloader_test)
    
    elif TEST_TYPE=="mAP":
        calculate_mAP(model,dataloader_test)

if __name__=="__main__":
    main()