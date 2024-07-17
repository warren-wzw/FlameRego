from cgi import test
import os
import sys
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt 
os.chdir(sys.path[0]) 
sys.path.append('..')
from model.DAE_ML import DAE
from torch.utils.data import (DataLoader)
from model.utils import PrintModelInfo,load_and_cache_withlabel,FireDataset
from sklearnex import patch_sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import KMeans,DBSCAN
from mpl_toolkits.mplot3d import Axes3D
patch_sklearn()
BATCH_SIZE=100
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_type="train"
image_path_test=f"../dataset/fine_tune_images_GPC/images/{test_type}"
label_path_test=f"../dataset/fine_tune_images_GPC/label/{test_type}.json"
cached_file_test=f"../dataset/cache/{test_type}_dl.pt"
VISUAL=2

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
    features = [item["feature"] for item in feature_status_test]
    labels = [item["status"] for item in feature_status_test]
    features = np.array(features)
    
    tsne = TSNE(n_components=VISUAL)
    features_low = tsne.fit_transform(features)
    # pca = PCA(n_components=8)  # 将数据降维到 50 维（可调整）
    # features_low = pca.fit_transform(features)
    # reducer = umap.UMAP(n_components=2)  # 降维到2维
    # features_low = reducer.fit_transform(features)
    kmeans = KMeans(n_clusters=63)  
    cluster_labels = kmeans.fit_predict(features_low)
    if VISUAL==2:
        """2D可视化结果"""
        plt.figure(figsize=(8, 6))
        plt.scatter(features_low[:, 0], features_low[:, 1], c=cluster_labels,  cmap='tab20c', alpha=0.7)
        plt.title('t-SNE Visualization of Test Features')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.colorbar(label='Status')
        plt.grid(True)
        plt.show()
    elif VISUAL==3:
        """3D可视化"""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(features_low[:, 0], features_low[:, 1], features_low[:, 2], c=cluster_labels,cmap='tab20c', alpha=0.7)
        plt.title('t-SNE Visualization of Train Features')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_zlabel('t-SNE Dimension 3')
        plt.colorbar(sc, label='Status')
        plt.grid(True)
        plt.show()
    
if __name__=="__main__":
    main()