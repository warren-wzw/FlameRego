import os
import sys
import torch
import cv2
from joblib import load
os.chdir(sys.path[0])
sys.path.append('..')
from model.DAE_ML import DAE
from model.utils import preprocess_image,GradCAM,apply_colormap_on_image
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_IMAGES="./test_images/"
CLASSIFIER_MODEL="../output/output_model/CLASSIFIER/RandomForest_model.pkl"

def main():
    model=DAE()
    model.to(DEVICE)
    model.load_state_dict(torch.load("../output/output_model/DAE_GAN/DaeModel_mse_ssim_wgan.pth"),strict=False)
    grad_cam = GradCAM(model, target_layer="conv6_en")
    files=os.listdir(TEST_IMAGES)
    model.eval()
    # with torch.no_grad():
    feature_status_test=[]
    for file in files:
        original_image = cv2.imread(TEST_IMAGES+file)
        image=preprocess_image(TEST_IMAGES+file)
        image=image.unsqueeze(0)
        image=image.to(DEVICE)
        image.requires_grad = True
        output=model(image)
        
        features_test = [item.reshape(-1).cpu().detach().numpy() for item in output]
        classifier=load(CLASSIFIER_MODEL)
        pred = classifier.predict(features_test)
        
      
if __name__=="__main__":
    main()