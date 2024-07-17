import os
import sys
import torch
import cv2
os.chdir(sys.path[0])
sys.path.append('..')
from model.DL import DL_3COV_RES_DPW_FC96
from model.utils import PrintModelInfo,preprocess_image,GradCAM,apply_colormap_on_image
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH="../output/output_model/DL/DL_3COV_RES_DPW_FC96.pth"
TEST_IMAGES="./test_images/"
         
def main():
    model=DL_3COV_RES_DPW_FC96()
    model.to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    # PrintModelInfo(model)
    grad_cam = GradCAM(model, target_layer="conv_pw3")
    files=os.listdir(TEST_IMAGES)
    model.eval()
    # with torch.no_grad():
    for i,file in enumerate(files):
        original_image = cv2.imread(TEST_IMAGES+file)
        image=preprocess_image(TEST_IMAGES+file)
        image=image.unsqueeze(0)
        image=image.to(DEVICE)
        image.requires_grad = True
        output=model(image)
        pred=torch.argmax(output, dim=-1).item()
        
        cam = grad_cam.generate_cam(image, pred)
        cam_image = apply_colormap_on_image(original_image, cam, 'jet')
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (60, 200)  # 文字的起始位置
        font_scale =1
        color = (0, 0, 255)  
        thickness = 1
        cv2.putText(original_image, f'Case: {pred}', org, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.imwrite(f"./predict_images/{file}",original_image)
        #cv2.imwrite(f"./predict_images/en3_{file}", cam_image)
        print(f"processe file {file},predict is {pred}")
        
if __name__=="__main__":
    main()