import os
import sys
import torch
os.chdir(sys.path[0])
sys.path.append('../../')
from model.DL import DL_MUL_COVRES_DPW_FC96
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH="../../output/output_model/DL/DL_4CONV_DPW_FC96.pth"
Encoder_Num=4
     
def main():
    hidden_size=96/Encoder_Num
    model=DL_MUL_COVRES_DPW_FC96(hidden=hidden_size,num_encoders=Encoder_Num)
    model.to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    input = torch.randn(1, 3, 256, 256, requires_grad=False).float().to(torch.device(DEVICE))
    torch.onnx.export(model,
                    input,
                    'flame_detect.onnx', # name of the exported onnx model
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=False)

if __name__=="__main__":
    main()