import os
import sys
os.chdir(sys.path[0])
import onnx 
import torch
import onnxruntime
import numpy as np
sys.path.append('../../')
from model.utils import PrintModelInfo,preprocess_image
ONNX_MODEL="./flame_detect.onnx"

def main():
    input=preprocess_image('../test_images/8_6_1.jpg')
    input=input.unsqueeze(0)
    input_array=np.array(input)

    onnx_model = onnxruntime.InferenceSession(ONNX_MODEL)
    input_name = onnx_model.get_inputs()[0].name
    out = onnx_model.run(None, {input_name:input_array})
    pred=np.argmax(out, axis=-1)
    print(pred)

if __name__=="__main__":
    main()