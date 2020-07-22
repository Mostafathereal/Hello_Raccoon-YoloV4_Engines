import torch
import sys
import onnx
import os
import argparse
import numpy as np
from models import Yolov4

def export_onnx_model(onnx_path, batches, num_classes, input_h, input_w):
    model = load_model_weight(onnx_path, num_classes)
    inputs = torch.randn((batches, 3, input_h, input_w), requires_grad=True)
    print("------------exporting")

    onnx_name = "Yolov4_{}batches_{}classes.onnx".format(batches, num_classes)

    torch.onnx.export(model, inputs, onnx_name, export_params=True, opset_version=11, do_constant_folding=True, input_names=['input'], output_names=['boxes', 'confs'], dynamic_axes=None)    
    print("------------done exporting")

def load_model_weight(model_path, num_classes):
    model = Yolov4(n_classes=num_classes) 
    print("------------loading model")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    print("------------done loading")
    return model



if __name__=="__main__":
    
    if (len(sys.argv) != 2):
        print("Usage: pytorch_to_onnx.py <model_weights_path>")
        sys.exit()
    model_path = sys.argv[1]
    
    export_onnx_model(model_path, 1, 1, 416, 416)


