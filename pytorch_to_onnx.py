import torch
from models import *

def export_onnx_model(model, input_shape, onnx_path, input_names=None, output_names=None, dynamic_axes=None):
    inputs = torch.ones(*input_shape)
    model(inputs)
    torch.onnx.export(model, inputs, onnx_path, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)

def load_model_weight(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    return model


if __name__=="__main__":
    model_path = "/home/mostafathereal/Desktop/Hello_Raccoon/Yolov4_epoch51.pth"
    model = Yolov4(n_classes=1)
    model = load_model_weight(model, model_path)
    # for name, v in model.named_parameters():
    #     print(name)
    input_shape = (1, 3, 416, 416)
    onnx_path = "Hello_raccoon.onnx"
    export_onnx_model(model, input_shape, onnx_path)
