# Hello_Raccoon

## From training YoloV4 to Inference on Nvidia Jetson DLA/GPU - Overview
- Data preprocessing and augmentation done using RoboFlow.ai
- YoloV4 (PyTorch) is trained on a raccoon dataset using Colab (bless you Google and your infinite cash pile) 
- PyTorch model is converted to ONNX Form
- ONNX model is used to then build a TensorRT Egnine
- running engines with on Nvidia Jetson Xavier NX to make use of those DLA's (Deep Learning Accelerators, for inference)

## Training on Google Colab
- Follow the "YoloV4_Custom_Train" notebook I created above 
- Data set for this project collected from Roboflow.ai 
  - They provide data preprossessing and augmentation services
- Depending on the format of your dataset you may have to change the `get_item` method in "dataset.py"
- When training is complete, download the weights file - "Yolov4_epoch<latest epoch>".pth
  
## Converting PyTorch Model to ONNX

```
$ python pytorch_to_onnx.py <.pth file path>
```

  ### Requirements
  - Protobuf
  - PyCuda
  - ONNX
  - TensorRT - comes pre-built on Jetpack SDK







## Credit
- Initial YoloV4 PyTorch Implementation from https://github.com/Tianxiaomo/pytorch-YOLOv4
