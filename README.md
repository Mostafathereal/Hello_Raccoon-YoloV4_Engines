# Hello_Raccoon

## From training YoloV4 to Inference on Nvidia Jetson DLA/GPU
- Data preprocessing and augmentation done using RoboFlow.ai
- YoloV4 (PyTorch) is trained on a raccoon dataset using Colab (bless you Google and your infinite cash pile) 
- PyTorch --> ONNX 
- ONNX model --> TensorRT Egnine
- running engines with on Nvidia Jetson Xavier NX to make use of those nice DLA's (Deep Learning Accelerators, for inference)

## Training on Google Colab
- Follow the "YoloV4_Custom_Train" notebook I created above 
- Data set for this project collected from Roboflow.ai 
  - They provide data preprossessing and augmentation services
- When training is complete, download the weights file - "Yolov4_epoch<latest epoch>".pth
  
## Converting PyTorch Model to ONNX
- 









### requirements
- pycuda
- 
