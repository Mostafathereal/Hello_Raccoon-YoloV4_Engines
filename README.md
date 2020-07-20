# Hello_Raccoon

## End-to-End solution to detecting raccoons in your trash 
- Data preprocessing and augmentation done using RoboFlow.ai
- YoloV4 (PyTorch) is trained on a raccoon dataset using Colab (bless you Google and your infinite cash pile) 
- PyTorch --> ONNX 
- Running model with TensorRT on Nvidia Jetson Xavier NX to make use of those nice DLA's (Deep Learning Accelerators, for inference)
