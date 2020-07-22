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
  - TensorRT

## Create TRT Engine from ONNX Model
First you need to have TensorRT installed on your machine. If you are working on a jetson, it comes pre-built with the Jetpack SDK. If not, follow installation instructions here https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

### Building trtexec - Command Line Program

trtexec has two main functionalities; benchmarking networks on random data, and building engines for inferencing on DLA's and GPU's. Ability to run models on DLA's is a big deal because it frees up your GPU which can then be used for vision related tasks that are not Deep Learning - eg. optical flow. In addition to practicality, trtexec allows us to load generated engines and test them with different parameters (like # of sterams) and benchmark them with other engines. We can also pull engines into programs which already run inference, and with multiple execution contexts, we can perform parallel inferencing - imagine running several GPU threads, and one on each DLA! Really fun tool for computer vision developers.

Once you have TensorRT installed, go to the trtexec sample folder and build the tool - simple as that. 
```
$ cd cd /usr/src/tensorrt/samples/trtexec 
$ sudo make
```
Compilation will prompt you to copy all the python example files to the bin folder of TensorRT, go ahead and do that.
```
$ cp *.py ../../bin/
```
Test out the tool.
```
$ cd ../../bin
$ ./trtexec -h
```
You should see a list of optional arguments. Yay done! Now for fun part, lets generate an inferencing engine.

### Generating an Inferencing Engine
An example;
```
$ /usr/src/tensorrt/bin/./trtexec --onnx=test1.onnx --explicitBatch --saveEngine=Yolov4_DLA1_int8.trt --useDLACore=1 --workspace=1000 --int8 --allowGPUFallback
```
- `--onnx`             path of onnx model
- `--explicitBatch`    Use explicit batch sizes when building the engine (default = implicit)
-  `--saveEngine`      path of trt file - where to save engine
- `--useDLACore`       which DLA core to use (0 or 1)
- `--workspace`        size of memory allowed (in MB) to be used during inferencing - you may need more
- `--fp16`             enable floating point 16 precision
- `allowGPUFallback`   not every operation is supported on the DLA's. You must have this to be able to support every operation

See https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/trtexec for more information about arguments and usage.

## Credit
- Initial YoloV4 PyTorch Implementation from https://github.com/Tianxiaomo/pytorch-YOLOv4
- Dataset from https://public.roboflow.ai/object-detection/raccoon
