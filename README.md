# Hello_Raccoon

## From training YoloV4 to Inference on Nvidia Jetson DLA/GPU - Overview
- Data preprocessing and augmentation done using RoboFlow.ai
- YoloV4 (PyTorch) is trained on a raccoon dataset using Colab (bless you Google and your infinite cash pile) 
- PyTorch model is converted to ONNX Form
- ONNX model is used to then build a TensorRT Egnine
- running engines with on Nvidia Jetson Xavier NX - using GPU and DLA's 

## Training on Google Colab
- Follow the "YoloV4_Custom_Train" notebook I created above 
- Data set for this project collected from Roboflow.ai 
  - They provide data preprossessing and augmentation services
- Depending on the format of your dataset you may have to change the `get_item` method in "dataset.py"
- When training is complete, download the weights file - "Yolov4_epoch<latest epoch>".pth
  
## Converting PyTorch Model to ONNX

### Requirements
- Protobuf
- PyCuda
- ONNX
- TensorRT

Firstly, make sure you have the PyTorch class of your model. Then run;
```
$ python pytorch_to_onnx.py <path to model weights> 
```
Feel free to edit the script to parameterize any of the arguments used in the export command

## Create TRT Engine from ONNX Model
First you need to have TensorRT installed on your machine. If you are working on a jetson, it comes pre-built with the Jetpack SDK. If not, follow installation instructions here https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

### Building trtexec - Command Line Program

trtexec has two main functionalities; benchmarking networks on random data, and building engines for inferencing on DLA's and GPU's. Ability to run models on DLA's is a big deal because it frees up your GPU which can then be used for vision related tasks that are not Deep Learning - eg. optical flow. In addition to practicality, trtexec allows us to load generated engines and test them with different parameters (like # of sterams) and benchmark them with other engines. We can also pull engines into programs which already run inference, and with multiple execution contexts, we can perform parallel inferencing - imagine running several GPU threads, and one on each DLA! Really fun tool for computer vision developers.

Once you have TensorRT installed, go to the trtexec sample folder and build the tool - simple as that. 
```
$ cd /usr/src/tensorrt/samples/trtexec 
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
- `--onnx=`              path of onnx model
- `--explicitBatch`      Use explicit batch sizes when building the engine (default = implicit)
-  `--saveEngine=`       path of trt file - where to save engine
- `--useDLACore=`        which DLA core to use (0 or 1)
- `--workspace=`         size of memory allowed (in MB) to be used during inferencing - you may need more
- `--fp16`               enable floating point 16 precision
- `--allowGPUFallback`   not every operation is supported on the DLA's. You must have this to be able to support every operation
- `--device=`            Which GPU device to use (default is 0) - Jetson only has one GPU so this wouldnt make adifference for us
See https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/trtexec for more information about arguments and usage.

## Benchmarking
If we pipe the information that the `trtexec` command displays, we can compare the performances of different engines. Below is some of the benchmarks I've done myself. Everything here is generated with a batch size of 1 and a stream of 1.

| Precision | DLA | Throughput (QPS) |
|-----------|-----|------------------|
| int8      | No  |     61.71        |
| fp16      | No  |     36.77        |
| int8      | Yes |     31.23        |
| fp16      | Yes |     20.69        |

I'll be updating this table as I continue benchmarking and acheivebetter results.

### Possible Improvements
- Increase batch size
- Increase # streams
- Use compact version of model (Yolov4-tiny)
- Allocate more memory to the engine 
 
## Credit
- Initial YoloV4 PyTorch Implementation from https://github.com/Tianxiaomo/pytorch-YOLOv4
  - models.py file
- Dataset from https://public.roboflow.ai/object-detection/raccoon
