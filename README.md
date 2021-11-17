# Face Mask Detector

A face mask detector built with Deep Learning and OpenCV

## About this project
Objectives of this project are to build a detector to detect whether a person is wearing a mask and further evaluate the performance of this detector. 

This detector is built using PyTorch, transfer learning is implemented with model from [FaceBoxes: A CPU Real-time Face Detector with High Accuracy](https://github.com/zisianw/FaceBoxes.PyTorch) and evaluation of the model performance is performed using [mAP (mean Average Precision)](https://github.com/Cartucho/mAP)   

## Requirements

Install: 

```bash 
pip install pytorch 
pip install opencv-python
```
Python 3.8.5, PyTorch v1.5.0 and OpenCV v4.0.1 are used. 

To do transfer learning, pre-trained model can be downloaded [here](https://drive.google.com/file/d/1tRVwOlu0QtjvADQ2H7vqrRwsWEmaqioI/) and place this in pretrained_model/ folder. This pretrained-model is shared by  authors of FaceBoxes in their repo. 


## Datasets 

Annotated datasets can be downloaded from [AIZOOTech](https://github.com/AIZOOTech/FaceMaskDetection), data are downloadable in both GoogleDrive and BaiduDisk. 

## Training
Navigate to face_mask_detector/ then run: 
```bash 
python train.py
```

## Evaluation 
To evaluate the trained model, run: 

```bash 
python evaluate_detector.py 
```
## Performance of model
The model is evaluated using mAP evaluation metrics 
![alt text](https://raw.githubusercontent.com/Jiaqi0602/face_mask_detector/main/evaluation/output/mAP.png) 
