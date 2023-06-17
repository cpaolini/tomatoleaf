# tomatoleaf
Tomato leaf disease inference using the Coral Edge TPU Dev Board
### Tomato Leaf Disease Object Detection Model using Coral Edge TPU Dev Board 
## Introduction
This project aims to detect 5 different tomato leaf diseases and 6 different class inlcuding Healthy leaf using a Coral Edge TPU Dev Board. As Google Coral Dev Board is resource scarce (in terms of using relatively low power) emmbedded device it has relatively small yet powerful processing capability using TPU (Tensor Processing Unit) with 8mb sram allowing us to perform real-time inference on tomato leaf images, possibly enabling early detection of diseases and potentially helping farmers and improving crop yield.

## Project Description
The tomato leaf disease object detection model is the pre-trained deep learning model, specifically designed for the detection of common tomato leaf diseases namely Bacterial Spot, Early Blight, Late Blight, Septoria Spot and Yellow Leaf Curl diseases. The model has been trained by transfer learning techniques using CPU trained coco 300x300 model which was trained on large dataset of different animal images, this particular model used for disease detection also used hundreds of annotated images (with corresponding bounding boxes) of different diseases and also healthy leaves. The model is trained on TF1 SSD Mobilenet V1 using WSL along with MobaXterm (WSL and MobaXterm are not actually required as it was just matter of personal preference). 

## Installation and Usage

In this model we used Coral Dev Board with 4GB ram, as it is ARM64 device it is no longer supported to compile models on device and it requires to be compiled either on PC or Cloud. 
Compiler version used for model was 16.0 and Edge TPU runtime 14 as it is the latest update as of writing this (June, 2023). 
In order to get started with the Dev Board and understand how to push files to mendel linux OS running on the board (including how to set up mendel itself) please refer to the following link:  [Get Started With Dev Board](https://coral.ai/docs/dev-board/get-started/)"Intro to Coral Dev Board")
. Generally there are several ways of accesing and downloading files: through MDT (Mendel Development Tool), SSH (Secure Shell Protocol, usually only one pc is allowed however it is possible to run openssh server and set up ftp communication as well by changing appropriate config values inside /etc/ssh/sshd_config) and via USB or internet. In order to install requirements on the board and be assured that it is capable of running please follow the instructions found [here, try PyCoral] (https://coral.ai/docs/dev-board/get-started/#run-pycoral, "PyCoral"), particularly 
mkdir coral && cd coral
git clone https://github.com/google-coral/pycoral.git
cd pycoral
bash examples/install_requirements.sh classify_image.py
python3 examples/classify_image.py \
--model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
--labels test_data/inat_bird_labels.txt \
--input test_data/parrot.jpg 
The following example is classification model and is differed from detection model as detection model includes boundig boxes of the object and respectively needs to be trained with bounding box data along with the labels, as well as different parameters and configuration. It is worth to mention that SSD Mobilenet is an architecture which can be used for both. 
To run the tomato leaf disease detection model download the following [tflite model] (https://github.com/cpaolini/tomatoleaf/blob/main/output_tflite_graph_edgetpu.tflite "Tomato Disease Detection TFlite Model") and also the [labels] (https://github.com/cpaolini/tomatoleaf/blob/main/labels2.txt "disease labels")
In order to detect image download or copy source code of [detect_image.py] (https://github.com/cpaolini/tomatoleaf/blob/main/detect_image.py)
Example of usage in the terminal of Mendel: 
python3 detect_image.py --model output_tflite_graph.tflite --labels labels2.txt --input Tomato_Late_blight-43.jpg --output lateblightbounded2.jpg 
Iff you want to run inference on multiple images located in one folder use the script [detectmany.py] (https://github.com/cpaolini/tomatoleaf/blob/main/detectmany.py)
Example of usage "inpfolder" is input folder containing pictures to run the inference on and inpout is the folder which will contain output images with bounding boxes: 
python3 detectmany.py --model output_tflite_graph_edgetpu.tflite --labels labels2.txt --input inpfolder/ --output inpout/ 

In order to use live inferecne model and stream and the same time you can use the following command: 

edgetpu_detect --model output_tflite_graph_edgetpu.tflite --labels labels2.txt --print


## File Description 

5D1H.csv - all labels and bounding box files for the thousands of disease (6 different classes of BacterialSpot, EarlyBlight, Healthy, LateBlight, SeptoriaSpot, YellowLeafCUrl) and healthy leaf labels in one csv files (CSV file was made manually after joining and filtering VGG anotator files)
generate_tfrecord_csv.py - Python Script to generate record files needed for training the model from the csv file
Syntax: python ../generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record 
method_david_change_csv.py - Python script specifically made for csv files like 5D1H in order to make it easier to create changed csv file involving specific pictures in the folder and also changing the original locations indicated in 5D1H file, which of course will be different for each different instance and each different computer/server/cloud 
Syntax: python method_david_change_csv.py "path of the csv file like 51DH.csv" "path of the new csv file which should be generated" "string to replace in csv file for new locations of files in generated new csv file" "folder location where those images are to include only those images and exclude all the other which are in csv file like 51DH.csv" "file type like .jpg" 
data200each.zip - images particurarly used for training the object detection model for leaf disease recognition 
data200eachtest.zip - same but images used for testing 
data200train.csv / data200test.csv - Appropriate csv files 
data200train.record / data200test.record - tensorflow record files which can be used at any platform and model as long as it is capable train tensorflow record files
output_tflite_graph.tflite - tflite generated after quantization (it still can be used for the model, but it won't be as accurate and as fast as edgetpu one) 
output_tflite_graph_edgetpu.tflite - tflite model which can successfully run on edge tpu


events.out.tfevents.1686813032.DJ3060 is the evaluation events file which can be opened with tensorboard (training events file was too large to upload in case of need contact me to get it) 
tflite_graph.pb (graph file associated with tomato disease trained model, pbtxt file was large so contact me to get it as well) 

# Training Tutorial Inside The Jupyter Notebook without completely depending on Docker or Colab:

Uninstalling current tf and installing tf1.15 with gpu support (sometimes it is problem since Google stopped supporting those versions, so if you do not use docker I strongly reccomend to first downgrade python to 3.7 then instaling tf 1.15 either via pip/pip3 or conda environment specifically made for this)  
 #! pip uninstall tensorflow -y
! pip install tensorflow-gpu==1.15
! pip install -U numpy==1.19.5
! pip install -U pycocotools==2.0.1

Check if tensorflow model was really installed with appropriate version: 

import tensorflow as tf
print(tf.__version__)

Download tensorlfow models repository 

! git clone https://github.com/tensorflow/models.git
 
Detach head from the original main of that repository

! cd models && git checkout f788046ca876a8820e05b0b48c1fc2e16b0955bc 

Install additional tools needed for object detection api 

! apt-get install -y python python-tk
! pip install Cython contextlib2 pillow lxml jupyter matplotlib 

Get protoc 3.0.0, rather than the old version already in the container
! wget https://www.github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
! unzip protoc-3.0.0-linux-x86_64.zip -d proto3
! mkdir -p local/bin && mkdir -p local/include
! mv proto3/bin/* local/bin
! mv proto3/include/* local/include
! rm -rf proto3 protoc-3.0.0-linux-x86_64.zip 

Install pycocoapi (technically its possible to train without it as well, but it's strongly recommended to see metrics associated with training, moreover pretrained model before freezing the layers was made using coco dataset)

! git clone --depth 1 https://github.com/cocodataset/cocoapi.git
! (cd cocoapi/PythonAPI && make -j8)
! cp -r cocoapi/PythonAPI/pycocotools/ models/research/
! rm -rf cocoapi

If you are using conda this step is strongly reccommended for the future commands as some of them might not work (at list it was like that in our case) 

! conda install -c conda-forge ncurses -y

Run protoc on the object detection repo (generate .py files from .proto)

%cd models/research/
! ../../local/bin/protoc object_detection/protos/*.proto --python_out=.

Indicate path after importing os pyhon module, it would be either PYTHONPATH or PATH depending on the case.
import os
os.environ['PATH'] += ":/content/models/research:/content/models/research/slim"

Just to verify everything is correctly set up run the following script and wait if everything runs succesfully: 

! python object_detection/builders/model_builder_test.py

If you want to fully follow tutorial you can use the following command to run bash script to download checkpoint (which is the same one we used to retrain out data meaning coco dataset checkpoint), in case you want to retrain whole model and not to train only part of it indicate true in the end: 

! ./prepare_checkpoint_and_dataset.sh --network_type mobilenet_v1_ssd --train_whole_model false

Training the model: 

%env NUM_TRAINING_STEPS=500
%env NUM_EVAL_STEPS=100

If you're retraining the whole model, Google/Coral.ai suggests these values:
%env NUM_TRAINING_STEPS=50000
%env NUM_EVAL_STEPS=2000

Start transferl learning / retraining values for your specific and in this case tomato disease dataset 

! ./retrain_detection_model.sh --num_training_steps $NUM_TRAINING_STEPS --num_eval_steps $NUM_EVAL_STEPS 

Run bash script to convert the model (after the training is done, retraining with small values does not take much time) 

For the next steps if you are using only notebook you might need to physically download gpg key and locate it while needed.
Following lines of code helps run the code without problems of no mean for inputting the password: 

import getpass
import os

password = getpass.getpass()
command = "sudo -S apt-key add key.gpg" # it actually can be any command it requires -S all tie time as it enables input from stdin 

Do not forget that you can physically locate the key on your own as well

! curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - 

! echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list

! #sudo apt-get update

I manually deleted all the learn_pet training files in order not to confuse anything during training (lear_pet folder is created by prepare_checkpoint_and_dataset automatically, while it also makes ground for manipulating with code as needed (It has Apache 2.0 Lincense and it is still reccomended to take it into account while leaving the state. 


%cd learn_pet/models/

! ls

Compile the model with the latest compiler and try to run model on it 
! edgetpu_compiler output_tflite_graph.tflite
