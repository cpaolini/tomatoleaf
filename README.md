# tomatoleaf
Tomato leaf disease inference using the Coral Edge TPU Dev Board
### Tomato Leaf Disease Object Detection Model using Coral Edge TPU Dev Board 
## Introduction
This project aims to detect 5 different tomato leaf diseases and 6 different classes including Healthy leaf using a Coral Edge TPU Dev Board. As Google Coral Dev Board is a resource-scarce (in terms of using relatively low power) embedded device it has a relatively small yet powerful processing capability using TPU (Tensor Processing Unit) with 8mb SRAM allowing us to perform real-time inference on tomato leaf images, possibly enabling early detection of diseases and potentially helping farmers and improving crop yield.

## Project Description
The tomato leaf disease object detection model is the pre-trained deep learning model, specifically designed for the detection of common tomato leaf diseases namely Bacterial Spot, Early Blight, Late Blight, Septoria Spot, and Yellow Leaf Curl diseases. The model has been trained by transfer learning techniques using CPU trained coco 300x300 model which was trained on a large dataset of different animal images, this particular model used for disease detection also used hundreds of annotated images (with corresponding bounding boxes) of different diseases and also healthy leaves. The model is trained on TF1 SSD Mobilenet V1 using WSL along with MobaXterm (WSL and MobaXterm are not required as it was just a matter of personal preference).

## Installation and Usage

In this model, we used Coral Dev Board with 4GB ram, as it is an ARM64 device it is no longer supported to compile models on the device and it requires to be compiled either on PC or Cloud. 
The compiler version used for the model was 16.0 and Edge TPU runtime was 14 as it is the latest update as of writing this (June 2023). 

To get started with the Dev Board and understand how to push files to Mendel Linux OS running on the board (including how to set up Mendel itself) please refer to the following link:  [Get Started With Dev Board](https://coral.ai/docs/dev-board/get-started/) "Intro to Coral Dev Board")

Generally, there are several ways of accessing and downloading files: through MDT (Mendel Development Tool), and SSH (Secure Shell Protocol, usually only one pc is allowed however it is possible to run an OpenSSH server and set up ftp communication as well by changing appropriate config values inside /etc/ssh/sshd_config) and via USB or internet. 

To install requirements on the board and be assured that it is capable of running please follow the instructions found [here, try PyCoral](https://coral.ai/docs/dev-board/get-started/#run-pycoral, "PyCoral"), particularly:

mkdir coral && cd coral
git clone https://github.com/google-coral/pycoral.git
cd pycoral
bash examples/install_requirements.sh classify_image.py
python3 examples/classify_image.py \
--model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
--labels test_data/inat_bird_labels.txt \
--input test_data/parrot.jpg 

The following example is the classification model and is different from the detection model as the latter one includes bounding boxes of the object and respectively needs to be trained with bounding box data along with the labels, as well as different parameters and configurations. It is worth mentioning that SSD Mobilenet is an architecture that can be used for both.

To run the tomato leaf disease detection model download the following [tflite model](https://github.com/cpaolini/tomatoleaf/blob/main/output_tflite_graph_edgetpu.tflite "Tomato Disease Detection TFlite Model") and also the [labels](https://github.com/cpaolini/tomatoleaf/blob/main/labels2.txt "disease labels")

To detect an image download or copy the source code of [detect_image.py](https://github.com/cpaolini/tomatoleaf/blob/main/detect_image.py)

Example of usage in the terminal of Mendel: 
python3 detect_image.py --model output_tflite_graph.tflite --labels labels2.txt --input Tomato_Late_blight-43.jpg --output lateblightbounded2.jpg 

If you want to run inference on multiple images located in one folder use the script [detectmany.py](https://github.com/cpaolini/tomatoleaf/blob/main/detectmany.py)

Example of usage of an upper mentioned script where "inpfolder" is the input folder containing pictures to run the inference on and "inpout" is the folder which will contain output images with bounding boxes: 

python3 detectmany.py --model output_tflite_graph_edgetpu.tflite --labels labels2.txt --input inpfolder/ --output inpout/ 

To use the live inference model and stream and the same time you can use the following command: 

edgetpu_detect --model output_tflite_graph_edgetpu.tflite --labels labels2.txt --print


## File Description 

55D1H.csv - all labels and bounding box files for the thousands of diseases (6 different classes of BacterialSpot, EarlyBlight, Healthy, LateBlight, SeptoriaSpot, YellowLeafCUrl) and healthy leaf labels in one CSV file (CSV file was made manually after joining and filtering VGG annotator files)

generate_tfrecord_csv.py - Python Script to generate record files needed for training the model from the CSV file
Syntax: python ../generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record 

method_david_change_csv.py - Python script specifically made for CSV files like 5D1H to make it easier to create changed CSV files involving specific pictures in the folder and also changing the original locations indicated in 5D1H file, which of course will be different for each different instance and each different computer/server/cloud 
Syntax: python method_david_change_csv.py "path of the CSV file like 51DH.csv" "path of the new CSV file which should be generated" "string to replace in CSV file for new locations of files in generated new CSV file" "folder location where those images are to include only those images and exclude all the other which are in CSV file like 51DH.csv" "file type like .jpg" 

data200each.zip - images particularly used for training the object detection model for leaf disease recognition 
data200eachtest.zip - same but images used for testing 
data200train.csv / data200test.csv - Appropriate CSV files 
data200train.record / data200test.record - TensorFlow record files that can be used at any platform and model as long as it is capable train TensorFlow record files

output_tflite_graph.tflite - tflite generated after quantization (it still can be used for the model, but it won't be as accurate and as fast as the Edge TPU one) 

output_tflite_graph_edgetpu.tflite - tflite model which can successfully run on Edge TPU

events.out.tfevents.1686813032.DJ3060 is the evaluation events file which can be opened with TensorBoard (the training events file was too large to upload in case of need contact me to get it) 

tflite_graph.pb (graph file associated with tomato disease trained model, .pbtxt file was large so contact me to get it as well) 

# Training Tutorial Inside The Jupyter Notebook without completely depending on Docker or Colab:

Uninstalling current tf and installing tf1.15 with GPU support (sometimes it is a problem since Google stopped supporting those versions, so if you do not use docker I strongly recommend first downgrading python to 3.7 then installing tf 1.15 either via pip/pip3 or conda environment specifically made for this)  
 #! pip uninstall tensorflow -y
! pip install tensorflow-gpu==1.15
! pip install -U numpy==1.19.5
! pip install -U pycocotools==2.0.1

Check if the TensorFlow model was installed with the appropriate version: 

import tensorflow as tf
print(tf.__version__)

Download the tensorlfow models repository 

! git clone https://github.com/tensorflow/models.git
 
Detach head from the original main of that repository

! cd models && git checkout f788046ca876a8820e05b0b48c1fc2e16b0955bc 

Install additional tools needed for object detection API 

! apt-get install -y python python-tk
! pip install Cython contextlib2 pillow lxml jupyter matplotlib 

Get protoc 3.0.0, rather than the old version already in the container
! wget https://www.github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
! unzip protoc-3.0.0-linux-x86_64.zip -d proto3
! mkdir -p local/bin && mkdir -p local/include
! mv proto3/bin/* local/bin
! mv proto3/include/* local/include
! rm -rf proto3 protoc-3.0.0-linux-x86_64.zip 

Install pycocoapi (technically its possible to train without it as well, but it's strongly recommended to see metrics associated with training, moreover pre-trained model before freezing the layers was made using the coco dataset)

! git clone --depth 1 https://github.com/cocodataset/cocoapi.git
! (cd cocoapi/PythonAPI && make -j8)
! cp -r cocoapi/PythonAPI/pycocotools/ models/research/
! rm -rf cocoapi

If you are using conda this step is strongly recommended for the future commands as some of them might not work (at least it was like that in our case) 

! conda install -c conda-forge ncurses -y

Run protoc on the object detection repo (generate .py files from .proto)

%cd models/research/
! ../../local/bin/protoc object_detection/protos/*.proto --python_out=.

Indicate the path after importing the os python module, it would be either PYTHONPATH or PATH depending on the case.
import os
os.environ['PATH'] += ":/content/models/research:/content/models/research/slim"

Just to verify everything is correctly set up run the following script and wait if everything runs successfully: 

! python object_detection/builders/model_builder_test.py

If you want to fully follow the tutorial you can use the following command to run the bash script to download checkpoint (which is the same one we used to retrain out data meaning the coco dataset checkpoint), in case you want to retrain the whole model and not train only part of it indicate true in the end: 

! ./prepare_checkpoint_and_dataset.sh --network_type mobilenet_v1_ssd --train_whole_model false

Training the model: 

%env NUM_TRAINING_STEPS=500
%env NUM_EVAL_STEPS=100

If you're retraining the whole model, Google/Coral.ai suggests these values:
%env NUM_TRAINING_STEPS=50000
%env NUM_EVAL_STEPS=2000

Start retraining values (transfer learning) for the specific training and testing record files. In this case, those record files are generated from the tomato disease dataset, which itself is the partition of the larger PlantVillage dataset [[1]](#1), while bounding box coordinates used for those images are made by us and they are available both in CSV and XML formats.  

! ./retrain_detection_model.sh --num_training_steps $NUM_TRAINING_STEPS --num_eval_steps $NUM_EVAL_STEPS 

Run bash script to convert the model (after the training is done, retraining with small values does not take much time) 

For the next steps if you are using only a notebook you might need to physically download the gpg key and locate it where needed.
The following lines of code help run the code without problems of no mean for inputting the password: 

import getpass
import os

password = getpass.getpass()
command = "sudo -S apt-key add key.gpg" # it actually can be any command it requires -S all the time as it enables input from stdin 

Do not forget that you can physically locate the key on your own as well

! curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - 

! echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list

! #sudo apt-get update

I manually deleted all the /learn_pet training files in order not to confuse anything during training. /learn_pet directory is created by prepare_checkpoint_and_dataset automatically, while it also makes ground for manipulating the script as needed (It has Apache 2.0 License). 


%cd learn_pet/models/

! ls

Compile the model with the latest compiler and try to run the model on it 
! edgetpu_compiler output_tflite_graph.tflite


##References

<a id="1">[1]</a> D. P. Hughes and M. Salathe, "An open access repository of images on plant health to enable the development of mobile disease diagnostics," arXiv:1511.08060 [cs.CV], Nov. 2015. [Online]. Available: https://doi.org/10.48550/arXiv.1511.08060. 
