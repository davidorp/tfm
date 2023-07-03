# TFM

Repository with the code used in my master's thesis. The main aim of this code is to build a model able to predict dementia cases using the Dementiabank dataset and automatic emotion detection using AphasiaBank dataset

The scripts are structured as follows:

dementia_prediction folder contains the scripts for the task of dementia prediction:

- PreProcessAudio and PreProcessText: Python scripts in order to perform the necessary preprocessing for the files.
- TrainAudio: This script is used to train the proposed audio model.
- TrainTexto: This script is used to train the two proposed text model models.
- TrainMultimodal: This script is used to train the two proposed multimodal models.

emotion_detection folder contains the script for the task of automatic emotion detection:
- script_corpus: This script automatically analyzes the whole corpus of AphasiaBank dataset.

For the realization of this project, the docker image in this link has been used https://hub.docker.com/r/huggingface/transformers-pytorch-gpu, where dependencies like torchvision, torchaudio, transformers and librosa are already installed.

The process to follow in order to run this work is running both preprocessing scripts and then training the different models. The script_corpus can be used at any time

