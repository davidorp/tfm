# TFG

Repository with the code used in my bachelor's degree. The main aim of this code is to build a model able to predict dementia cases using the Dementiabank dataset.

The scripts are structured as follows:

- PreProcessAudio and PreProcessText: Python scripts in order to perform the necessary preprocessing for the files.
- TrainAudio: This script is used to train the proposed audio model.
- TrainTexto: This script is used to train the two proposed text model model.
- TrainMultimodal: This script is used to train the two the proposed multimodal models.

For the realization of this project, the docker image in this link has been used https://hub.docker.com/r/huggingface/transformers-pytorch-gpu, where dependencies like torchvision, torchaudio, transformers and librosa are already installed.

The process to follow in order to run this work is running both preprocessing scripts and then train the different models.
