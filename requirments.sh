#!/bin/bash

# base system ubuntu 22.04
# CPU only!

sudo apt install python3-venv
python3 -m venv virenv
source ./virenv/bin/activate

pip3 install jupyter
pip3 install tensorflow-cpu
pip3 install tensorflow-probability
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install pyro-ppl
pip3 install graphviz
sudo apt install graphviz