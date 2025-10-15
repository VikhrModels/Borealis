#!/bin/bash

sudo apt update
sudo apt upgrade -y
sudo apt install -y wget pv pigz

wget -O archive.tar.gz https://huggingface.co/datasets/Vikhrmodels/Audio_Augs/resolve/main/archive.tar.gz

pv archive.tar.gz | pigz -dc | tar -x

rm archive.tar.gz