# Transformer-based Image Captioning with Vision Transformer Encoder

[![pytorch version](https://img.shields.io/badge/pytorch-2.1.2-yellow.svg)](https://pypi.org/project/torch/2.1.2-/)
[![torchvision version](https://img.shields.io/badge/torchvision-0.16.2-yellow.svg)](https://pypi.org/project/torchvision/0.16.2-/)
[![numpy version](https://img.shields.io/badge/numpy-1.26.4-blue.svg)](https://pypi.org/project/numpy/1.26.4/)
[![PIL version](https://img.shields.io/badge/PIL-10.2.0-green.svg)](https://pypi.org/project/Pillow/10.2.0/)
## Abstract

In this project, we apply the Transformer architecture to the image captioning task. We combine a Vision Transformer (ViT) encoder with a standard Transformer decoder to generate captions for input images. Our model is trained from scratch on the Flickr8k dataset, demonstrating that the ViT encoder can effectively capture visual features without the need for large-scale pre-training.

## Introduction

The Transformer architecture has shown remarkable performance in various natural language processing tasks. In this project, we explore its application to the image captioning task by combining a Vision Transformer (ViT) encoder (with a Shifted Patch Tokenization (SPT) and a Locality Self-Attention (LSA)), with a standard Transformer decoder.

Unlike the original ViT, which relies on pre-training using large-size datasets, we train our model from scratch on the Flickr8k dataset.

## Method

### Encoder: Vision Transformer (ViT)

The ViT encoder processes input images by splitting them into patches and linearly embedding these patches. The resulting patch embeddings are fed into a Transformer encoder, which uses multi-head self-attention and feed-forward neural networks to capture visual features.

### Decoder: Standard Transformer

The Transformer decoder generates output captions autoregressively, token by token. It takes the encoded visual features and previously generated tokens as input and uses multi-head self-attention, encoder-decoder attention, and feed-forward neural networks to generate the next token.

## Dataset

We train and evaluate our model on the [Flickr8k dataset](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Krasin_et_al_CVPR13.pdf), which consists of 8,000 images, each paired with 5 human-annotated captions. 

## How to Use
To install the depedencies
```
pip install -r  requirements.txt
python -m spacy download en_core_web_sm
```
To train the model, run the following command:

```python
python training.py --epochs 100 --height 224 --width 224 --patch_size 16
```


## Citation

```
@article{lee2021vision,
  title={Vision Transformer for Small-Size Datasets},
  author={Lee, Seung Hoon and Lee, Seunghyun and Song, Byung Cheol},
  journal={arXiv preprint arXiv:2112.13492},
  year={2021}
}
```
