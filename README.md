# DBS: Dynamic human body shape model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12wh8pFp8AVmR4Jzp1VgcrRY6f4KXch3u?usp=sharing)

## Overview
This project provides a dynamic human body shape model (DBS) based on the state of art sequence modelling structures (LSTM, GRU, TCN). Compared with SMPL and DMPL models, it can better predict the dynamic deformation of human tissues in the case of movements. Different from dynamic models such as DMPL and Dyna, the DBS model uses more historical information and uses a deep sequence model to train a nonlinear function of pose rotation matrices to represent the pose blend shape. In this paper, the DBS model is trained with the Dyna’s 4D scans and AMASS’s parameters, and shows a good training effect. The DBS model can run in a GPU environment and has high computational efficiency. I also wrote animation rendering scripts to better visualize the dynamic deformation of the human body. However, the DBS model also has the problem of uneven human skin, which will be studied in future work.

## Sample animation
<p float="center">
  <img src="https://i.makeagif.com/media/8-31-2020/uczsHQ.gif" width="50%" />
</p>

## Features
This implementation:

- has the training code for DBS implemented purely in PyTorch.
- can predict body deformations on arbitrary pose sequence from AMASS dataset with multiple subjects.
- supports both CPU and GPU inference (GPU is recommended).
- dynamic human body shape vertices predicting speed is up-to 5000 FPS on Tesla K80.
- includes LSTM, GRU, TCN dynamic blend shape layers, where TCN model's receptive field is 15 time steps, much larger than DMPL and Dyna models.
- provides script for animations and generating videos.

## Google Colab
I provide a script for model training on colab. It provides Tesla T4 GPU for free, which can train the model very quickly.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12wh8pFp8AVmR4Jzp1VgcrRY6f4KXch3u?usp=sharing)

## References
I borrowed some functions and the construction of the SMPL baseline model in pytorch. I used the AMASS dataset and the pre-trained models of SMPL and DMPL. These resources are listed below.

- SMPL model in pytorch is from [SMPL](https://github.com/CalciferZh/SMPL).
- TCN blocks are borrowed from [TCN](https://github.com/locuslab/TCN).
- Training 4D body scans and some function for animations are borrowed from [Dyna](http://dyna.is.tue.mpg.de/).
- Training parameter data are used from [AMASS](https://amass.is.tue.mpg.de/)
