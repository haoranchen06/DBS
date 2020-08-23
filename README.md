# DBS
Dynamic human body shape model

## Overview
This project provides a dynamic human body shape model (DBS) based on the state of art sequence modelling structures (LSTM, GRU, TCN). Compared with SMPL and DMPL models, it can better predict the dynamic deformation of human tissues in the case of movements. Different from dynamic models such as DMPL and Dyna, the DBS model uses more historical information and uses a deep sequence model to train a nonlinear function of pose rotation matrices to represent the pose blend shape. In this paper, the DBS model is trained with the Dyna’s 4D scans and AMASS’s parameters, and shows a good training effect. The DBS model can run in a GPU environment and has high computational efficiency. I also wrote animation rendering scripts to better visualize the dynamic deformation of the human body. However, the DBS model also has the problem of uneven human skin, which will be studied in future work.
