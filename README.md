#A Convolutional Neural Network Cascade for Face Detection

following this paper:

http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf


This is an implementaton of a fast face detector based on my blog: https://deeplearningmania.quora.com/
An inspiration has been taken from the following paper:
http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf


# Dependencies

This code is written in Torch7. To use it you will need:

A recent version of Torch7: https://github.com/torch/torch7/wiki/Cheatsheet#installing-and-running-torch
To work with a GPU use CudaTensor: https://github.com/torch/cutorch


# Data

FDDB: http://vis-www.cs.umass.edu/fddb/
AFLW: https://lrs.icg.tugraz.at/download.php
PASCAL: http://host.robots.ox.ac.uk/pascal/VOC/databases.html


* This code contains only 12-net and 24-net convolutional networks. 
In order for the detector to give a high recall as stated in the blog - 
add a 48-net network to the pipeline. 
