# [ACCV'18] Region-Semantics Preserving Image Synthesis
A <b>TensorFlow</b> implementation of PreservingGAN

[Paper](https://tsujuifu.github.io/pubs/accv18_preserving-gan.pdf) | [Video](https://youtu.be/UwBjSUpjZU8)

![](./imgs/demo.gif)

## Overview
PreservingGAN is an implementation of <br>
"[Region-Semantics Preserving Image Synthesis](https://tsujuifu.github.io/pubs/accv18_preserving-gan.pdf)" <br>
[Kang-Jun Liu](https://dblp.org/pid/191/6724), [Tsu-Jui Fu](https://tsujuifu.github.io/), [Shan-Hung Wu](http://www.cs.nthu.edu.tw/~shwu/) <br>
in Asian Conference on Computer Vision (<b>ACCV</b>) 2018 <br>

<img src='./imgs/model.png' width='75%' /> <br>

Given a <b>reference image and R</b>, the Fast-RSPer synthesis an image by <b>finding (using the gradient descent) an input variable z</b> for the generator such that, at a deep layer where neurons <b>capture the semantics of the reference R</b>, the feature extractor <b>maps the synthesized region to features similar</b> to those of the reference region. Since both the generator and feature extractor are pre-trained, the Fast-RSPer has <b>no dedicated training phase</b> and can generate images efficiently. <br>

<img src='./imgs/result.jpg' width='75%' />

## Requirements
This code is implemented under <b>Python3</b> and [TensorFlow](https://www.tensorflow.org/). <br>
Following libraries are also required: <br>
+ [TensorFlow](https://www.tensorflow.org/) >= 1.6
+ [PyQt5](https://pypi.org/project/PyQt5/)
+ [opencv-python](https://pypi.org/project/opencv-python/)
+ [matplotlib](https://matplotlib.org/)

## Usage
+ First download the [model](https://drive.google.com/drive/folders/1IVx1mwN_e6iZlVNhj5FpKwYnI66PU-g8) and put them under [Model](./Model)
+ GUI
```
python -m main_bedroom
```
+ Ipynb
```
PreservingGAN_Bedroom.ipynb
```
[Here](./Dataset) are some example inputs.

## Resources
+ [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
+ [Bedroom Dataset](http://lsun.cs.princeton.edu/2017/)
+ [This project](https://drive.google.com/drive/folders/1B38jFQ-VV52t8WoZh7QaRTVjZFvbTOYj)

## Citation
```
@inproceedings{liu2018preserving-gan,
  author = {Kang-Jun Liu and Tsu-Jui Fu and Shan-Hung Wu}, 
  title = {Region-Semantics Preserving Image Synthesis}, 
  booktitle = {Asian Conference on Computer Vision (ACCV)}, 
  year = {2018} 
}
```

## Acknowledgement
+ Our <b>CelebA model</b> is based on [EBGAN](https://github.com/carpedm20/BEGAN-tensorflow)
+ Our <b>Bedroom model</b> is based on [WGAN-GP](https://github.com/khanrc/tf.gans-comparison)
+ Our PreservingGAN is also based on [NeuralStyle](https://github.com/anishathalye/neural-style)
