# Deep Learning for Autonomous Vehicles


## Getting started

#### Venv Setup
First set up a python >= 3.7 virtual environment:
```
$ cd <desired-dir>
$ python3 -m venv <desired venv name>
$ activate desired venv name
```
Make sure to install python 3 in case you do not have it.

#### Module Installation
Execute in this directory:
```
$ pip install -r requirements.txt 
#download libs and install it
$ git submodule update --init #to download submodule for deep-person-reid
$ cd libs
$ cd deep-person-reid
$ pip install -e .
$ cd ../ #go back at libs/ level
$ git clone git@github.com:open-mmlab/mmtracking.git
$ cd mmtracking 
# be sure follow the install instructions from official documentation: https://mmtracking.readthedocs.io/en/latest/install.html
$ pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
$ pip install mmdet
$ pip install -r requirements/build.txt
$ pip install -v -e .
#installing our custom package perceptionloomo
$ cd .../../python
$ pip install -e .


```

#### Downloading pretrained models
```
#stark: https://github.com/open-mmlab/mmtracking/tree/master/configs/sot/stark
$ cd src/perceptionloomo
$ mkdir checkpoints
$ wget https://download.openmmlab.com/mmtracking/sot/stark/stark_st2_r50_50e_lasot/stark_st2_r50_50e_lasot_20220416_170201-b1484149.pth

#siamese: https://github.com/open-mmlab/mmtracking/tree/master/configs/sot/siamese_rpn
wget https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_20e_lasot_20220420_181845-dd0f151e.pth

```




download theo's weights into deepsort/deep/checkpoint
run SOTA weights script
