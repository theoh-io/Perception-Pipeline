#!/usr/bin/env python

import numpy as np
from torchreid.utils import load_checkpoint 

#imports for CustomResNet50
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms
from torch.nn import functional as F

#imports for custom load pretrained
from collections import OrderedDict
import warnings

def info_pretrained(weights):
    global verbose
    # Print model's state_dict
    checkpoint=load_checkpoint(weights)
    print(f"the type of checkpoint is {type(checkpoint)}")
    print(checkpoint.keys())
    weights_dict=checkpoint['state_dict']
    if verbose is True: print("weights's state_dict:")
    for param_tensor in weights_dict:
        if verbose is True:
            print(param_tensor, "\t", weights_dict[param_tensor].size())
    return weights_dict

def info_model(model_dict):
    if verbose is True: print("Model's state_dict:")
    for param_tensor in model_dict:
        if verbose is True:
            print(param_tensor, "\t", model_dict[param_tensor].size())

def loading_pretrained(state_dict, model_dict):
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    
    #print(model_dict.keys())
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'
        )
    else:
        print(
            'Successfully loaded pretrained weights'
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )



class CustomResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(CustomResNet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048
        self.cam = False

    def __call__(self, x):
        return self.forward()

    def forward(self, x):
        x = self.base(x)
        if self.cam:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


if __name__ == "__main__":
    verbose=False #if activate verbose will print the 2 state dict must redirect to a txt file
    path_weights="../src/dlav22/deep_sort/deep/checkpoint"
    weights_name="/resnet50_theo.pth.tar"
    weights=path_weights+weights_name
    weights_state_dict=info_pretrained(weights)
    model=CustomResNet50(751)
    model_dict=model.state_dict()
    info_model(model_dict)
    #print(model_dict)
    loading_pretrained(weights_state_dict, model_dict)