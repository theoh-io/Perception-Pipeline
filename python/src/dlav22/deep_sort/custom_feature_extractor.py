from __future__ import absolute_import
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import sys

from torchreid.utils import (
    check_isfile, load_pretrained_weights, compute_model_complexity
)
from torchreid.models import build_model

#sys.path.append('deep_sort/deep/reid')
from torchreid.utils import FeatureExtractor

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
from torchreid.utils import load_checkpoint 

import os

#Goal is to be adaptive to load any weights, theo model needs special ResNet50, SOTA models use regular Resnet50 but change the dictionary



class CustomFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        model_name='',
        model_path='',
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device='cuda',
        verbose=True
    ):
        # super(CustomFeatureExtractor, self).__init__(model_name,
        # model_path,
        # image_size,
        # pixel_mean,
        # pixel_std,
        # pixel_norm,
        # device,
        # verbose)

        # Build model
        model = build_model(
            model_name,
            num_classes=1,
            pretrained=not (model_path and check_isfile(model_path)),
            use_gpu=device.startswith('cuda')
        )
        model.eval()

        if verbose:
            num_params, flops = compute_model_complexity(
                model, (1, 3, image_size[0], image_size[1])
            )
            print('Model: {}'.format(model_name))
            print('- params: {:,}'.format(num_params))
            print('- flops: {:,}'.format(flops))

        if model_path and check_isfile(model_path):
            print("loading custom weights")
            custom_load_pretrained_weights(model, model_path)

        # Build transform functions
        transforms = []
        transforms += [T.Resize(image_size)]
        transforms += [T.ToTensor()]
        if pixel_norm:
            transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
        preprocess = T.Compose(transforms)

        to_pil = T.ToPILImage()

        device = torch.device(device)
        model.to(device)

        # Class attributes
        self.model = model
        self.preprocess = preprocess
        self.to_pil = to_pil
        self.device = device


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


def custom_load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> from torchreid.utils import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """

    #print("weight_path is", weight_path)
    #print("model is :", model)
    checkpoint = load_checkpoint(weight_path)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    if os.path.basename(weight_path)=="resnet50_theo.pth.tar":
        #print("successfully in theo's custom Reid")
        model=CustomResNet50(751)
        model_dict=model.state_dict()

    else:
        model_dict = model.state_dict()

    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    
    #print(model_dict.keys())
    for k, v in state_dict.items():
        #print(k)
        # if k in model_dict:
        #     #print(model_dict[k].size())
        # #print(v)
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
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
            format(weight_path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )


