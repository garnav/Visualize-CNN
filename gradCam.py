# gradCam.py
# Arnav Ghosh
# 22nd March 2020

import copy
import cv2
import numpy as np

import torch
import torch.nn as nn

class GradCam(object):

    def __init__(self, model):
        self.input_shape = None
        self.output = None # shape : (1, C)
        self.model = model
        self.model.eval()

        self.target_layer = None
        self.grad_hook = None
        self.layer_hook = None
        self.target_grad = None
        self.layer_output = None

    def _set_grad(self, grad):
        if self.target_grad is not None:
            raise Exception("Layer Gradient already exists.")
        self.target_grad = grad

    def _set_layer_output(self, output):
        if self.layer_output is not None:
            raise Exception("Layer Output already exists.")
        self.layer_output = output

    def _remove_grads(self):
        self.model.zero_grad()
        self.target_grad = None

    def _remove_layer_output(self):
        self.layer_output = None

    def remove_diagnostic_tools(self):
        if self.layer_hook is not None:
            self.layer_hook.remove()

        if self.grad_hook is not None:
            self.grad_hook.remove()
        
        self._remove_layer_output()
        self._remove_grads()

    def set_model(self, model):
        self.model = model
        self.remove_diagnostic_tools()

    def set_target_layer(self, target_layer):
        self.remove_diagnostic_tools()

        self.target_layer = target_layer
        target_module = dict(self.model.named_modules())[self.target_layer]
        self.layer_hook = target_module.register_forward_hook(lambda module, input, output : self._set_layer_output(output[0]))
        self.grad_hook = target_module.register_backward_hook(lambda module, grad_input, grad_output : self._set_grad(grad_output[0]))

    def run_model(self, input):
        self._remove_layer_output()
        self.input_shape = tuple(input.size())
        self.output = self.model(input)
        
    def __call__(self, target_idx, device):
        if self.output is None:
            raise Exception("Output doesn't exist. Use run_model first.")

        in_gradient = torch.zeros(self.output.size()).double().to(device)
        in_gradient[0, target_idx] = 1
        self._remove_grads()
        self.output.backward(gradient=in_gradient, retain_graph=True)
        weights = torch.mean(self.target_grad, dim=(2,3)).unsqueeze(2).unsqueeze(3)
        weighted_grad = weights * self.layer_output
        cam = torch.sum(weighted_grad, dim=1).squeeze(0)
        cam = nn.functional.relu(cam).cpu().data.numpy()
        cam = cv2.resize(cam, self.input_shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam