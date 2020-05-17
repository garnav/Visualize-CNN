# gradCam.py
# Arnav Ghosh
# 11th April 2020

import copy
import cv2
import numpy as np

import torch
import torch.nn as nn

class GradCam(object):

    def __init__(self, model):
        self.model = model.to(dtype=torch.get_default_dtype())
        self.model.eval()

        self.input_shape = None
        self.outputs = None # shape : (num_inputs, C)

        self.target_layers = [] # [layer1, layer2, ...]
        self.layer_hooks = {} # {<layer1> : hook}
        self.grad_hooks = {} # {<layer1> : hook}
        self.layer_outputs = {} # {<layer1> : array of shape (num_inputs, *output_shape)}
        self.layer_grads = {} # {<layer1> : array of shape (num_inputs, *grad_shape)}

    def _remove_layer_hooks(self):
        for v in self.layer_hooks.values():
            v.remove()
        self.layer_hooks = {}

    def _remove_grad_hooks(self):
        for v in self.grad_hooks.values():
            v.remove()
        self.grad_hooks = {}

    def _remove_grads(self):
        self.model.zero_grad()

        for k in self.layer_grads.keys():
            self.layer_grads[k] = None

    def _remove_layer_outputs(self):
        for k in self.layer_outputs.keys():
            self.layer_outputs[k] = None

    # resets everything but the model
    def reset_gradcam(self):
        self.input_shape = None
        self.outputs = None
        self.target_layers = None

        self._remove_layer_hooks()
        self._remove_grad_hooks()
        
        self._remove_layer_outputs()
        self._remove_grads()

    def set_model(self, model):
        self.reset_gradcam()
        self.model = model
        self.model.eval()

    def _set_grad(self, layer, grad):
        if self.layer_grads[layer] is not None:
            raise Exception(f"{layer} gradient already exists.")
        self.layer_grads[layer] = grad

    def _set_layer_output(self, layer, output):
        if self.layer_outputs[layer] is not None:
            raise Exception(f"{layer} output already exists.")
        self.layer_outputs[layer] = output

    def set_target_layers(self, target_layers):
        self.reset_gradcam()

        self.target_layers = target_layers
        self.layer_hooks = {}
        self.grad_hooks = {}

        all_modules = dict(self.model.named_modules())
        for layer in self.target_layers:
            target_module = all_modules[layer]

            # check requires_grad is True
            for name, param in target_module.named_parameters():
                if not param.requires_grad:
                    raise Exception(f"{name} has requires_grad == False. \
                                        For all parameters in target layers, requires_grad should be True.")                
            
            # Python scoping prevents 'layer' from being used as is. Requires Intermediate Func.
            self.layer_hooks[layer] = target_module.register_forward_hook((lambda ly : lambda module, input, output : self._set_layer_output(ly, output))(layer))
            self.grad_hooks[layer] = target_module.register_backward_hook((lambda ly : lambda module, grad_input, grad_output : self._set_grad(ly, grad_output[0]))(layer))

            self.layer_outputs[layer] = None
            self.layer_grads[layer] = None

    # Assumes that the input and the model are on the same device
    def run_model(self, input):
        self._remove_layer_outputs()
        self.input_shape = tuple(input.size())
        self.output = self.model(input.to(dtype=torch.get_default_dtype()))

    # TODO : Vectorize Heatmap Calculation for a single layer
    def grad_cam(self, layer, image_idx):
        weights = torch.mean(self.layer_grads[layer][image_idx:(image_idx + 1)], dim=(2,3)).unsqueeze(2).unsqueeze(3)
        weighted_grad = weights * self.layer_outputs[layer][image_idx:(image_idx + 1)]
        
        cam = torch.sum(weighted_grad, dim=1).squeeze(0)
        cam = nn.functional.relu(cam).cpu().data.numpy()
        cam = cv2.resize(cam, self.input_shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam
        
    # cam shape == inputs shape (except for channels)
    def __call__(self, target_idx, device):
        if self.output is None:
            raise Exception("Output doesn't exist. Use run_model first.")

        in_gradient = torch.zeros(self.output.size()).double().to(device=device, dtype=torch.get_default_dtype())
        in_gradient[:, target_idx] = 1
        self._remove_grads()
        self.output.backward(gradient=in_gradient, retain_graph=True)

        all_cams = {}
        for layer in self.target_layers:
            if layer not in all_cams:
                all_cams[layer] = np.zeros((self.input_shape[0], self.input_shape[2], self.input_shape[3]))
            
            for i in range(self.input_shape[0]):
                all_cams[layer][i, :, :] = self.grad_cam(layer, i)

        return all_cams