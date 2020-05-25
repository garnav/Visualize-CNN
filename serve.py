import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms            

from VisualizeCNN.gradCam import GradCam

# ===== MODEL ==================================================================
def get_model():
    return models.vgg16(pretrained=True)

def get_conv_layers(model):
    all_modules = dict(model.named_modules())
    conv_layers = []
    for name, module in all_modules.items():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(name)
    return conv_layers

def get_model_classes():
    return [("0", "0"), ("1", "1")]

def inv_model_classes(model_classes):
    inv_classes = {}
    for name, val in model_classes:
        inv_classes[val] = name
    return inv_classes

def get_resize_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224))
    ])

def get_transforms():
    transform = transforms.Compose([
        get_resize_transform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform
        
# ===== TOOLS ==================================================================
def serve_gradcam(model, transform, chosen_layers):

    def run_gradcam(image, idx, device):
        grad_cam = GradCam(model)
        grad_cam.set_target_layers(chosen_layers)
        grad_cam.run_model(transform(image).to(device).unsqueeze(0))
        cam = grad_cam(idx, device)

        grad_cam.reset_gradcam()
        return cam
    
    return run_gradcam

# ===== MISC ==================================================================
def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")