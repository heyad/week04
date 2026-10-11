# --------------------------------------------------
# Grad-CAM explanation
# --------------------------------------------------

import numpy as np
import torch
from torch import nn


class GradCAM:
    """A simple Grad-CAM implementation for ResNet18."""

    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        layer = self.model
        for name in self.target_layer.split("."):
            layer = getattr(layer, name)

        self._feature_hook = layer.register_forward_hook(self._feature_hook_fn)
        self._gradient_hook = layer.register_full_backward_hook(self._gradient_hook_fn)

    def _feature_hook_fn(self, module, inputs, output):
        self.feature_maps = output.detach()

    def _gradient_hook_fn(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._feature_hook.remove()
        self._gradient_hook.remove()

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = None):
        """Generate a Grad-CAM heatmap for the chosen class."""
        self.model.eval()

        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        loss = logits[:, target_class].sum()
        loss.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.feature_maps).sum(dim=1, keepdim=True)
        heatmap = torch.relu(heatmap)
        heatmap = heatmap.squeeze(0).squeeze(0)

        heatmap = heatmap.cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap


def create_gradcam(model: nn.Module):
    """Return a Grad-CAM object using the final convolutional layer of ResNet18."""
    return GradCAM(model, target_layer="layer4")
