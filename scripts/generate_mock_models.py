import torch
import torch.nn as nn
import os

os.makedirs("data/models", exist_ok=True)

class MockSegmentation(nn.Module):
    def forward(self, x):
        # x is (B, 1, H, W). Returns a mask of 1s in the center, 0s on the edge.
        B, C, H, W = x.shape
        y_coords = torch.linspace(-1, 1, H).view(-1, 1).expand(H, W)
        x_coords = torch.linspace(-1, 1, W).view(1, -1).expand(H, W)
        dist = torch.sqrt(x_coords**2 + y_coords**2)
        # 1 inside radius 0.8, 0 outside
        mask = (dist < 0.8).float().unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
        return mask

class MockEnhancement(nn.Module):
    def forward(self, x):
        # Just return the image unchanged (or slightly blurred/sharpened to simulate enhancement)
        # but masked to the segmentation mask roughly
        return x

class MockExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple corner detector simulated as a convolution to get "minutiae" heatmaps
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
        # Set weights to detect strong local contrast (edges/corners)
        with torch.no_grad():
            w = torch.tensor([[-1., -1., -1.],
                              [-1.,  8., -1.],
                              [-1., -1., -1.]])
            self.conv.weight[0, 0] = w  # Terminations
            self.conv.weight[1, 0] = -w # Bifurcations (inverse)

    def forward(self, x):
        B, C, H, W = x.shape
        # Create center mask
        y_coords = torch.linspace(-1, 1, H).view(-1, 1).expand(H, W)
        x_coords = torch.linspace(-1, 1, W).view(1, -1).expand(H, W)
        dist = torch.sqrt(x_coords**2 + y_coords**2)
        mask = (dist < 0.7).float().to(x.device).unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)

        heatmaps = self.conv(x)
        # Normalize and mask edges!
        heatmaps = torch.sigmoid(heatmaps) * mask
        return heatmaps

print("Exporting Segmentation ONNX...")
dummy_input = torch.randn(1, 1, 512, 512)
torch.onnx.export(MockSegmentation(), dummy_input, "data/models/segment.onnx", opset_version=18,
                  input_names=["input"], output_names=["output"])

print("Exporting Enhancement ONNX...")
torch.onnx.export(MockEnhancement(), dummy_input, "data/models/enhance.onnx", opset_version=18,
                  input_names=["input"], output_names=["output"])

print("Exporting Extraction ONNX...")
torch.onnx.export(MockExtraction(), dummy_input, "data/models/extract.onnx", opset_version=18,
                  input_names=["input"], output_names=["output"])

print("Models created successfully in data/models/")
