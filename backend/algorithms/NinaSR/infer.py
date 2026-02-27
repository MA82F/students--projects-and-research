#!/usr/bin/env python3
"""
NinaSR Inference Script (Simplified version without TorchSR dependency)
Uses pre-downloaded model weights and basic PyTorch operations
"""

import argparse
import os
import sys
import json
import time
import uuid
import math
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from urllib.request import urlretrieve

# NinaSR Model Definition (copied from torchsr to avoid dependencies)
class AttentionBlock(nn.Module):
    """A typical Squeeze-Excite attention block, with a local pooling instead of global"""
    
    def __init__(self, n_feats, reduction=4, stride=16):
        super(AttentionBlock, self).__init__()
        self.body = nn.Sequential(
            nn.AvgPool2d(2 * stride - 1, stride=stride, padding=stride - 1, count_include_pad=False),
            nn.Conv2d(n_feats, n_feats // reduction, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(n_feats // reduction, n_feats, 1, bias=True),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=stride, mode="nearest"),
        )

    def forward(self, x):
        res = self.body(x)
        if res.shape != x.shape:
            res = res[:, :, : x.shape[2], : x.shape[3]]
        return res * x

class ResBlock(nn.Module):
    def __init__(self, n_feats, mid_feats, in_scale, out_scale):
        super(ResBlock, self).__init__()
        self.in_scale = in_scale
        self.out_scale = out_scale

        m = []
        conv1 = nn.Conv2d(n_feats, mid_feats, 3, padding=1, bias=True)
        nn.init.kaiming_normal_(conv1.weight)
        nn.init.zeros_(conv1.bias)
        m.append(conv1)
        m.append(nn.ReLU(True))
        m.append(AttentionBlock(mid_feats))
        conv2 = nn.Conv2d(mid_feats, n_feats, 3, padding=1, bias=False)
        nn.init.kaiming_normal_(conv2.weight)
        m.append(conv2)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x * self.in_scale) * (2 * self.out_scale)
        res += x
        return res

class Rescale(nn.Module):
    def __init__(self, sign):
        super(Rescale, self).__init__()
        rgb_mean = (0.4488, 0.4371, 0.4040)
        bias = sign * torch.Tensor(rgb_mean).reshape(1, 3, 1, 1)
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return x + self.bias

class NinaSR(nn.Module):
    def __init__(self, n_resblocks, n_feats, scale, expansion=2.0):
        super(NinaSR, self).__init__()
        self.scale = scale

        n_colors = 3
        self.head = self.make_head(n_colors, n_feats)
        self.body = self.make_body(n_resblocks, n_feats, expansion)
        self.tail = self.make_tail(n_colors, n_feats, scale)

    @staticmethod
    def make_head(n_colors, n_feats):
        m_head = [
            Rescale(-1),
            nn.Conv2d(n_colors, n_feats, 3, padding=1, bias=False),
        ]
        return nn.Sequential(*m_head)

    @staticmethod
    def make_body(n_resblocks, n_feats, expansion):
        mid_feats = int(n_feats * expansion)
        out_scale = 4 / n_resblocks
        expected_variance = 1.0
        m_body = []
        for i in range(n_resblocks):
            in_scale = 1.0 / math.sqrt(expected_variance)
            m_body.append(ResBlock(n_feats, mid_feats, in_scale, out_scale))
            expected_variance += out_scale**2
        return nn.Sequential(*m_body)

    @staticmethod
    def make_tail(n_colors, n_feats, scale):
        m_tail = [
            nn.Conv2d(n_feats, n_colors * scale**2, 3, padding=1, bias=True),
            nn.PixelShuffle(scale),
            Rescale(1),
        ]
        return nn.Sequential(*m_tail)

    def forward(self, x, scale=None):
        if scale is not None and scale != self.scale:
            raise ValueError(f"Network scale is {self.scale}, not {scale}")
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

# Model creation functions
def ninasr_b0(scale):
    model = NinaSR(10, 16, scale)
    return model

def ninasr_b1(scale):
    model = NinaSR(26, 32, scale)
    return model

def ninasr_b2(scale):
    model = NinaSR(84, 56, scale)
    return model

# URLs for pretrained models
model_urls = {
    "b0": {
        2: "https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b0_x2.pt",
        3: "https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b0_x3.pt",
        4: "https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b0_x4.pt",
        8: "https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b0_x8.pt",
    },
    "b1": {
        2: "https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b1_x2.pt",
        3: "https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b1_x3.pt",
        4: "https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b1_x4.pt",
        8: "https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b1_x8.pt",
    },
    "b2": {
        2: "https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b2_x2.pt",
        3: "https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b2_x3.pt",
        4: "https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b2_x4.pt",
        8: "https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b2_x8.pt",
    }
}

def download_model_weights(variant, scale, device):
    """Download and cache model weights"""
    if variant not in model_urls or scale not in model_urls[variant]:
        raise ValueError(f"No pretrained model available for {variant} x{scale}")
    
    # Create models directory
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Model file path
    model_filename = f"ninasr_{variant}_x{scale}.pt"
    model_path = os.path.join(models_dir, model_filename)
    
    # Download if not exists
    if not os.path.exists(model_path):
        print(f"DEBUG NINASR: Downloading {variant} x{scale} model...")
        url = model_urls[variant][scale]
        try:
            urlretrieve(url, model_path)
            print(f"DEBUG NINASR: Model downloaded to {model_path}")
        except Exception as e:
            print(f"ERROR NINASR: Failed to download model: {e}")
            raise
    
    # Load model
    try:
        state_dict = torch.load(model_path, map_location=device)
        print(f"DEBUG NINASR: Model weights loaded from {model_path}")
        return state_dict
    except Exception as e:
        print(f"ERROR NINASR: Failed to load model weights: {e}")
        raise

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

def prepare_image(image_path, device):
    """Load and prepare image for inference"""
    image = Image.open(image_path).convert('RGB')
    
    # Convert to tensor and normalize to [0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    return image_tensor, image

def save_output_image(tensor, output_path):
    """Save tensor as image"""
    # Convert tensor back to numpy array
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image and save
    output_image = Image.fromarray(tensor)
    output_image.save(output_path)
    return output_image

def create_lr_image(hr_image, scale_factor, uploads_dir, base_name, ext):
    """Create and save low resolution image for comparison"""
    lr_width = hr_image.width // scale_factor
    lr_height = hr_image.height // scale_factor
    
    lr_image = hr_image.resize((lr_width, lr_height), Image.BICUBIC)
    
    # Save LR image
    lr_filename = f"{base_name}_lr_x{scale_factor}{ext}"
    lr_path = os.path.join(uploads_dir, lr_filename)
    lr_image.save(lr_path)
    
    print(f"DEBUG NINASR: Low resolution image saved to {lr_path}")
    return lr_filename

def save_results_to_json(input_path, output_path, psnr, processing_time, scale_factor, variant, unique_id, lr_filename=None):
    """Save inference results to JSON file"""
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    temp_dir = os.path.join(backend_dir, 'temp_results')
    os.makedirs(temp_dir, exist_ok=True)
    
    results = {
        'input_path': input_path,
        'output_path': output_path,
        'psnr': psnr,
        'processing_time': processing_time,
        'scale_factor': scale_factor,
        'variant': variant,
        'algorithm': 'ninasr'
    }
    
    if lr_filename:
        results['lr_filename'] = lr_filename
    
    # Save results
    result_file = os.path.join(temp_dir, f'ninasr_results_{unique_id}.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"DEBUG NINASR: Results saved to {result_file}")

def main():
    parser = argparse.ArgumentParser(description="NinaSR Super-Resolution Inference")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory")
    parser.add_argument("-s", "--scale", type=int, choices=[2, 3, 4, 8], default=3, help="Scaling factor")
    parser.add_argument("-v", "--variant", type=str, choices=['b0', 'b1', 'b2'], default='b1', help="Model variant")
    parser.add_argument("--unique-id", type=str, required=True, help="Unique identifier for this request")
    
    args = parser.parse_args()
    
    print(f"DEBUG NINASR: Starting inference with scale={args.scale}, variant={args.variant}")
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEBUG NINASR: Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    uploads_dir = os.path.join(os.path.dirname(args.output), 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Load model based on variant
    model_functions = {
        'b0': ninasr_b0,
        'b1': ninasr_b1,
        'b2': ninasr_b2
    }
    
    try:
        print(f"DEBUG NINASR: Loading {args.variant.upper()} model with {args.scale}x scaling...")
        
        # Create model
        model = model_functions[args.variant](scale=args.scale)
        
        # Download and load pretrained weights
        state_dict = download_model_weights(args.variant, args.scale, device)
        model.load_state_dict(state_dict)
        
        model = model.to(device)
        model.eval()
        print(f"DEBUG NINASR: Model loaded successfully")
    except Exception as e:
        print(f"ERROR NINASR: Failed to load model: {e}")
        return
    
    # Prepare input image
    try:
        input_tensor, original_image = prepare_image(args.input, device)
        print(f"DEBUG NINASR: Input image loaded: {original_image.size}")
    except Exception as e:
        print(f"ERROR NINASR: Failed to load input image: {e}")
        return
    
    # Create LR image for comparison
    input_filename = os.path.basename(args.input)
    base_name, ext = os.path.splitext(input_filename)
    ext = ext.lower() if ext else '.png'
    
    lr_filename = create_lr_image(original_image, args.scale, uploads_dir, base_name, ext)
    
    # Run inference
    try:
        with torch.no_grad():
            start_time = time.time()
            sr_tensor = model(input_tensor)
            processing_time = time.time() - start_time
            
        print(f"DEBUG NINASR: Inference completed in {processing_time:.3f} seconds")
    except Exception as e:
        print(f"ERROR NINASR: Inference failed: {e}")
        return
    
    # Save output
    try:
        random_id = str(uuid.uuid4())[:8]
        output_filename = f"{base_name}_ninasr_{args.variant}_x{args.scale}_{random_id}{ext}"
        output_path = os.path.join(args.output, output_filename)
        
        output_image = save_output_image(sr_tensor, output_path)
        print(f"DEBUG NINASR: Output saved to {output_path}")
    except Exception as e:
        print(f"ERROR NINASR: Failed to save output: {e}")
        return
    
    # Calculate PSNR (if possible)
    try:
        # Resize original to match output for PSNR calculation
        target_size = (original_image.width * args.scale, original_image.height * args.scale)
        hr_resized = original_image.resize(target_size, Image.BICUBIC)
        
        # Convert to tensor for PSNR calculation
        hr_array = np.array(hr_resized).astype(np.float32) / 255.0
        hr_tensor = torch.from_numpy(hr_array).permute(2, 0, 1).unsqueeze(0).to(device)
        
        psnr = calculate_psnr(sr_tensor, hr_tensor)
        print(f"DEBUG NINASR: PSNR: {psnr:.2f} dB")
    except Exception as e:
        print(f"WARNING NINASR: Could not calculate PSNR: {e}")
        psnr = 0.0
    
    # Save results to JSON
    save_results_to_json(
        args.input, output_path, psnr, processing_time,  # Already in seconds
        args.scale, args.variant, args.unique_id, lr_filename
    )
    
    print("=== NinaSR Inference Results ===")
    print(f"Input Image: {args.input}")
    print(f"Output Image: {output_path}")
    print(f"Scale Factor: {args.scale}x")
    print(f"Model Variant: {args.variant.upper()}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Processing Time: {processing_time*1000:.1f} ms")

if __name__ == "__main__":
    main()
