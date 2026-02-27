import argparse
import json
import os
import sys
import time
import torch
import torchvision.transforms.functional as F
from PIL import Image
import subprocess

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

def prepare_image_tensor(image_path):
    """Load image and convert to tensor"""
    image = Image.open(image_path).convert('RGB')
    # Convert to tensor and normalize to [0, 1]
    tensor = F.to_tensor(image)
    return tensor, image

def resize_image_for_comparison(original_tensor, target_size):
    """Resize original image to match super-resolved output for PSNR calculation"""
    # target_size should be (H, W)
    resized = F.resize(original_tensor.unsqueeze(0), target_size, antialias=True)
    return resized.squeeze(0)

def save_results_to_json(input_path, output_path, psnr, processing_time, scale_factor, architecture, unique_id):
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
        'architecture': architecture,
        'algorithm': 'edsr'
    }
    
    # Save results
    result_file = os.path.join(temp_dir, f'edsr_results_{unique_id}.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"DEBUG EDSR: Results saved to {result_file}")
    return result_file

def main():
    parser = argparse.ArgumentParser(description='EDSR Super-Resolution with PSNR calculation')
    parser.add_argument('-i', '--input', required=True, help='Input image path')
    parser.add_argument('-o', '--output-dir', required=True, help='Output directory')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4], help='Scaling factor')
    parser.add_argument('--arch', default='edsr_baseline', choices=['edsr', 'edsr_baseline'], help='EDSR architecture')
    parser.add_argument('--unique-id', required=True, help='Unique identifier for this run')
    
    args = parser.parse_args()
    
    print(f"=== EDSR Inference Starting ===")
    print(f"Input: {args.input}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Scale: {args.scale}x")
    print(f"Architecture: {args.arch}")
    print(f"Unique ID: {args.unique_id}")
    
    # Load original image for PSNR calculation
    try:
        original_tensor, original_pil = prepare_image_tensor(args.input)
        print(f"DEBUG EDSR: Original image size: {original_pil.size}")
    except Exception as e:
        print(f"ERROR EDSR: Failed to load input image: {e}")
        return
    
    # Create output filename
    input_filename = os.path.basename(args.input)
    name, ext = os.path.splitext(input_filename)
    output_filename = f"{name}_edsr_{args.arch}_x{args.scale}{ext}"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Run TorchSR
    start_time = time.time()
    try:
        print(f"DEBUG EDSR: Running TorchSR command...")
        
        # Build TorchSR command
        python_executable = sys.executable
        command = [
            python_executable, '-m', 'torchsr.train',
            '--arch', args.arch,
            '--scale', str(args.scale),
            '--download-pretrained',
            '--images', args.input,
            '--destination', args.output_dir,
            '--cpu'
        ]
        
        print(f"DEBUG EDSR: Command: {' '.join(command)}")
        
        # Execute TorchSR
        result = subprocess.run(command, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        if result.returncode != 0:
            print(f"ERROR EDSR: TorchSR failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
            return
        
        processing_time = time.time() - start_time
        print(f"DEBUG EDSR: TorchSR completed in {processing_time:.2f} seconds")
        
    except Exception as e:
        print(f"ERROR EDSR: Failed to run TorchSR: {e}")
        return
    
    # Find the generated output file (TorchSR might use different naming)
    generated_files = []
    for file in os.listdir(args.output_dir):
        if file.startswith(name) and file.endswith(ext) and 'x' in file:
            generated_files.append(os.path.join(args.output_dir, file))
    
    if not generated_files:
        print(f"ERROR EDSR: No output file found in {args.output_dir}")
        return
    
    # Use the first generated file (or rename it to our expected name)
    generated_path = generated_files[0]
    if generated_path != output_path:
        try:
            os.rename(generated_path, output_path)
            print(f"DEBUG EDSR: Renamed {generated_path} to {output_path}")
        except Exception as e:
            print(f"WARNING EDSR: Could not rename file: {e}")
            output_path = generated_path
    
    # Calculate PSNR
    psnr = 0.0
    try:
        # Load super-resolved image
        sr_tensor, sr_pil = prepare_image_tensor(output_path)
        print(f"DEBUG EDSR: Super-resolved image size: {sr_pil.size}")
        
        # Resize original to match output for PSNR calculation
        target_size = (sr_tensor.shape[1], sr_tensor.shape[2])  # (H, W)
        hr_tensor = resize_image_for_comparison(original_tensor, target_size)
        
        # Calculate PSNR
        psnr = calculate_psnr(sr_tensor, hr_tensor)
        print(f"DEBUG EDSR: PSNR: {psnr:.2f} dB")
        
    except Exception as e:
        print(f"WARNING EDSR: Could not calculate PSNR: {e}")
        psnr = 0.0
    
    # Save results to JSON
    save_results_to_json(
        args.input, output_path, psnr, processing_time * 1000,  # Convert to ms
        args.scale, args.arch, args.unique_id
    )
    
    print("=== EDSR Inference Results ===")
    print(f"Input Image: {args.input}")
    print(f"Output Image: {output_path}")
    print(f"Scale Factor: {args.scale}x")
    print(f"Architecture: {args.arch}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Processing Time: {processing_time*1000:.1f} ms")

if __name__ == '__main__':
    main()
