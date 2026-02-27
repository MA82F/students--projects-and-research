#!/usr/bin/env python3
"""
RCAN (Residual Channel Attention Network) Inference Script
Using TorchSR command line interface
"""

import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import numpy as np

def calculate_psnr(img1_path, img2_path):
    """Calculate PSNR between two images"""
    try:
        # Load images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # Resize img2 to match img1 if needed
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.LANCZOS)
        
        # Convert to numpy arrays
        img1_np = np.array(img1).astype(np.float64)
        img2_np = np.array(img2).astype(np.float64)
        
        # Calculate MSE
        mse = np.mean((img1_np - img2_np) ** 2)
        
        if mse == 0:
            return float('inf')
        
        # Calculate PSNR
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return psnr
    except Exception as e:
        print(f"Error calculating PSNR: {e}")
        return None

def create_bicubic_reference(input_path, output_path, scale):
    """Create bicubic upscaled reference image"""
    try:
        with Image.open(input_path) as img:
            # Get current size
            width, height = img.size
            
            # Calculate new size
            new_width = width * scale
            new_height = height * scale
            
            # Resize using bicubic interpolation
            bicubic_img = img.resize((new_width, new_height), Image.BICUBIC)
            bicubic_img.save(output_path)
            
        return True
    except Exception as e:
        print(f"Error creating bicubic reference: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='RCAN Super-Resolution Inference')
    parser.add_argument('--input', required=True, help='Path to input image')
    parser.add_argument('--output', required=True, help='Path to output image')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4], 
                       help='Scale factor (default: 2)')
    parser.add_argument('--temp_results', required=True, help='Temporary results directory')
    
    args = parser.parse_args()
    
    # Record start time
    start_time = time.time()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input image '{args.input}' not found")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create temp_results directory
    os.makedirs(args.temp_results, exist_ok=True)
    
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input = os.path.join(temp_dir, 'input.png')
            temp_output = os.path.join(temp_dir, 'output.png')
            bicubic_reference = os.path.join(temp_dir, 'bicubic.png')
            
            # Copy input to temp directory
            shutil.copy2(args.input, temp_input)
            
            # Create bicubic reference
            create_bicubic_reference(temp_input, bicubic_reference, args.scale)
            
            print(f"Running RCAN inference with scale {args.scale}x...")
            
            # Build TorchSR command
            cmd = [
                sys.executable, '-m', 'torchsr.train',
                '--arch', 'rcan',
                '--scale', str(args.scale),
                '--download-pretrained',
                '--images', temp_input,
                '--destination', temp_dir + '/',
                '--cpu'  # Force CPU usage
            ]
            
            print(f"Command: {' '.join(cmd)}")
            
            # Run TorchSR command
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)
            
            if result.returncode != 0:
                print(f"TorchSR command failed:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                sys.exit(1)
            
            print("TorchSR processing completed successfully")
            
            # Find the generated output file
            output_files = []
            for file in os.listdir(temp_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')) and file != 'input.png' and file != 'bicubic.png':
                    output_files.append(file)
            
            if not output_files:
                print("No output image found")
                sys.exit(1)
            
            # Use the first output file found
            generated_output = os.path.join(temp_dir, output_files[0])
            
            # Copy result to final destination
            shutil.copy2(generated_output, args.output)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Calculate PSNR against bicubic reference
            psnr = calculate_psnr(args.output, bicubic_reference)
            
            # Prepare results
            results = {
                'processing_time': processing_time,  # Already in seconds
                'scale': args.scale,
                'psnr': psnr,
                'output_path': args.output,
                'algorithm': 'rcan'
            }
            
            # Save results to JSON file
            result_file = os.path.join(args.temp_results, 'rcan_results.json')
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to: {result_file}")
            print(f"Processing time: {processing_time:.2f}s")
            if psnr is not None:
                print(f"PSNR vs bicubic: {psnr:.2f} dB")
            print(f"Output saved to: {args.output}")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        # Create a minimal results file even on error
        results = {
            'processing_time': time.time() - start_time,  # Already in seconds
            'scale': args.scale,
            'psnr': None,
            'error': str(e),
            'algorithm': 'rcan'
        }
        
        result_file = os.path.join(args.temp_results, 'rcan_results.json')
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        sys.exit(1)

if __name__ == '__main__':
    main()
