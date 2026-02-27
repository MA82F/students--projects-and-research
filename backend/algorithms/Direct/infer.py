#!/usr/bin/env python3
"""
Direct Upscaling (Bicubic) Custom Wrapper Script
Simple bicubic interpolation for image upscaling
"""

import argparse
import json
import os
import sys
import time
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
        
        # Handle case where images are identical (avoid inf)
        if mse == 0:
            return 100.0  # Return a high but finite PSNR value
        
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

def direct_upscale(input_path, output_path, scale):
    """Direct bicubic upscaling implementation"""
    try:
        # Load image
        img = Image.open(input_path).convert('RGB')
        
        # Get current size
        width, height = img.size
        new_width = width * scale
        new_height = height * scale
        
        # Use bicubic interpolation
        result = img.resize((new_width, new_height), Image.BICUBIC)
        result.save(output_path)
        
        print(f"Direct bicubic upscaling completed")
        return True
        
    except Exception as e:
        print(f"Error in direct upscaling: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Direct Upscaling (Bicubic) Inference')
    parser.add_argument('--input', required=True, help='Path to input image')
    parser.add_argument('--output', required=True, help='Path to output image')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4, 8], 
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
            
            print(f"Running Direct bicubic upscaling with scale {args.scale}x...")
            
            # Run direct upscaling
            success = direct_upscale(temp_input, temp_output, args.scale)
            
            if not success:
                print("Direct upscaling failed")
                sys.exit(1)
            
            # Copy result to final destination
            shutil.copy2(temp_output, args.output)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # For Direct upscaling, we'll compare the upscaled result with a downsampled version
            # This gives us a meaningful PSNR that shows how well the upscaling preserves detail
            try:
                # Create a downsampled version of our upscaled result for comparison
                upscaled_img = Image.open(args.output)
                original_img = Image.open(args.input)
                
                # Downsample the upscaled image back to original size
                downsampled = upscaled_img.resize(original_img.size, Image.LANCZOS)
                downsampled_path = os.path.join(temp_dir, 'downsampled.png')
                downsampled.save(downsampled_path)
                
                # Calculate PSNR between original and downsampled version
                psnr = calculate_psnr(args.input, downsampled_path)
            except Exception as e:
                print(f"Error calculating PSNR: {e}")
                psnr = None
            
            # Prepare results
            results = {
                'processing_time': processing_time * 1000,  # Convert to milliseconds
                'scale': args.scale,
                'psnr': psnr,
                'output_path': args.output,
                'algorithm': 'direct'
            }
            
            # Save results to JSON file
            result_file = os.path.join(args.temp_results, 'direct_results.json')
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to: {result_file}")
            print(f"Processing time: {processing_time:.2f}s")
            if psnr is not None:
                print(f"PSNR (upscale→downscale vs original): {psnr:.2f} dB")
            print(f"Output saved to: {args.output}")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
