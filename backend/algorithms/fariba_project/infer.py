import argparse
import os
import sys
import subprocess
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image, center_crop, resize
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import numpy as np
import math
import time
import uuid
from dotenv import load_dotenv, set_key
from models import Compression, Classification

# Load environment variables
load_dotenv()   

# label names
class_names = ['Faces', 'Faces_easy', 'Leopards', 'Motorbikes', 'accordion',
       'airplanes', 'anchor', 'ant', 'barrel', 'bass', 'beaver',
       'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha',
       'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan',
       'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face',
       'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup',
       'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly',
       'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry',
       'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone',
       'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter',
       'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp',
       'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly',
       'menorah', 'metronome', 'minaret', 'nautilus', 'octopus', 'okapi',
       'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid',
       'revolver', 'rhino', 'rooster', 'saxophone', 'schooner',
       'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball',
       'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry',
       'sunflower', 'tick', 'trilobite', 'umbrella', 'watch',
       'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench',
       'yin_yang']
lb = LabelEncoder()
lb.classes_ = np.array(class_names)

def load_image(image_path, patch_size=224):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    w = min(width, height)
    cropped = center_crop(image, [w])  # This is the original center crop at full resolution
    resized = resize(cropped, [patch_size])  # This is the 224x224 version for model
    tensor_image = to_tensor(resized).unsqueeze(0)  
    return tensor_image, resized, cropped

def save_image(tensor, output_path):
    tensor = tensor.squeeze(0).clamp(0, 1)
    image = to_pil_image(tensor.cpu())
    image.save(output_path)

def run_hyperIQA_for_fariba(image_path):
    """Run hyperIQA on the given image and return quality score"""
    try:
        # Get the backend directory (3 levels up from this file)
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        hyperIQA_script = os.path.join(backend_dir, 'algorithms', 'noRefrence_imageQuality', 'hyperIQA', 'demo.py')
        
        if not os.path.exists(hyperIQA_script):
            print(f"hyperIQA script not found: {hyperIQA_script}")
            return None
            
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None
            
        # Run hyperIQA without JSON output (use old method)
        python_executable = sys.executable
        command = [python_executable, hyperIQA_script, '-i', image_path]
        
        result = subprocess.run(command, capture_output=True, text=True, cwd=backend_dir)
        
        if result.returncode == 0:
            # Parse the quality score from stdout
            for line in result.stdout.split('\n'):
                if 'Predicted quality score:' in line:
                    try:
                        score = float(line.split(':')[1].strip())
                        return score
                    except:
                        pass
        else:
            print(f"hyperIQA error: {result.stderr}")
            
    except Exception as e:
        print(f"Error running hyperIQA: {e}")
        
    return None

def save_results_to_json(input_path, output_path, pred_label, pred_index, psnr, bpp, decode_time, unique_id, cropped_filename=None):
    """Save inference results to a unique JSON file"""
    # Get the directory containing the backend folder
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    temp_dir = os.path.join(backend_dir, 'temp_results')
    os.makedirs(temp_dir, exist_ok=True)
    
    results = {
        'input_path': input_path,
        'output_path': output_path,
        'predicted_class': pred_label,
        'predicted_index': pred_index,
        'psnr': psnr,
        'bitrate': bpp,
        'decode_time': decode_time
    }
    
    # Add cropped filename if provided
    if cropped_filename:
        results['cropped_filename'] = cropped_filename
    
    # Run hyperIQA on the output image and add quality score
    print("Running hyperIQA for image quality assessment...")
    image_quality = run_hyperIQA_for_fariba(output_path)
    if image_quality is not None:
        results['image_quality'] = image_quality
        print(f"Image quality score: {image_quality:.3f}")
    else:
        print("Failed to get image quality score")
    
    # Save to unique file based on unique_id
    result_file = os.path.join(temp_dir, f'fariba_results_{unique_id}.json')
    import json
    with open(result_file, 'w') as f:
        json.dump(results, f)

def compute_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float("inf")
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

def compute_bpp(likelihoods, img_shape):
    N, C, H, W = img_shape
    num_pixels = N * H * W
    bpp = sum(torch.log(likelihood).sum() / (-math.log(2) * num_pixels)
              for likelihood in likelihoods.values()).item()
    return bpp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--quality", type=int, choices=[1, 2, 3, 4, 5], required=True, 
                        help="Quality level (1-5)")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory")
    parser.add_argument("--unique-id", type=str, required=True, help="Unique identifier for this request")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output, exist_ok=True)

    # Map quality to weight paths
    quality_to_weights = {
        # 1: "C:\\Users\\Asus\\Desktop\\project\\checkpoints\\focalq1.pth.tar",
        # 2: "C:\\Users\\Asus\\Desktop\\project\\checkpoints\\focalq2.pth.tar",
        # 3: "C:\\Users\\Asus\\Desktop\\project\\checkpoints\\focalq3.pth.tar",
        # 4: "C:\\Users\\Asus\\Desktop\\project\\checkpoints\\focalq4.pth.tar",
        # 5: "C:\\Users\\Asus\\Desktop\\project\\checkpoints\\focalq5.pth.tar"
        1: os.path.join(os.path.dirname(__file__), "checkpoints", "focalq1.pth.tar"),
        2: os.path.join(os.path.dirname(__file__), "checkpoints", "focalq2.pth.tar"),
        3: os.path.join(os.path.dirname(__file__), "checkpoints", "focalq3.pth.tar"),
        4: os.path.join(os.path.dirname(__file__), "checkpoints", "focalq4.pth.tar"),
        5: os.path.join(os.path.dirname(__file__), "checkpoints", "focalq5.pth.tar")
    }

    weights_path = quality_to_weights[args.quality]

    # Load image
    input_tensor, resized_224, original_cropped = load_image(args.input)
    input_tensor = input_tensor.to(device)

    # Load models
    compression_model = Compression(N=128, M=192).to(device)
    classification_model = Classification(num_classes=101).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    compression_model.load_state_dict(checkpoint["state_dict1"])
    classification_model.load_state_dict(checkpoint["state_dict2"])
    compression_model.eval()
    classification_model.eval()

    with torch.no_grad():
        # Compression model
        start_decode = time.time()
        compressed = compression_model(input_tensor)
        x_hat = compressed["x_hat"].clamp(0, 1)
        decode_time = time.time() - start_decode

        # Classification model
        class_output = classification_model(compressed["y_hat"])["class_output"]
        pred_index = torch.argmax(F.softmax(class_output, dim=1), dim=1).item()
        pred_label = lb.inverse_transform([pred_index])[0]

        # Save reconstructed image with random filename
        base_name, ext = os.path.splitext(os.path.basename(args.input))
        random_id = str(uuid.uuid4())[:8]
        output_filename = f"{base_name}_fariba_{random_id}.png"
        recon_path = os.path.join(args.output, output_filename)
        save_image(x_hat, recon_path)

        # Save center cropped image to uploads directory
        uploads_dir = os.path.join(os.path.dirname(args.output), 'uploads')
        cropped_filename = f"{base_name}_cropped.png"
        cropped_path = os.path.join(uploads_dir, cropped_filename)
        resized_224.save(cropped_path)
        print(f"DEBUG FARIBA: Center cropped image saved to {cropped_path}")

        # PSNR
        psnr = compute_psnr(x_hat, input_tensor)

        # Bitrate
        bpp = compute_bpp(compressed["likelihoods"], input_tensor.shape)

        # Save results to JSON file with unique ID
        save_results_to_json(args.input, recon_path, pred_label, pred_index, psnr, bpp, decode_time*1000, args.unique_id, cropped_filename)

    print("=== Inference Results ===")
    print(f"Input Image: {args.input}")
    print(f"Reconstructed Image Saved to: {recon_path}")
    print(f"Predicted Class: {pred_label} (index: {pred_index})")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Bitrate: {bpp:.4f} bpp")
    print(f"Decoding Time: {decode_time*1000:.1f} ms")

if __name__ == "__main__":
    main()