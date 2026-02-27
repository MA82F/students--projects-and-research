# Import Dependencies
import os
import cv2
import PIL.Image as Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import json
from model import ESPCN
from utils import calculate_psnr
from dotenv import load_dotenv, set_key

# Load environment variables
load_dotenv()


def prepare_image(hr_image, device, args):
    """ Function to prepare hr/lr/bicubic images for inference and performance comparison

        hr_image: high resolution input image
        device: 'cpu', 'cuda'
        args: command line arguments
        returns: lr/hr Y channel, bicubic YCbCr and original Bicubic Images
    """

    # Load HR image: rH x rW x C, r: scaling factor (fixed to 3)
    scaling_factor = 3
    hr_width = (hr_image.width // scaling_factor) * scaling_factor
    hr_height = (hr_image.height // scaling_factor) * scaling_factor
    hr_image = hr_image.resize((hr_width, hr_height), resample=Image.BICUBIC)

    # LR Image: H x W x C
    # As in paper, Sec. 3.2: sub-sample images by up-scaling factor
    lr_image = hr_image.resize((hr_image.width // scaling_factor, hr_image.height // scaling_factor),
                            resample=Image.BICUBIC)

    # Save LR image to uploads directory for user to see
    input_filename = os.path.basename(args.fpath_image)
    base_name, ext = os.path.splitext(input_filename)
    ext = ext.lower() if ext else '.png'  # Default to .png if no extension
    
    # Save to uploads directory
    uploads_dir = os.path.join(os.path.dirname(args.dirpath_out), 'uploads')
    lr_filename = f"{base_name}_lr_x3{ext}"
    lr_path = os.path.join(uploads_dir, lr_filename)
    lr_image.save(lr_path)
    print(f"DEBUG ESPCN: Low resolution image saved to {lr_path}")

    # Generate Bicubic image for performance comparison
    bicubic_image = lr_image.resize((lr_image.width * scaling_factor, lr_image.height * scaling_factor),
                                    resample=Image.BICUBIC)
    
    # Get original file extension and name for bicubic
    input_filename = os.path.basename(args.fpath_image)
    base_name, ext = os.path.splitext(input_filename)
    ext = ext.lower() if ext else '.png'  # Default to .png if no extension
    
    bicubic_filename = f"{base_name}_bicubic_x3{ext}"
    bicubic_image.save(os.path.join(
        args.dirpath_out,
        bicubic_filename
    ))

    # Convert PIL image to numpy array
    hr_image = np.array(hr_image).astype(np.float32)
    lr_image = np.array(lr_image).astype(np.float32)
    bicubic_image = np.array(bicubic_image).astype(np.float32)

    # Convert RGB to YCbCr
    hr_image = cv2.cvtColor(hr_image, cv2.COLOR_RGB2YCrCb)
    lr_image = cv2.cvtColor(lr_image, cv2.COLOR_RGB2YCrCb)
    bicubic_image_ycrcb = cv2.cvtColor(bicubic_image, cv2.COLOR_RGB2YCrCb)

    # As per paper, using only the luminescence channel gave the best outcome
    hr_y = hr_image[:, :, 0]
    lr_y = lr_image[:, :, 0]

    # Normalize images
    hr_y /= 255.
    lr_y /= 255.
    bicubic_image /= 255.

    # Convert Numpy to Torch Tensor and send to device
    hr_y = torch.from_numpy(hr_y).to(device)
    hr_y = hr_y.unsqueeze(0).unsqueeze(0)

    lr_y = torch.from_numpy(lr_y).to(device)
    lr_y = lr_y.unsqueeze(0).unsqueeze(0)

    return lr_y, hr_y, bicubic_image_ycrcb, bicubic_image, lr_filename, bicubic_filename


def infer(args):
    """ Function to perform inference on test images

        args
    """
    import time
    start_time = time.time()
    
    print(f"DEBUG ESPCN: Starting inference with args: {vars(args)}")

    # Select Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    # Load Model
    # Note: The pre-trained model was trained with scaling_factor=3
    # For now, we'll use the trained scaling factor and apply bicubic interpolation for other scales
    trained_scaling_factor = 3
    model = ESPCN(num_channels=1, scaling_factor=trained_scaling_factor)
    model.load_state_dict(torch.load(args.fpath_weights))
    model.to(device)
    model.eval()

    # Read & Prepare Image for Inference
    # Load HR image: rH x rW x C, r: scaling factor
    hr_image = Image.open(args.fpath_image).convert('RGB')
    lr_y, hr_y, ycbcr, bicubic_image, lr_filename, bicubic_filename = prepare_image(hr_image, device, args)

    with torch.no_grad():
        preds = model(lr_y)

    psnr_hr_sr = calculate_psnr(hr_y, preds)
    print('PSNR (HR/SR): {:.2f}'.format(psnr_hr_sr))
    
    # Save output image
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(cv2.cvtColor(output, cv2.COLOR_YCrCb2RGB), 0.0, 255.0).astype(np.uint8)
    output = Image.fromarray(output)
    
    # Handle different scaling factors
    # ESPCN is fixed to 3x scaling factor
    print(f"DEBUG ESPCN: Using fixed 3x scaling factor")
    
    # Get original file extension and name
    input_filename = os.path.basename(args.fpath_image)
    base_name, ext = os.path.splitext(input_filename)
    ext = ext.lower() if ext else '.png'  # Default to .png if no extension
    
    output_filename = f"{base_name}_espcn_x3{ext}"
    output_path = os.path.join(args.dirpath_out, output_filename)
    output.save(output_path)
    
    # Save results to JSON file if unique_id is provided
    if args.unique_id:
        print(f"DEBUG ESPCN: Saving results to JSON with unique_id: {args.unique_id}")
        # Create temp_results directory
        temp_results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'temp_results')
        os.makedirs(temp_results_dir, exist_ok=True)
        
        # Prepare results data
        processing_time = time.time() - start_time
        results_data = {
            'psnr': float(psnr_hr_sr),
            'output_path': output_path,
            'output_filename': output_filename,
            'lr_filename': lr_filename,
            'bicubic_filename': bicubic_filename,
            'algorithm': 'espcn',
            'scaling_factor': 3,
            'processing_time': processing_time
        }
        
        print(f"DEBUG ESPCN: Results data: {results_data}")
        
        # Save to JSON file
        json_file_path = os.path.join(temp_results_dir, f'espcn_results_{args.unique_id}.json')
        with open(json_file_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"DEBUG ESPCN: Results saved to: {json_file_path}")
    else:
        print(f"DEBUG ESPCN: No unique_id provided, using fallback .env method")
        # Fallback: Save PSNR to .env file (for backward compatibility)
        env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
        set_key(env_path, 'PSNR_RESULT', str(float(psnr_hr_sr)))

    # Plot Image Comparison
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(np.array(hr_image))
    ax1.set_title("HR Image")
    ax2.imshow(bicubic_image)
    ax2.set_title("Bicubic Image x3")
    ax3.imshow(np.array(output))
    ax3.set_title("SR Image x3 (PSNR: {:.2f} dB)".format(psnr_hr_sr))
    fig.suptitle('ESPCN Single Image Super Resolution')
    # plt.show()
    fig.set_size_inches(20, 10, forward=True)
    
    # Use same base name and extension as input for result plot
    result_plot_filename = f"{base_name}_result{ext}"
    fig.savefig(os.path.join(args.dirpath_out, result_plot_filename), dpi=100)

    # Update results_data with result plot filename if unique_id is provided
    if args.unique_id:
        # Re-read and update the JSON file to include result plot filename
        json_file_path = os.path.join(temp_results_dir, f'espcn_results_{args.unique_id}.json')
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                results_data = json.load(f)
            
            # Add result plot filename
            results_data['result_plot_filename'] = result_plot_filename
            
            # Save updated data
            with open(json_file_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            print(f"DEBUG ESPCN: Updated results with result plot: {result_plot_filename}")


def build_parser():
    parser = ArgumentParser(prog="ESPCN Inference")
    parser.add_argument("-w", "--fpath_weights", required=True, type=str,
                        help="Required. Path to trained model weights.")
    parser.add_argument("-i", "--fpath_image", required=True, type=str,
                        help="Required. Path to image file to perform inference on.")
    parser.add_argument("-o", "--dirpath_out", required=True, type=str,
                        help="Required. Path to output image directory.")
    parser.add_argument("-sf", "--scaling_factor", default=3, required=False, type=int,
                        help="Optional. Image Up-scaling factor.")
    parser.add_argument("--unique-id", type=str, default=None,
                        help="Optional. Unique ID for this processing session.")

    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    infer(args)
