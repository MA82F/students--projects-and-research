import torch
import torchvision
import models
from PIL import Image
import numpy as np
import argparse
import os

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def main():
    parser = argparse.ArgumentParser(description='HyperIQA Image Quality Assessment')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input image')
    parser.add_argument('-o', '--output', type=str, help='Path to output JSON file for results')
    args = parser.parse_args()

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return

    im_path = args.input
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)
    model_hyper.train(False)
    
    # load our pre-trained model on the koniq-10k dataset
    # Use absolute path to find the pretrained model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(script_dir, 'pretrained', 'koniq_pretrained.pkl')
    
    if not os.path.exists(pretrained_path):
        print(f"Error: Pretrained model '{pretrained_path}' not found")
        return
        
    model_hyper.load_state_dict((torch.load(pretrained_path, map_location=device)))

    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.Resize((512, 384)),
                        torchvision.transforms.RandomCrop(size=224),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                         std=(0.229, 0.224, 0.225))])

    # random crop 10 patches and calculate mean quality score
    pred_scores = []
    for i in range(10):
        img = pil_loader(im_path)
        img = transforms(img)
        img = torch.tensor(img).unsqueeze(0).to(device)
        paras = model_hyper(img)  # 'paras' contains the network weights conveyed to target network

        # Building target network
        model_target = models.TargetNet(paras).to(device)
        for param in model_target.parameters():
            param.requires_grad = False

        # Quality prediction
        pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
        pred_scores.append(float(pred.item()))
    
    score = np.mean(pred_scores)
    
    # Save quality score to JSON file if output path is provided
    if args.output:
        import json
        
        # Read existing JSON file if it exists
        if os.path.exists(args.output):
            with open(args.output, 'r') as f:
                results_data = json.load(f)
        else:
            results_data = {}
        
        # Add quality score to the results
        results_data['image_quality'] = float(score)
        
        # Save updated results back to JSON file
        with open(args.output, 'w') as f:
            json.dump(results_data, f)
        
        print(f'Quality score saved to JSON file: {args.output}')
    else:
        # No output file provided - just print the score
        print(f'No output file specified - quality score not saved')
    
    # quality score ranges from 0-100, a higher score indicates a better quality
    print(f'Input Image: {im_path}')
    print(f'Predicted quality score: {score:.2f}')

if __name__ == "__main__":
    main()

