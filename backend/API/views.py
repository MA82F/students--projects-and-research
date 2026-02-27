import os
import sys
import subprocess
import uuid
import json
import shutil
import time
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from .serializers import ImageUploadSerializer, ComparisonSerializer
from dotenv import load_dotenv, set_key, dotenv_values

# Load environment variables
load_dotenv()

def generate_unique_filename(input_filename, algorithm, scale=None):
    """Generate unique filename for output images"""
    # Get base name and extension
    base_name = os.path.splitext(input_filename)[0]
    ext = os.path.splitext(input_filename)[1] or '.png'
    
    # Generate unique ID
    unique_id = str(uuid.uuid4())[:8]
    
    # Create filename with algorithm and scale
    if scale:
        output_filename = f"{base_name}_{unique_id}_{algorithm}_x{scale}{ext}"
    else:
        output_filename = f"{base_name}_{unique_id}_{algorithm}{ext}"
    
    return output_filename

def run_hyperIQA(image_path, results_file_path):
    """
    Run hyperIQA quality assessment on processed image
    Saves quality score to JSON file and returns quality score
    """
    try:
        print(f"Running hyperIQA on: {image_path}")
        
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None
            
        python_executable = sys.executable
        command = [
            python_executable,
            'algorithms/noRefrence_imageQuality/hyperIQA/demo.py',
            '-i', image_path,
            '-o', results_file_path
        ]
        
        print(f"Running command: {' '.join(command)}")
        
        # Run hyperIQA
        result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=settings.BASE_DIR)
        
        print(f"hyperIQA stdout: {result.stdout}")
        print(f"hyperIQA stderr: {result.stderr}")
        print(f"hyperIQA return code: {result.returncode}")
        
        # Read quality score from JSON file
        if os.path.exists(results_file_path):
            with open(results_file_path, 'r') as f:
                results_data = json.load(f)
            
            quality_score = results_data.get('image_quality')
            print(f"Quality score from JSON: {quality_score}")
            
            if quality_score is not None:
                return float(quality_score)
            
    except Exception as e:
        print(f"Error running hyperIQA: {e}")
        
    return None

class ImageUploadView(APIView):
    def post(self, request, *args, **kwargs):
        response_data = {}
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            # Save uploaded file directly without database
            uploaded_file = serializer.validated_data['image']
            algorithm = serializer.validated_data.get('algorithm', 'espcn').lower()
            scale = serializer.validated_data.get('scale', '2')
            
            # Create unique filename
            unique_id = str(uuid.uuid4())[:8]
            base_name, ext = os.path.splitext(uploaded_file.name)
            input_filename = f"{base_name}_{unique_id}{ext}"
            
            # Save file directly to uploads directory
            input_path = default_storage.save(f'uploads/{input_filename}', uploaded_file)
            full_input_path = os.path.join(settings.MEDIA_ROOT, input_path)
            
            print(f"DEBUG: Saved file to: {full_input_path}")
            print(f"DEBUG: Received algorithm: {algorithm}")

            # Output directory
            output_dir = os.path.join(settings.MEDIA_ROOT, 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            # List of supported torchSR algorithms
            torchsr_algorithms = [
            ]
            
            # EDSR algorithms (using custom wrapper)
            edsr_algorithms = ['edsr', 'edsr_baseline']
            
            # RDN algorithms (using custom wrapper)
            rdn_algorithms = ['rdn']
            
            # RCAN algorithms (using custom wrapper)
            rcan_algorithms = ['rcan']
            
            # CARN algorithms (using custom wrapper)
            carn_algorithms = ['carn', 'carn_m']
            
            # VDSR algorithms (using custom wrapper)
            vdsr_algorithms = ['vdsr']
            
            # Direct algorithms (using custom wrapper)
            direct_algorithms = ['direct']
            
            # Fariba algorithm
            fariba_algorithms = ['fariba']
            
            # ESPCN algorithm
            espcn_algorithms = ['espcn']
            
            # NinaSR algorithm
            ninasr_algorithms = ['ninasr']
            
            # Initialize variables for cleanup
            unique_id = str(uuid.uuid4())[:8]  # Generate unique ID for all algorithms
            result_file_path = None
            output_filename = None  # Will be set per algorithm
            
            # In the ImageUploadView class, modify the command for torchSR algorithms:
            python_executable = sys.executable
            env = os.environ.copy()  # Ensure env is always defined
            if algorithm in torchsr_algorithms:
                command = [
                    python_executable, '-m', 'torchsr.train',
                    '--arch', algorithm,
                    '--scale', scale,
                    '--download-pretrained',
                    '--images', full_input_path,
                    '--destination', output_dir,
                    '--cpu'
                ]
            elif algorithm in fariba_algorithms:
                # For Fariba algorithm, use scale as quality level
                quality = request.data.get('scale', '1')  # Use scale field as quality
                command = [
                    python_executable,
                    'algorithms/fariba_project/infer.py',
                    '-q', quality,
                    '-i', full_input_path,
                    '-o', output_dir,
                    '--unique-id', unique_id
                ]
            elif algorithm in espcn_algorithms:
                # ESPCN command - fixed to 3x scaling
                print(f"DEBUG: Processing ESPCN algorithm with fixed 3x scaling")  # Debug log
                command = [
                    'python',
                    'algorithms/ESPCN/infer.py',
                    '-w', 'algorithms/ESPCN/assets/models/best_model.pth',
                    '-i', full_input_path,
                    '-o', output_dir,
                    '--unique-id', unique_id
                ]
            elif algorithm in edsr_algorithms:
                # EDSR command with configurable scale
                scale_factor = request.data.get('scale', '2')  # Default to 2x
                print(f"DEBUG: Processing EDSR algorithm {algorithm} with scale={scale_factor}")
                command = [
                    python_executable,
                    'algorithms/EDSR/infer.py',
                    '-i', full_input_path,
                    '--output-dir', output_dir,
                    '--scale', scale_factor,
                    '--arch', algorithm,
                    '--unique-id', unique_id
                ]
            elif algorithm in rdn_algorithms:
                # RDN command with configurable scale
                scale_factor = request.data.get('scale', '2')  # Default to 2x
                print(f"DEBUG: Processing RDN algorithm with scale={scale_factor}")
                temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
                os.makedirs(temp_dir, exist_ok=True)
                
                # Generate unique output filename
                output_filename = generate_unique_filename(input_filename, 'rdn', scale_factor)
                
                command = [
                    python_executable,
                    'algorithms/RDN/infer.py',
                    '--input', full_input_path,
                    '--output', os.path.join(output_dir, output_filename),
                    '--scale', scale_factor,
                    '--temp_results', temp_dir
                ]
            elif algorithm in rcan_algorithms:
                # RCAN command with configurable scale
                scale_factor = request.data.get('scale', '2')  # Default to 2x
                print(f"DEBUG: Processing RCAN algorithm with scale={scale_factor}")
                temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
                os.makedirs(temp_dir, exist_ok=True)
                
                # Generate unique output filename
                output_filename = generate_unique_filename(input_filename, 'rcan', scale_factor)
                
                command = [
                    python_executable,
                    'algorithms/RCAN/infer.py',
                    '--input', full_input_path,
                    '--output', os.path.join(output_dir, output_filename),
                    '--scale', scale_factor,
                    '--temp_results', temp_dir
                ]
            elif algorithm in carn_algorithms:
                # CARN command with configurable scale and model variant
                scale_factor = request.data.get('scale', '2')  # Default to 2x
                model_variant = 'carn' if algorithm == 'carn' else 'carn_m'
                print(f"DEBUG: Processing CARN algorithm with scale={scale_factor}, model={model_variant}")
                
                # Generate unique output filename
                output_filename = generate_unique_filename(input_filename, model_variant, scale_factor)
                
                command = [
                    python_executable,
                    'algorithms/CARN/infer.py',
                    '--input', full_input_path,
                    '--output', os.path.join(output_dir, output_filename),
                    '--scale', scale_factor,
                    '--model', model_variant
                ]
            elif algorithm in vdsr_algorithms:
                # VDSR command with configurable scale
                scale_factor = request.data.get('scale', '2')  # Default to 2x
                print(f"DEBUG: Processing VDSR algorithm with scale={scale_factor}")
                
                # Generate unique output filename
                output_filename = generate_unique_filename(input_filename, 'vdsr', scale_factor)
                
                command = [
                    python_executable,
                    'algorithms/VDSR/infer.py',
                    '--input', full_input_path,
                    '--output', os.path.join(output_dir, output_filename),
                    '--scale', scale_factor
                ]
            elif algorithm in direct_algorithms:
                # Direct Upscaling command with configurable scale
                scale_factor = request.data.get('scale', '2')  # Default to 2x
                print(f"DEBUG: Processing Direct Upscaling with scale={scale_factor}")
                temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
                os.makedirs(temp_dir, exist_ok=True)
                
                # Generate unique output filename
                output_filename = generate_unique_filename(input_filename, 'direct', scale_factor)
                
                command = [
                    python_executable,
                    'algorithms/NinaSR/infer.py',
                    '--input', full_input_path,
                    '--output', os.path.join(output_dir, output_filename),
                    '--scale', scale_factor,
                    '--temp_results', temp_dir
                ]
            elif algorithm in ninasr_algorithms:
                # NinaSR command with configurable scale and variant
                variant = request.data.get('variant', 'b1')  # Default to B1
                scale_factor = request.data.get('scale', '3')  # Default to 3x
                print(f"DEBUG: Processing NinaSR algorithm with variant={variant}, scale={scale_factor}")
                command = [
                    python_executable,
                    'algorithms/NinaSR/infer.py',
                    '-i', full_input_path,
                    '-o', output_dir,
                    '-s', scale_factor,
                    '-v', variant,
                    '--unique-id', unique_id
                ]
            else:
                # Unknown algorithm, return error
                return Response({'error': f'Unsupported algorithm: {algorithm}'}, status=400)
            
            # For fariba algorithm, set up result file path for cleanup
            if algorithm in fariba_algorithms:
                temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
                result_file_path = os.path.join(temp_dir, f'fariba_results_{unique_id}.json')
            elif algorithm in espcn_algorithms:
                temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
                result_file_path = os.path.join(temp_dir, f'espcn_results_{unique_id}.json')
            elif algorithm in edsr_algorithms:
                temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
                result_file_path = os.path.join(temp_dir, f'edsr_results_{unique_id}.json')
            elif algorithm in ninasr_algorithms:
                temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
                result_file_path = os.path.join(temp_dir, f'ninasr_results_{unique_id}.json')
            else:
                result_file_path = None
            
            try:
                print(f"DEBUG: Running command: {' '.join(command)}")  # Debug log
                subprocess.run(command, check=True, env=env)

                # Create output filenames based on input filename
                base_name, ext = os.path.splitext(os.path.basename(input_path))
                ext = ext.lower() if ext else '.png'  # Default to .png if no extension
                if algorithm in torchsr_algorithms:
                    output_filename = f"{base_name}_{algorithm}_x{scale}{ext}"
                    response_data['processed_image'] = f"{settings.MEDIA_URL}output/{output_filename}"
                    response_data['input_image'] = f"{settings.MEDIA_URL}uploads/{input_filename}"
                    
                    # Create temporary JSON file for results
                    temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_results_file = os.path.join(temp_dir, f'torchsr_results_{unique_id}.json')
                    
                    # Run hyperIQA on processed image
                    output_image_path = os.path.join(output_dir, output_filename)
                    quality_score = run_hyperIQA(output_image_path, temp_results_file)
                    if quality_score is not None:
                        response_data['image_quality'] = quality_score
                    
                    # Read PSNR from .env file for torchSR algorithms too
                    env_path = os.path.join(settings.BASE_DIR, '.env')
                    env_vars = dotenv_values(env_path)
                    psnr_value = env_vars.get('PSNR_RESULT')
                    if psnr_value and psnr_value.strip():
                        response_data['psnr'] = float(psnr_value)
                        # Clear the PSNR_RESULT from .env file after reading
                        set_key(env_path, 'PSNR_RESULT', '')
                    
                    # Clean up temporary file
                    if os.path.exists(temp_results_file):
                        os.remove(temp_results_file)
                elif algorithm in fariba_algorithms:
                    # Read Fariba algorithm results from JSON file
                    temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
                    result_file = os.path.join(temp_dir, f'fariba_results_{unique_id}.json')
                    
                    if os.path.exists(result_file):
                        with open(result_file, 'r') as f:
                            results = json.load(f)
                        
                        # Extract output filename
                        output_path = results.get('output_path', '')
                        if output_path:
                            output_filename = os.path.basename(output_path)
                            response_data['processed_image'] = f"{settings.MEDIA_URL}output/{output_filename}"
                        
                        response_data['input_image'] = f"{settings.MEDIA_URL}uploads/{input_filename}"
                        
                        # Add all Fariba results to response (including image_quality from JSON)
                        if results.get('predicted_class'):
                            response_data['predicted_class'] = results.get('predicted_class')
                        if results.get('predicted_index') is not None:
                            response_data['predicted_index'] = int(results.get('predicted_index'))
                        if results.get('psnr') is not None:
                            response_data['psnr'] = float(results.get('psnr'))
                        if results.get('cropped_filename'):
                            response_data['cropped_filename'] = results.get('cropped_filename')
                            print(f"DEBUG: Set cropped_filename to: {response_data['cropped_filename']}")  # Debug log
                        if results.get('bitrate') is not None:
                            response_data['bitrate'] = float(results.get('bitrate'))
                        if results.get('decode_time') is not None:
                            response_data['decode_time'] = float(results.get('decode_time'))
                        if results.get('image_quality') is not None:
                            response_data['image_quality'] = float(results.get('image_quality'))
                        
                        # Clean up the temporary file
                        os.remove(result_file)
                elif algorithm in ninasr_algorithms:
                    # Read NinaSR algorithm results from JSON file
                    print(f"DEBUG: Reading NinaSR results for unique_id: {unique_id}")
                    temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
                    result_file = os.path.join(temp_dir, f'ninasr_results_{unique_id}.json')
                    
                    print(f"DEBUG: Looking for NinaSR result file: {result_file}")
                    
                    if os.path.exists(result_file):
                        with open(result_file, 'r') as f:
                            results = json.load(f)
                        
                        print(f"DEBUG: NinaSR JSON results: {results}")
                        
                        # Extract output filename
                        output_path = results.get('output_path', '')
                        if output_path:
                            output_filename = os.path.basename(output_path)
                            response_data['processed_image'] = f"{settings.MEDIA_URL}output/{output_filename}"
                        
                        response_data['input_image'] = f"{settings.MEDIA_URL}uploads/{input_filename}"
                        
                        # Add NinaSR results to response
                        if results.get('psnr') is not None:
                            response_data['psnr'] = float(results.get('psnr'))
                        if results.get('processing_time') is not None:
                            response_data['processing_time'] = float(results.get('processing_time'))
                        if results.get('scale_factor') is not None:
                            response_data['scale_factor'] = results.get('scale_factor')
                        if results.get('variant') is not None:
                            response_data['variant'] = results.get('variant')
                        
                        # Add lr_filename to response for NinaSR
                        if results.get('lr_filename'):
                            response_data['lr_filename'] = results.get('lr_filename')
                            print(f"DEBUG: Set lr_filename to: {response_data['lr_filename']}")
                        
                        # Run hyperIQA on NinaSR processed image
                        if output_path and os.path.exists(output_path):
                            temp_results_file = os.path.join(temp_dir, f'ninasr_hyperIQA_{unique_id}.json')
                            quality_score = run_hyperIQA(output_path, temp_results_file)
                            if quality_score is not None:
                                response_data['image_quality'] = quality_score
                            
                            # Clean up temporary hyperIQA file
                            if os.path.exists(temp_results_file):
                                os.remove(temp_results_file)
                        
                        # Clean up the temporary file
                        os.remove(result_file)
                        print(f"DEBUG: Cleaned up NinaSR result file")
                    else:
                        print(f"DEBUG: NinaSR JSON file not found")
                        # Fallback response
                        response_data['input_image'] = f"{settings.MEDIA_URL}uploads/{input_filename}"
                elif algorithm in edsr_algorithms:
                    # Read EDSR algorithm results from JSON file
                    print(f"DEBUG: Reading EDSR results for unique_id: {unique_id}")
                    temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
                    result_file = os.path.join(temp_dir, f'edsr_results_{unique_id}.json')
                    
                    print(f"DEBUG: Looking for EDSR result file: {result_file}")
                    
                    if os.path.exists(result_file):
                        with open(result_file, 'r') as f:
                            results = json.load(f)
                        
                        print(f"DEBUG: EDSR JSON results: {results}")
                        
                        # Extract output filename
                        output_path = results.get('output_path', '')
                        if output_path:
                            output_filename = os.path.basename(output_path)
                            response_data['processed_image'] = f"{settings.MEDIA_URL}output/{output_filename}"
                        
                        response_data['input_image'] = f"{settings.MEDIA_URL}uploads/{input_filename}"
                        
                        # Add EDSR results to response
                        if results.get('psnr') is not None:
                            response_data['psnr'] = float(results.get('psnr'))
                        if results.get('processing_time') is not None:
                            response_data['processing_time'] = float(results.get('processing_time'))
                        if results.get('scale_factor') is not None:
                            response_data['scale_factor'] = results.get('scale_factor')
                        if results.get('architecture') is not None:
                            response_data['architecture'] = results.get('architecture')
                        
                        # Run hyperIQA on EDSR processed image
                        if output_path and os.path.exists(output_path):
                            temp_results_file = os.path.join(temp_dir, f'edsr_hyperIQA_{unique_id}.json')
                            quality_score = run_hyperIQA(output_path, temp_results_file)
                            if quality_score is not None:
                                response_data['image_quality'] = quality_score
                            
                            # Clean up temporary hyperIQA file
                            if os.path.exists(temp_results_file):
                                os.remove(temp_results_file)
                        
                        # Clean up the temporary file
                        os.remove(result_file)
                        print(f"DEBUG: Cleaned up EDSR result file")
                    else:
                        print(f"DEBUG: EDSR JSON file not found")
                        # Fallback response
                        response_data['input_image'] = f"{settings.MEDIA_URL}uploads/{input_filename}"
                elif algorithm in rdn_algorithms:
                    # Read RDN algorithm results from JSON file
                    print(f"DEBUG: Reading RDN results")
                    temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
                    result_file = os.path.join(temp_dir, 'rdn_results.json')
                    
                    print(f"DEBUG: Looking for RDN result file: {result_file}")
                    
                    if os.path.exists(result_file):
                        with open(result_file, 'r') as f:
                            results = json.load(f)
                        
                        print(f"DEBUG: RDN JSON results: {results}")
                        
                        # Use the unique output filename generated earlier
                        response_data['processed_image'] = f"{settings.MEDIA_URL}output/{output_filename}"
                        response_data['input_image'] = f"{settings.MEDIA_URL}uploads/{input_filename}"
                        
                        # Add RDN results to response
                        if results.get('psnr') is not None:
                            response_data['psnr'] = float(results.get('psnr'))
                        if results.get('processing_time') is not None:
                            response_data['processing_time'] = float(results.get('processing_time'))
                        if results.get('scale') is not None:
                            response_data['scale_factor'] = results.get('scale')
                        
                        # Run hyperIQA on RDN processed image
                        output_path = os.path.join(output_dir, output_filename)
                        if os.path.exists(output_path):
                            temp_results_file = os.path.join(temp_dir, f'rdn_hyperIQA_{unique_id}.json')
                            quality_score = run_hyperIQA(output_path, temp_results_file)
                            if quality_score is not None:
                                response_data['image_quality'] = quality_score
                            
                            # Clean up temporary hyperIQA file
                            if os.path.exists(temp_results_file):
                                os.remove(temp_results_file)
                        
                        # Clean up the temporary file
                        os.remove(result_file)
                        print(f"DEBUG: Cleaned up RDN result file")
                    else:
                        print(f"DEBUG: RDN JSON file not found")
                        # Fallback response
                        response_data['input_image'] = f"{settings.MEDIA_URL}uploads/{input_filename}"
                elif algorithm in rcan_algorithms:
                    # Read RCAN algorithm results from JSON file
                    print(f"DEBUG: Reading RCAN results")
                    temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
                    result_file = os.path.join(temp_dir, 'rcan_results.json')
                    
                    print(f"DEBUG: Looking for RCAN result file: {result_file}")
                    
                    if os.path.exists(result_file):
                        with open(result_file, 'r') as f:
                            results = json.load(f)
                        
                        print(f"DEBUG: RCAN JSON results: {results}")
                        
                        # Use the unique output filename generated earlier
                        response_data['processed_image'] = f"{settings.MEDIA_URL}output/{output_filename}"
                        response_data['input_image'] = f"{settings.MEDIA_URL}uploads/{input_filename}"
                        
                        # Add RCAN results to response
                        if results.get('psnr') is not None:
                            response_data['psnr'] = float(results.get('psnr'))
                        if results.get('processing_time') is not None:
                            response_data['processing_time'] = float(results.get('processing_time'))
                        if results.get('scale') is not None:
                            response_data['scale_factor'] = results.get('scale')
                        
                        # Run hyperIQA on RCAN processed image
                        output_path = os.path.join(output_dir, output_filename)
                        if os.path.exists(output_path):
                            temp_results_file = os.path.join(temp_dir, f'rcan_hyperIQA_{unique_id}.json')
                            quality_score = run_hyperIQA(output_path, temp_results_file)
                            if quality_score is not None:
                                response_data['image_quality'] = quality_score
                            
                            # Clean up temporary hyperIQA file
                            if os.path.exists(temp_results_file):
                                os.remove(temp_results_file)
                        
                        # Clean up the temporary file
                        os.remove(result_file)
                        print(f"DEBUG: Cleaned up RCAN result file")
                    else:
                        print(f"DEBUG: RCAN JSON file not found")
                        # Fallback response
                        response_data['input_image'] = f"{settings.MEDIA_URL}uploads/{input_filename}"
                elif algorithm in carn_algorithms:
                    # Read CARN algorithm results from JSON file
                    print(f"DEBUG: Reading CARN results")
                    temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
                    result_file = os.path.join(temp_dir, 'carn_results.json')
                    
                    print(f"DEBUG: Looking for CARN result file: {result_file}")
                    
                    if os.path.exists(result_file):
                        with open(result_file, 'r') as f:
                            results = json.load(f)
                        
                        print(f"DEBUG: CARN JSON results: {results}")
                        
                        # Use the unique output filename generated earlier
                        response_data['processed_image'] = f"{settings.MEDIA_URL}output/{output_filename}"
                        response_data['input_image'] = f"{settings.MEDIA_URL}uploads/{input_filename}"
                        
                        # Add CARN results to response
                        if results.get('psnr') is not None:
                            response_data['psnr'] = float(results.get('psnr'))
                        if results.get('processing_time') is not None:
                            response_data['processing_time'] = float(results.get('processing_time'))
                        if results.get('scale') is not None:
                            response_data['scale_factor'] = results.get('scale')
                        if results.get('model') is not None:
                            response_data['model_variant'] = results.get('model')
                        
                        # Run hyperIQA on CARN processed image
                        output_path = os.path.join(output_dir, output_filename)
                        if os.path.exists(output_path):
                            temp_results_file = os.path.join(temp_dir, f'carn_hyperIQA_{unique_id}.json')
                            quality_score = run_hyperIQA(output_path, temp_results_file)
                            if quality_score is not None:
                                response_data['image_quality'] = quality_score
                            
                            # Clean up temporary hyperIQA file
                            if os.path.exists(temp_results_file):
                                os.remove(temp_results_file)
                        
                        # Clean up the temporary file
                        os.remove(result_file)
                        print(f"DEBUG: Cleaned up CARN result file")
                    else:
                        print(f"DEBUG: CARN JSON file not found")
                        # Fallback response
                        response_data['input_image'] = f"{settings.MEDIA_URL}uploads/{input_filename}"
                elif algorithm in vdsr_algorithms:
                    # Read VDSR algorithm results from JSON file
                    print(f"DEBUG: Reading VDSR results")
                    temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
                    result_file = os.path.join(temp_dir, 'vdsr_results.json')
                    
                    print(f"DEBUG: Looking for VDSR result file: {result_file}")
                    
                    if os.path.exists(result_file):
                        with open(result_file, 'r') as f:
                            results = json.load(f)
                        
                        print(f"DEBUG: VDSR JSON results: {results}")
                        
                        # Use the unique output filename generated earlier
                        response_data['processed_image'] = f"{settings.MEDIA_URL}output/{output_filename}"
                        response_data['input_image'] = f"{settings.MEDIA_URL}uploads/{input_filename}"
                        
                        # Add VDSR results to response
                        if results.get('psnr') is not None:
                            response_data['psnr'] = float(results.get('psnr'))
                        if results.get('processing_time') is not None:
                            response_data['processing_time'] = float(results.get('processing_time'))
                        if results.get('scale') is not None:
                            response_data['scale_factor'] = results.get('scale')
                        
                        # Run hyperIQA on VDSR processed image
                        output_path = os.path.join(output_dir, output_filename)
                        if os.path.exists(output_path):
                            temp_results_file = os.path.join(temp_dir, f'vdsr_hyperIQA_{unique_id}.json')
                            quality_score = run_hyperIQA(output_path, temp_results_file)
                            if quality_score is not None:
                                response_data['image_quality'] = quality_score
                            
                            # Clean up temporary hyperIQA file
                            if os.path.exists(temp_results_file):
                                os.remove(temp_results_file)
                        
                        # Clean up the temporary file
                        os.remove(result_file)
                        print(f"DEBUG: Cleaned up VDSR result file")
                    else:
                        print(f"DEBUG: VDSR JSON file not found")
                        # Fallback response
                        response_data['input_image'] = f"{settings.MEDIA_URL}uploads/{input_filename}"
                elif algorithm in direct_algorithms:
                    # Read Direct algorithm results from JSON file
                    print(f"DEBUG: Reading Direct results")
                    temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
                    result_file = os.path.join(temp_dir, 'direct_results.json')
                    
                    print(f"DEBUG: Looking for Direct result file: {result_file}")
                    
                    if os.path.exists(result_file):
                        with open(result_file, 'r') as f:
                            results = json.load(f)
                        
                        print(f"DEBUG: Direct JSON results: {results}")
                        
                        # Use the unique output filename generated earlier
                        response_data['processed_image'] = f"{settings.MEDIA_URL}output/{output_filename}"
                        response_data['input_image'] = f"{settings.MEDIA_URL}uploads/{input_filename}"
                        
                        # Add Direct results to response
                        if results.get('psnr') is not None:
                            response_data['psnr'] = float(results.get('psnr'))
                        if results.get('processing_time') is not None:
                            response_data['processing_time'] = float(results.get('processing_time'))
                        if results.get('scale') is not None:
                            response_data['scale_factor'] = results.get('scale')
                        
                        # Run hyperIQA on Direct processed image
                        output_path = os.path.join(output_dir, output_filename)
                        if os.path.exists(output_path):
                            temp_results_file = os.path.join(temp_dir, f'direct_hyperIQA_{unique_id}.json')
                            quality_score = run_hyperIQA(output_path, temp_results_file)
                            if quality_score is not None:
                                response_data['image_quality'] = quality_score
                            
                            # Clean up temporary hyperIQA file
                            if os.path.exists(temp_results_file):
                                os.remove(temp_results_file)
                        
                        # Clean up the temporary file
                        os.remove(result_file)
                        print(f"DEBUG: Cleaned up Direct result file")
                    else:
                        print(f"DEBUG: Direct JSON file not found")
                        # Fallback response
                        response_data['input_image'] = f"{settings.MEDIA_URL}uploads/{input_filename}"
                elif algorithm in espcn_algorithms:
                    # Read ESPCN algorithm results from JSON file
                    print(f"DEBUG: Reading ESPCN results for unique_id: {unique_id}")  # Debug log
                    temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
                    result_file = os.path.join(temp_dir, f'espcn_results_{unique_id}.json')
                    
                    print(f"DEBUG: Looking for result file: {result_file}")  # Debug log
                    print(f"DEBUG: File exists: {os.path.exists(result_file)}")  # Debug log
                    
                    if os.path.exists(result_file):
                        print(f"DEBUG: Reading JSON file")  # Debug log
                        with open(result_file, 'r') as f:
                            results = json.load(f)
                        
                        print(f"DEBUG: JSON results: {results}")  # Debug log
                        
                        # Extract output filename
                        output_filename = results.get('output_filename', '')
                        if output_filename:
                            response_data['processed_image'] = f"{settings.MEDIA_URL}output/{output_filename}"
                        
                        response_data['input_image'] = f"{settings.MEDIA_URL}uploads/{input_filename}"
                        
                        # Add ESPCN results to response
                        if results.get('psnr') is not None:
                            response_data['psnr'] = float(results.get('psnr'))
                            print(f"DEBUG: Set PSNR to: {response_data['psnr']}")  # Debug log
                        
                        # Add processing_time to response for ESPCN
                        if results.get('processing_time') is not None:
                            response_data['processing_time'] = float(results.get('processing_time'))
                            print(f"DEBUG: Set processing_time to: {response_data['processing_time']}")  # Debug log
                        
                        # Add lr_filename to response for ESPCN
                        if results.get('lr_filename'):
                            response_data['lr_filename'] = results.get('lr_filename')
                            print(f"DEBUG: Set lr_filename to: {response_data['lr_filename']}")  # Debug log
                        
                        # Add bicubic_filename to response for ESPCN
                        if results.get('bicubic_filename'):
                            response_data['bicubic_filename'] = results.get('bicubic_filename')
                            print(f"DEBUG: Set bicubic_filename to: {response_data['bicubic_filename']}")  # Debug log
                        
                        # Add result plot to response for ESPCN (comparison plot)
                        if results.get('result_plot_filename'):
                            response_data['result_plot_filename'] = results.get('result_plot_filename')
                            print(f"DEBUG: Set result_plot_filename to: {response_data['result_plot_filename']}")  # Debug log
                        
                        # Generate additional images for compatibility (bicubic, result)
                        base_name, ext = os.path.splitext(os.path.basename(input_path))
                        bicubic_file = f"{base_name}_bicubic_x{scale}{ext}"
                        result_file_name = f"{base_name}_result{ext}"
                        
                        response_data.update({
                            'bicubic_image': f"{settings.MEDIA_URL}output/{bicubic_file}",
                            'espcn_image': f"{settings.MEDIA_URL}output/{output_filename}",
                            'result_image': f"{settings.MEDIA_URL}output/{result_file_name}",
                        })
                        
                        # Run hyperIQA on ESPCN processed image
                        espcn_output_path = os.path.join(output_dir, output_filename)
                        
                        # Create temporary JSON file for hyperIQA results
                        temp_results_file = os.path.join(temp_dir, f'espcn_hyperIQA_{unique_id}.json')
                        
                        quality_score = run_hyperIQA(espcn_output_path, temp_results_file)
                        if quality_score is not None:
                            response_data['image_quality'] = quality_score
                        
                        # Clean up temporary files
                        if os.path.exists(temp_results_file):
                            os.remove(temp_results_file)
                        os.remove(result_file)
                        print(f"DEBUG: Cleaned up result file")  # Debug log
                    else:
                        print(f"DEBUG: JSON file not found, using fallback method")  # Debug log
                        # Fallback to old method if JSON file doesn't exist
                        base_name, ext = os.path.splitext(os.path.basename(input_path))
                        bicubic_file = f"{base_name}_bicubic_x{scale}{ext}"
                        espcn_file = f"{base_name}_espcn_x{scale}{ext}"
                        result_file_name = f"{base_name}_result{ext}"
                        response_data.update({
                            'bicubic_image': f"{settings.MEDIA_URL}output/{bicubic_file}",
                            'espcn_image': f"{settings.MEDIA_URL}output/{espcn_file}",
                            'result_image': f"{settings.MEDIA_URL}output/{result_file_name}",
                        })
                        response_data['input_image'] = f"{settings.MEDIA_URL}uploads/{input_filename}"
                        
                        # Read PSNR from .env file as fallback
                        env_path = os.path.join(settings.BASE_DIR, '.env')
                        env_vars = dotenv_values(env_path)
                        psnr_value = env_vars.get('PSNR_RESULT')
                        if psnr_value and psnr_value.strip():
                            response_data['psnr'] = float(psnr_value)
                            set_key(env_path, 'PSNR_RESULT', '')

                print(f"DEBUG: Final response_data: {response_data}")  # Debug log
                return Response(response_data, status=status.HTTP_200_OK)

            except subprocess.CalledProcessError as e:
                return Response({'error': f'Error during image processing: {str(e)}'}, status=500)
            finally:
                # Clean up temporary result file for fariba and espcn algorithms
                if result_file_path and os.path.exists(result_file_path):
                    try:
                        os.remove(result_file_path)
                    except OSError:
                        pass  # Ignore cleanup errors
        else:
            return Response(serializer.errors, status=400)


class ComparisonView(APIView):
    """View for running comparison between multiple algorithms using file system only"""
    
    def post(self, request):
        serializer = ComparisonSerializer(data=request.data)
        if serializer.is_valid():
            uploaded_file = serializer.validated_data['image']
            algorithms = serializer.validated_data['algorithms']
            scale = serializer.validated_data.get('scale', '2')
            parameters = serializer.validated_data.get('parameters', {})
            
            # Save uploaded file temporarily
            unique_id = str(uuid.uuid4())[:8]
            base_name, ext = os.path.splitext(uploaded_file.name)
            input_filename = f"{base_name}_comparison_{unique_id}{ext}"
            
            input_path = default_storage.save(f'comparison_inputs/{input_filename}', uploaded_file)
            full_input_path = os.path.join(settings.MEDIA_ROOT, input_path)
            
            results_data = {}
            
            # Process each algorithm
            for algorithm in algorithms:
                try:
                    # Get algorithm-specific parameters
                    algo_params = parameters.get(algorithm, {})
                    algo_scale = algo_params.get('scale', scale)
                    
                    result = self.process_single_algorithm(
                        full_input_path, algorithm, algo_scale, unique_id, algo_params
                    )
                    if result:
                        results_data[algorithm] = result
                except Exception as e:
                    print(f"Error processing {algorithm}: {e}")
                    continue
            
            # Clean up input file
            try:
                os.remove(full_input_path)
            except OSError:
                pass
            
            return Response({
                'results': results_data,
                'total_algorithms': len(algorithms),
                'successful_results': len(results_data),
            }, status=200)
        
        return Response(serializer.errors, status=400)
    
    def process_single_algorithm(self, input_path, algorithm, scale, unique_id, params):
        """Process a single algorithm and return results as dictionary"""
        start_time = time.time()
        
        output_dir = os.path.join(settings.MEDIA_ROOT, 'comparison_outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # Algorithm lists
        rdn_algorithms = ['rdn']
        rcan_algorithms = ['rcan'] 
        carn_algorithms = ['carn', 'carn_m']
        espcn_algorithms = ['espcn']
        
        try:
            python_executable = sys.executable
            env = os.environ.copy()
            
            # Build command based on algorithm type
            if algorithm in rdn_algorithms:
                temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
                os.makedirs(temp_dir, exist_ok=True)
                command = [
                    python_executable,
                    'algorithms/RDN/infer.py',
                    '--input', input_path,
                    '--output', os.path.join(output_dir, f'result_{algorithm}_{unique_id}.png'),
                    '--scale', scale,
                    '--temp_results', temp_dir
                ]
            elif algorithm in rcan_algorithms:
                temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
                os.makedirs(temp_dir, exist_ok=True)
                command = [
                    python_executable,
                    'algorithms/RCAN/infer.py',
                    '--input', input_path,
                    '--output', os.path.join(output_dir, f'result_{algorithm}_{unique_id}.png'),
                    '--scale', scale,
                    '--temp_results', temp_dir
                ]
            elif algorithm in carn_algorithms:
                model_variant = 'carn' if algorithm == 'carn' else 'carn_m'
                command = [
                    python_executable,
                    'algorithms/CARN/infer.py',
                    '--input', input_path,
                    '--output', os.path.join(output_dir, f'result_{algorithm}_{unique_id}.png'),
                    '--scale', scale,
                    '--model', model_variant
                ]
            elif algorithm in espcn_algorithms:
                command = [
                    python_executable,
                    'algorithms/ESPCN/infer.py',
                    '-w', 'algorithms/ESPCN/assets/models/best_model.pth',
                    '-i', input_path,
                    '-o', output_dir,
                    '--unique-id', f'{algorithm}_{unique_id}'
                ]
            else:
                return None
            
            # Run algorithm
            subprocess.run(command, check=True, cwd=settings.BASE_DIR, env=env)
            
            # Calculate processing time in seconds
            processing_time = time.time() - start_time
            
            # Read results from JSON files
            temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
            result_data = {
                'algorithm_name': algorithm,
                'processing_time': processing_time,
                'scale_factor': scale
            }
            
            if algorithm in ['rdn', 'rcan', 'carn']:
                result_file = os.path.join(temp_dir, f'{algorithm}_results.json')
            elif algorithm == 'espcn':
                result_file = os.path.join(temp_dir, f'espcn_results_{algorithm}_{unique_id}.json')
            else:
                result_file = None
            
            if result_file and os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    json_results = json.load(f)
                
                # Extract relevant data
                if json_results.get('output_path'):
                    output_filename = os.path.basename(json_results['output_path'])
                    result_data['processed_image'] = f"{settings.MEDIA_URL}comparison_outputs/{output_filename}"
                
                if json_results.get('psnr') is not None:
                    result_data['psnr'] = float(json_results['psnr'])
                
                if json_results.get('processing_time') is not None:
                    result_data['processing_time'] = float(json_results['processing_time'])
                
                # Run hyperIQA if output exists
                if json_results.get('output_path') and os.path.exists(json_results['output_path']):
                    temp_results_file = os.path.join(temp_dir, f'comparison_{algorithm}_{unique_id}_hyperIQA.json')
                    quality_score = run_hyperIQA(json_results['output_path'], temp_results_file)
                    if quality_score is not None:
                        result_data['image_quality'] = quality_score
                    
                    # Cleanup hyperIQA temp file
                    if os.path.exists(temp_results_file):
                        os.remove(temp_results_file)
                
                # Cleanup algorithm result file
                os.remove(result_file)
            
            return result_data
            
        except Exception as e:
            print(f"Error processing {algorithm}: {e}")
            return None


@api_view(['GET'])
def health_check(request):
    """Simple health check endpoint"""
    return Response({'status': 'ok', 'message': 'API is running with file-based approach only'})
