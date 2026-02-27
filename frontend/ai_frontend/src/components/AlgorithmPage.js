import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
    Container,
    Grid,
    Typography,
    Button,
    Box,
    CircularProgress,
    Alert,
    Chip,
    IconButton,
    Paper,
    Slider
} from '@mui/material';
import { CloudUpload, PhotoLibrary, ArrowBack, Download } from '@mui/icons-material';
import axios from 'axios';
import ALGORITHMS from '../constants/algorithms';
import ProcessingLoader from './ProcessingLoader';

function AlgorithmPage() {
    const { algorithmId } = useParams();
    const navigate = useNavigate();

    // Find algorithm
    const algorithm = ALGORITHMS.find(alg => alg.id === algorithmId);

    // State management
    const [selectedImage, setSelectedImage] = useState(null);
    const [resultImage, setResultImage] = useState(null);
    const [lrImage, setLrImage] = useState(null);
    const [bicubicImage, setBicubicImage] = useState(null);
    const [resultPlotImage, setResultPlotImage] = useState(null);
    const [croppedImage, setCroppedImage] = useState(null);
    const [processing, setProcessing] = useState(false);
    const [previewImage, setPreviewImage] = useState(null);
    const [psnrValue, setPsnrValue] = useState(null);
    const [qualityValue, setQualityValue] = useState(null);
    const [processingTime, setProcessingTime] = useState(null);
    const [faribaQuality, setFaribaQuality] = useState(3);
    const [ninasrScale, setNinasrScale] = useState(3);
    const [ninasrVariant, setNinasrVariant] = useState('b1');
    const [edsrScale, setEdsrScale] = useState(2);
    const [rdnScale, setRdnScale] = useState(2);
    const [rcanScale, setRcanScale] = useState(2);
    const [carnScale, setCarnScale] = useState(2);
    const [carnMScale, setCarnMScale] = useState(2);
    const [vdsrScale, setVdsrScale] = useState(2);
    const [directScale, setDirectScale] = useState(2);
    const [error, setError] = useState('');
    const [predictedClass, setPredictedClass] = useState(null);
    const [bitrate, setBitrate] = useState(null);
    const [decodeTime, setDecodeTime] = useState(null);
    const [parametersChanged, setParametersChanged] = useState(false);

    // Ref for scrolling to results
    const resultsRef = useRef(null);

    useEffect(() => {
        if (!algorithm) {
            navigate('/');
        }
    }, [algorithm, navigate]);

    if (!algorithm) {
        return null;
    }

    const handleImageUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
            setSelectedImage(file);
            const reader = new FileReader();
            reader.onload = (e) => {
                setPreviewImage(e.target.result);
            };
            reader.readAsDataURL(file);
            resetResults();
        }
    };

    const markParametersChanged = () => {
        setParametersChanged(true);
    };

    const resetResults = () => {
        setResultImage(null);
        setLrImage(null);
        setBicubicImage(null);
        setResultPlotImage(null);
        setCroppedImage(null);
        setError('');
        setPsnrValue(null);
        setQualityValue(null);
        setProcessingTime(null);
        setPredictedClass(null);
        setBitrate(null);
        setDecodeTime(null);
        setParametersChanged(false);
    };

    const processImage = async () => {
        if (!selectedImage) {
            setError('Please select an image first');
            return;
        }

        setProcessing(true);
        setError('');
        setParametersChanged(false);

        try {
            const formData = new FormData();
            formData.append('image', selectedImage);
            formData.append('algorithm', algorithmId);

            // Set scale or quality based on algorithm
            if (algorithmId === 'fariba') {
                formData.append('scale', faribaQuality.toString());
            } else if (algorithmId === 'ninasr') {
                formData.append('scale', ninasrScale.toString());
                formData.append('variant', ninasrVariant);
            } else if (algorithmId === 'edsr' || algorithmId === 'edsr_baseline') {
                formData.append('scale', edsrScale.toString());
            } else if (algorithmId === 'rdn') {
                formData.append('scale', rdnScale.toString());
            } else if (algorithmId === 'rcan') {
                formData.append('scale', rcanScale.toString());
            } else if (algorithmId === 'carn') {
                formData.append('scale', carnScale.toString());
            } else if (algorithmId === 'carn_m') {
                formData.append('scale', carnMScale.toString());
            } else if (algorithmId === 'vdsr') {
                formData.append('scale', vdsrScale.toString());
            } else if (algorithmId === 'direct') {
                formData.append('scale', directScale.toString());
            } else {
                formData.append('scale', '3'); // Default scale for other super resolution algorithms
            }

            const response = await axios.post('http://localhost:8000/api/upload/', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            const data = response.data;
            console.log('API Response data:', data); // Debug log

            // Set result image URL
            if (data.processed_image) {
                const imageUrl = `http://localhost:8000${data.processed_image}`;
                setResultImage(imageUrl);
            }

            // Set LR image URL for ESPCN algorithm
            if (data.lr_filename && algorithmId === 'espcn') {
                const lrImageUrl = `http://localhost:8000/media/uploads/${data.lr_filename}`;
                console.log('Setting LR image URL:', lrImageUrl); // Debug log
                setLrImage(lrImageUrl);
            }

            // Set Bicubic image URL for ESPCN algorithm
            if (data.bicubic_filename && algorithmId === 'espcn') {
                const bicubicImageUrl = `http://localhost:8000/media/output/${data.bicubic_filename}`;
                console.log('Setting Bicubic image URL:', bicubicImageUrl); // Debug log
                setBicubicImage(bicubicImageUrl);
            }

            // Set Result Plot image URL for ESPCN algorithm (comparison plot)
            if (data.result_plot_filename && algorithmId === 'espcn') {
                const resultPlotImageUrl = `http://localhost:8000/media/output/${data.result_plot_filename}`;
                console.log('Setting Result Plot image URL:', resultPlotImageUrl); // Debug log
                setResultPlotImage(resultPlotImageUrl);
            }

            // Set cropped image URL for Fariba algorithm
            if (data.cropped_filename && algorithmId === 'fariba') {
                const croppedImageUrl = `http://localhost:8000/media/uploads/${data.cropped_filename}`;
                console.log('Setting cropped image URL:', croppedImageUrl); // Debug log
                setCroppedImage(croppedImageUrl);
            }

            // Set metrics from response
            if (data.psnr !== undefined) setPsnrValue(data.psnr);
            if (data.image_quality !== undefined) setQualityValue(data.image_quality);
            if (data.bitrate !== undefined) setBitrate(data.bitrate);
            if (data.predicted_class) setPredictedClass(data.predicted_class);
            if (data.predicted_index !== undefined) console.log('Predicted Index:', data.predicted_index);
            if (data.decode_time !== undefined) setDecodeTime(data.decode_time);

            // Set processing time from response (in seconds from backend)
            if (data.processing_time !== undefined) {
                setProcessingTime(data.processing_time);
            } else {
                // Fallback to mock processing time for algorithms that don't provide it
                setProcessingTime(algorithm.processingTime || 2.5);
            }

        } catch (error) {
            console.error('Error processing image:', error);
            if (error.response?.data?.error) {
                setError(`Error: ${error.response.data.error}`);
            } else {
                setError('Error processing image. Please try again.');
            }
        } finally {
            setProcessing(false);
            // Scroll to results after a short delay
            setTimeout(() => {
                resultsRef.current?.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }, 500);
        }
    }; const downloadResult = async () => {
        if (resultImage) {
            try {
                // Fetch the image as blob for download
                const response = await fetch(resultImage);
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);

                const link = document.createElement('a');
                link.href = url;
                link.download = `${algorithmId}_result.png`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);

                // Clean up the object URL
                URL.revokeObjectURL(url);
            } catch (error) {
                console.error('Error downloading image:', error);
                // Fallback: try direct link
                const link = document.createElement('a');
                link.href = resultImage;
                link.download = `${algorithmId}_result.png`;
                link.target = '_blank';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
        }
    };

    const downloadLrImage = async () => {
        if (lrImage) {
            try {
                // Fetch the image as blob for download
                const response = await fetch(lrImage);
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);

                const link = document.createElement('a');
                link.href = url;
                link.download = `${algorithmId}_lr_image.png`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);

                // Clean up the object URL
                URL.revokeObjectURL(url);
            } catch (error) {
                console.error('Error downloading LR image:', error);
                // Fallback: try direct link
                const link = document.createElement('a');
                link.href = lrImage;
                link.download = `${algorithmId}_lr_image.png`;
                link.target = '_blank';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
        }
    };

    const downloadBicubicImage = async () => {
        if (bicubicImage) {
            try {
                // Fetch the image as blob for download
                const response = await fetch(bicubicImage);
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);

                const link = document.createElement('a');
                link.href = url;
                link.download = `${algorithmId}_bicubic_image.png`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);

                // Clean up the object URL
                URL.revokeObjectURL(url);
            } catch (error) {
                console.error('Error downloading Bicubic image:', error);
                // Fallback: try direct link
                const link = document.createElement('a');
                link.href = bicubicImage;
                link.download = `${algorithmId}_bicubic_image.png`;
                link.target = '_blank';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
        }
    };

    const downloadResultPlotImage = async () => {
        if (resultPlotImage) {
            try {
                // Fetch the image as blob for download
                const response = await fetch(resultPlotImage);
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);

                const link = document.createElement('a');
                link.href = url;
                link.download = `${algorithmId}_comparison_plot.png`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);

                // Clean up the object URL
                URL.revokeObjectURL(url);
            } catch (error) {
                console.error('Error downloading Result Plot image:', error);
                // Fallback: try direct link
                const link = document.createElement('a');
                link.href = resultPlotImage;
                link.download = `${algorithmId}_comparison_plot.png`;
                link.target = '_blank';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
        }
    };

    const downloadCroppedImage = async () => {
        if (croppedImage) {
            try {
                // Fetch the image as blob for download
                const response = await fetch(croppedImage);
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);

                const link = document.createElement('a');
                link.href = url;
                link.download = `${algorithmId}_cropped_image.png`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);

                // Clean up the object URL
                URL.revokeObjectURL(url);
            } catch (error) {
                console.error('Error downloading cropped image:', error);
                // Fallback: try direct link
                const link = document.createElement('a');
                link.href = croppedImage;
                link.download = `${algorithmId}_cropped_image.png`;
                link.target = '_blank';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
        }
    };

    return (
        <>
            <Container maxWidth="xl" sx={{ py: 4 }}>
                {/* Header */}
                <Box sx={{ mb: 4, display: 'flex', alignItems: 'center', gap: 2 }}>
                    <IconButton onClick={() => navigate('/')} color="primary">
                        <ArrowBack />
                    </IconButton>
                    <Typography variant="h4" component="h1">
                        {algorithm.name}
                    </Typography>
                    <Chip
                        label={algorithm.category}
                        color="primary"
                        variant="outlined"
                    />
                </Box>

                <Grid container spacing={4}>
                    {/* Left Side - Algorithm Info */}
                    <Grid item xs={12} md={4}>
                        <Paper sx={{ p: 3, mb: 3 }}>
                            <Typography variant="h6" gutterBottom>
                                Algorithm Information
                            </Typography>
                            <Typography variant="body2" color="text.secondary" paragraph>
                                {algorithm.description}
                            </Typography>

                            {algorithm.englishDescription && (
                                <>
                                    <Typography variant="subtitle2" sx={{ mt: 2, mb: 1, fontWeight: 'bold' }}>
                                        Technical Description:
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary" paragraph sx={{ fontStyle: 'italic' }}>
                                        {algorithm.englishDescription}
                                    </Typography>
                                </>
                            )}

                            <Box sx={{ mb: 2 }}>
                                <Typography variant="body2" sx={{ mb: 1 }}>
                                    Speed: {algorithm.speed}/5
                                </Typography>
                                <Box sx={{ display: 'flex', gap: 1 }}>
                                    {[1, 2, 3, 4, 5].map((i) => (
                                        <Box
                                            key={i}
                                            sx={{
                                                width: 20,
                                                height: 4,
                                                bgcolor: i <= algorithm.speed ? 'primary.main' : 'grey.300',
                                                borderRadius: 1
                                            }}
                                        />
                                    ))}
                                </Box>
                            </Box>

                            <Box sx={{ mb: 2 }}>
                                <Typography variant="body2" sx={{ mb: 1 }}>
                                    Quality: {algorithm.quality}/5
                                </Typography>
                                <Box sx={{ display: 'flex', gap: 1 }}>
                                    {[1, 2, 3, 4, 5].map((i) => (
                                        <Box
                                            key={i}
                                            sx={{
                                                width: 20,
                                                height: 4,
                                                bgcolor: i <= algorithm.quality ? 'secondary.main' : 'grey.300',
                                                borderRadius: 1
                                            }}
                                        />
                                    ))}
                                </Box>
                            </Box>

                            <Typography variant="body2" sx={{ mb: 1 }}>
                                Features:
                            </Typography>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                                {algorithm.features.map((feature, index) => (
                                    <Chip
                                        key={index}
                                        label={feature}
                                        size="small"
                                        variant="outlined"
                                    />
                                ))}
                            </Box>
                        </Paper>

                        {/* Input Controls Section */}
                        <Paper sx={{ p: 3 }}>
                            <Typography variant="h6" gutterBottom>
                                Input Settings
                            </Typography>

                            {/* Fariba Quality Control */}
                            {algorithmId === 'fariba' && (
                                <>
                                    <Typography variant="body2" color="text.secondary" gutterBottom>
                                        Compression Quality (1-5)
                                    </Typography>
                                    <Slider
                                        value={faribaQuality}
                                        onChange={(e, value) => {
                                            setFaribaQuality(value);
                                            markParametersChanged();
                                        }}
                                        min={1}
                                        max={5}
                                        step={1}
                                        marks={[
                                            { value: 1, label: '1' },
                                            { value: 2, label: '2' },
                                            { value: 3, label: '3' },
                                            { value: 4, label: '4' },
                                            { value: 5, label: '5' }
                                        ]}
                                        valueLabelDisplay="auto"
                                        sx={{ mb: 2 }}
                                    />
                                    <Typography variant="caption" color="text.secondary">
                                        1: High Compression (Small File) | 5: High Quality (Large File)
                                    </Typography>
                                </>
                            )}

                            {/* EDSR Parameter Controls */}
                            {(algorithmId === 'edsr' || algorithmId === 'edsr_baseline') && (
                                <>
                                    <Typography variant="body2" color="text.secondary" gutterBottom>
                                        Scale Factor
                                    </Typography>
                                    <Slider
                                        value={edsrScale}
                                        onChange={(e, value) => {
                                            setEdsrScale(value);
                                            markParametersChanged();
                                        }}
                                        min={2}
                                        max={4}
                                        step={1}
                                        marks={[
                                            { value: 2, label: '2x' },
                                            { value: 3, label: '3x' },
                                            { value: 4, label: '4x' }
                                        ]}
                                        valueLabelDisplay="auto"
                                        sx={{ mb: 2 }}
                                    />
                                    <Typography variant="caption" color="text.secondary">
                                        {algorithmId === 'edsr' ?
                                            '2x: ~38.19 dB | 3x: ~34.68 dB | 4x: ~32.48 dB (40.7M params)' :
                                            '2x: ~37.98 dB | 3x: ~34.37 dB | 4x: ~32.09 dB (1.37M params)'
                                        }
                                    </Typography>
                                </>
                            )}

                            {/* RDN Parameter Controls */}
                            {algorithmId === 'rdn' && (
                                <>
                                    <Typography variant="body2" color="text.secondary" gutterBottom>
                                        Scale Factor
                                    </Typography>
                                    <Slider
                                        value={rdnScale}
                                        onChange={(e, value) => {
                                            setRdnScale(value);
                                            markParametersChanged();
                                        }}
                                        min={2}
                                        max={4}
                                        step={1}
                                        marks={[
                                            { value: 2, label: '2x' },
                                            { value: 3, label: '3x' },
                                            { value: 4, label: '4x' }
                                        ]}
                                        valueLabelDisplay="auto"
                                        sx={{ mb: 2 }}
                                    />
                                    <Typography variant="caption" color="text.secondary">
                                        2x: ~38.24 dB | 3x: ~34.71 dB | 4x: ~32.47 dB (22.3M params)
                                    </Typography>
                                </>
                            )}

                            {/* RCAN Parameter Controls */}
                            {algorithmId === 'rcan' && (
                                <>
                                    <Typography variant="body2" color="text.secondary" gutterBottom>
                                        Scale Factor
                                    </Typography>
                                    <Slider
                                        value={rcanScale}
                                        onChange={(e, value) => {
                                            setRcanScale(value);
                                            markParametersChanged();
                                        }}
                                        min={2}
                                        max={4}
                                        step={1}
                                        marks={[
                                            { value: 2, label: '2x' },
                                            { value: 3, label: '3x' },
                                            { value: 4, label: '4x' }
                                        ]}
                                        valueLabelDisplay="auto"
                                        sx={{ mb: 2 }}
                                    />
                                    <Typography variant="caption" color="text.secondary">
                                        2x: ~38.27 dB | 3x: ~34.74 dB | 4x: ~32.63 dB (15.6M params)
                                    </Typography>
                                </>
                            )}

                            {/* CARN Parameter Controls */}
                            {algorithmId === 'carn' && (
                                <>
                                    <Typography variant="body2" color="text.secondary" gutterBottom>
                                        Scale Factor
                                    </Typography>
                                    <Slider
                                        value={carnScale}
                                        onChange={(e, value) => {
                                            setCarnScale(value);
                                            markParametersChanged();
                                        }}
                                        min={2}
                                        max={4}
                                        step={1}
                                        marks={[
                                            { value: 2, label: '2x' },
                                            { value: 3, label: '3x' },
                                            { value: 4, label: '4x' }
                                        ]}
                                        valueLabelDisplay="auto"
                                        sx={{ mb: 2 }}
                                    />
                                    <Typography variant="caption" color="text.secondary">
                                        2x: ~37.76 dB | 3x: ~34.29 dB | 4x: ~32.13 dB (1.59M params)
                                    </Typography>
                                </>
                            )}

                            {/* CARN-M Parameter Controls */}
                            {algorithmId === 'carn_m' && (
                                <>
                                    <Typography variant="body2" color="text.secondary" gutterBottom>
                                        Scale Factor
                                    </Typography>
                                    <Slider
                                        value={carnMScale}
                                        onChange={(e, value) => {
                                            setCarnMScale(value);
                                            markParametersChanged();
                                        }}
                                        min={2}
                                        max={4}
                                        step={1}
                                        marks={[
                                            { value: 2, label: '2x' },
                                            { value: 3, label: '3x' },
                                            { value: 4, label: '4x' }
                                        ]}
                                        valueLabelDisplay="auto"
                                        sx={{ mb: 2 }}
                                    />
                                    <Typography variant="caption" color="text.secondary">
                                        2x: ~37.53 dB | 3x: ~34.26 dB | 4x: ~32.09 dB (412K params)
                                    </Typography>
                                </>
                            )}

                            {/* VDSR Parameter Controls */}
                            {algorithmId === 'vdsr' && (
                                <>
                                    <Typography variant="body2" color="text.secondary" gutterBottom>
                                        Scale Factor
                                    </Typography>
                                    <Slider
                                        value={vdsrScale}
                                        onChange={(e, value) => {
                                            setVdsrScale(value);
                                            markParametersChanged();
                                        }}
                                        min={2}
                                        max={8}
                                        step={1}
                                        marks={[
                                            { value: 2, label: '2x' },
                                            { value: 3, label: '3x' },
                                            { value: 4, label: '4x' },
                                            { value: 8, label: '8x' }
                                        ]}
                                        valueLabelDisplay="auto"
                                        sx={{ mb: 2 }}
                                    />
                                    <Typography variant="caption" color="text.secondary">
                                        2x: ~37.53 dB | 3x: ~33.66 dB | 4x: ~31.35 dB | 8x: ~25.93 dB
                                    </Typography>
                                </>
                            )}

                            {/* Direct Upscaling Parameter Controls */}
                            {algorithmId === 'direct' && (
                                <>
                                    <Typography variant="body2" color="text.secondary" gutterBottom>
                                        Scale Factor
                                    </Typography>
                                    <Slider
                                        value={directScale}
                                        onChange={(e, value) => {
                                            setDirectScale(value);
                                            markParametersChanged();
                                        }}
                                        min={2}
                                        max={8}
                                        step={1}
                                        marks={[
                                            { value: 2, label: '2x' },
                                            { value: 3, label: '3x' },
                                            { value: 4, label: '4x' },
                                            { value: 8, label: '8x' }
                                        ]}
                                        valueLabelDisplay="auto"
                                        sx={{ mb: 2 }}
                                    />
                                    <Typography variant="caption" color="text.secondary">
                                        Simple bicubic interpolation upscaling
                                    </Typography>
                                </>
                            )}

                            {/* NinaSR Parameter Controls */}
                            {algorithmId === 'ninasr' && (
                                <>
                                    {/* Scale Factor Control */}
                                    <Typography variant="body2" color="text.secondary" gutterBottom>
                                        Scale Factor
                                    </Typography>
                                    <Slider
                                        value={ninasrScale}
                                        onChange={(e, value) => {
                                            setNinasrScale(value);
                                            markParametersChanged();
                                        }}
                                        min={2}
                                        max={8}
                                        step={1}
                                        marks={[
                                            { value: 2, label: '2x' },
                                            { value: 3, label: '3x' },
                                            { value: 4, label: '4x' },
                                            { value: 8, label: '8x' }
                                        ]}
                                        valueLabelDisplay="auto"
                                        sx={{ mb: 2 }}
                                    />

                                    {/* Model Variant Control */}
                                    <Typography variant="body2" color="text.secondary" gutterBottom sx={{ mt: 2 }}>
                                        Model Variant
                                    </Typography>
                                    <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                                        {['b0', 'b1', 'b2'].map((variant) => (
                                            <Button
                                                key={variant}
                                                variant={ninasrVariant === variant ? 'contained' : 'outlined'}
                                                size="small"
                                                onClick={() => {
                                                    setNinasrVariant(variant);
                                                    markParametersChanged();
                                                }}
                                            >
                                                {variant.toUpperCase()}
                                                {variant === 'b0' && ' (0.1M)'}
                                                {variant === 'b1' && ' (1M)'}
                                                {variant === 'b2' && ' (10M)'}
                                            </Button>
                                        ))}
                                    </Box>
                                    <Typography variant="caption" color="text.secondary">
                                        B0: Fastest (0.1M params) | B1: Balanced (1M params) | B2: Best Quality (10M params)
                                    </Typography>
                                </>
                            )}

                            {/* Upload Section */}
                            <Box sx={{ mt: (algorithmId === 'fariba' || algorithmId === 'ninasr') ? 3 : 0 }}>
                                <input
                                    accept="image/*"
                                    style={{ display: 'none' }}
                                    id="image-upload"
                                    type="file"
                                    onChange={handleImageUpload}
                                />
                                <label htmlFor="image-upload">
                                    <Button
                                        variant="outlined"
                                        component="span"
                                        startIcon={<PhotoLibrary />}
                                        fullWidth
                                        sx={{ mb: 2 }}
                                    >
                                        Select Image
                                    </Button>
                                </label>

                                <Button
                                    variant="contained"
                                    onClick={processImage}
                                    disabled={!selectedImage || processing}
                                    startIcon={processing ? <CircularProgress size={20} /> : <CloudUpload />}
                                    fullWidth
                                >
                                    {processing ? 'Processing...' : 'Process Image'}
                                </Button>

                                {parametersChanged && resultImage && (
                                    <Alert severity="warning" sx={{ mt: 2 }}>
                                        Parameters have been changed. Click "Process Image" again to see updated results.
                                    </Alert>
                                )}
                            </Box>
                        </Paper>
                    </Grid>

                    {/* Right Side - Image Processing */}
                    <Grid item xs={12} md={8}>
                        <Paper sx={{ p: 3 }} ref={resultsRef}>
                            <Typography variant="h6" gutterBottom>
                                Processing Results
                            </Typography>

                            {error && (
                                <Alert severity="error" sx={{ mb: 3 }}>
                                    {error}
                                </Alert>
                            )}

                            {/* Image Display */}
                            {!processing && (
                                <Grid container spacing={2}>
                                    {previewImage && (
                                        <Grid item xs={12} md={6}>
                                            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, color: 'primary.main', mb: 2 }}>
                                                Original Image
                                            </Typography>
                                            <Box
                                                component="img"
                                                src={previewImage}
                                                alt="Original"
                                                sx={{
                                                    width: '100%',
                                                    height: 'auto',
                                                    maxHeight: 400,
                                                    objectFit: 'contain',
                                                    borderRadius: 1
                                                }}
                                            />
                                        </Grid>
                                    )}

                                    {resultImage && (
                                        <Grid item xs={12} md={6}>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                                                <Typography variant="h6" sx={{ fontWeight: 600, color: 'secondary.main' }}>
                                                    Processed Image
                                                </Typography>
                                                <IconButton onClick={downloadResult} color="primary" sx={{
                                                    backgroundColor: 'primary.main',
                                                    color: 'white',
                                                    '&:hover': {
                                                        backgroundColor: 'primary.dark'
                                                    }
                                                }}>
                                                    <Download />
                                                </IconButton>
                                            </Box>
                                            <Box sx={{ position: 'relative' }}>
                                                <Box
                                                    component="img"
                                                    src={resultImage}
                                                    alt="Result"
                                                    sx={{
                                                        width: '100%',
                                                        height: 'auto',
                                                        maxHeight: 400,
                                                        objectFit: 'contain',
                                                        borderRadius: 1,
                                                        opacity: parametersChanged ? 0.6 : 1,
                                                        filter: parametersChanged ? 'grayscale(30%)' : 'none',
                                                        transition: 'opacity 0.3s ease, filter 0.3s ease'
                                                    }}
                                                />
                                                {parametersChanged && (
                                                    <Box
                                                        sx={{
                                                            position: 'absolute',
                                                            top: 10,
                                                            right: 10,
                                                            backgroundColor: 'rgba(255, 152, 0, 0.9)',
                                                            color: 'white',
                                                            padding: '4px 8px',
                                                            borderRadius: 1,
                                                            fontSize: '0.75rem',
                                                            fontWeight: 'bold'
                                                        }}
                                                    >
                                                        Parameters Changed
                                                    </Box>
                                                )}
                                            </Box>
                                        </Grid>
                                    )}
                                </Grid>
                            )}

                            {/* Low Resolution Image for ESPCN */}
                            {!processing && (
                                <>
                                    {console.log('lrImage state:', lrImage, 'algorithmId:', algorithmId)}
                                    {lrImage && algorithmId === 'espcn' && (
                                        <Box sx={{ mt: 3 }}>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                                                <Typography variant="h6" sx={{ fontWeight: 600, color: 'secondary.main' }}>
                                                    Low Resolution Input (Generated)
                                                </Typography>
                                                <IconButton onClick={downloadLrImage} color="primary" sx={{
                                                    backgroundColor: 'primary.main',
                                                    color: 'white',
                                                    '&:hover': {
                                                        backgroundColor: 'primary.dark'
                                                    }
                                                }}>
                                                    <Download />
                                                </IconButton>
                                            </Box>
                                            <Box sx={{ position: 'relative' }}>
                                                <Box
                                                    component="img"
                                                    src={lrImage}
                                                    alt="Low Resolution"
                                                    onLoad={() => console.log('LR image loaded successfully')}
                                                    onError={(e) => console.error('LR image failed to load:', e)}
                                                    sx={{
                                                        width: '100%',
                                                        height: 'auto',
                                                        maxHeight: 300,
                                                        objectFit: 'contain',
                                                        borderRadius: 1,
                                                        opacity: parametersChanged ? 0.6 : 1,
                                                        filter: parametersChanged ? 'grayscale(30%)' : 'none',
                                                        transition: 'opacity 0.3s ease, filter 0.3s ease'
                                                    }}
                                                />
                                                {parametersChanged && (
                                                    <Box
                                                        sx={{
                                                            position: 'absolute',
                                                            top: 10,
                                                            right: 10,
                                                            backgroundColor: 'rgba(255, 152, 0, 0.9)',
                                                            color: 'white',
                                                            padding: '4px 8px',
                                                            borderRadius: 1,
                                                            fontSize: '0.75rem',
                                                            fontWeight: 'bold'
                                                        }}
                                                    >
                                                        Parameters Changed
                                                    </Box>
                                                )}
                                            </Box>
                                        </Box>
                                    )}

                                    {/* Bicubic Image for ESPCN */}
                                    {bicubicImage && algorithmId === 'espcn' && (
                                        <Box sx={{ mt: 3 }}>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                                                <Typography variant="h6" sx={{ fontWeight: 600, color: 'secondary.main' }}>
                                                    Bicubic Upscaled Reference
                                                </Typography>
                                                <IconButton onClick={downloadBicubicImage} color="primary" sx={{
                                                    backgroundColor: 'primary.main',
                                                    color: 'white',
                                                    '&:hover': {
                                                        backgroundColor: 'primary.dark'
                                                    }
                                                }}>
                                                    <Download />
                                                </IconButton>
                                            </Box>
                                            <Box sx={{ position: 'relative' }}>
                                                <Box
                                                    component="img"
                                                    src={bicubicImage}
                                                    alt="Bicubic Upscaled"
                                                    onLoad={() => console.log('Bicubic image loaded successfully')}
                                                    onError={(e) => console.error('Bicubic image failed to load:', e)}
                                                    sx={{
                                                        width: '100%',
                                                        height: 'auto',
                                                        maxHeight: 300,
                                                        objectFit: 'contain',
                                                        borderRadius: 1,
                                                        opacity: parametersChanged ? 0.6 : 1,
                                                        filter: parametersChanged ? 'grayscale(30%)' : 'none',
                                                        transition: 'opacity 0.3s ease, filter 0.3s ease'
                                                    }}
                                                />
                                                {parametersChanged && (
                                                    <Box
                                                        sx={{
                                                            position: 'absolute',
                                                            top: 10,
                                                            right: 10,
                                                            backgroundColor: 'rgba(255, 152, 0, 0.9)',
                                                            color: 'white',
                                                            padding: '4px 8px',
                                                            borderRadius: 1,
                                                            fontSize: '0.75rem',
                                                            fontWeight: 'bold'
                                                        }}
                                                    >
                                                        Parameters Changed
                                                    </Box>
                                                )}
                                            </Box>
                                        </Box>
                                    )}

                                    {/* Result Plot Image for ESPCN (Comparison Plot) */}
                                    {resultPlotImage && algorithmId === 'espcn' && (
                                        <Box sx={{ mt: 3 }}>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                                                <Typography variant="h6" sx={{ fontWeight: 600, color: 'secondary.main' }}>
                                                    Comparison Plot (HR vs Bicubic vs ESPCN)
                                                </Typography>
                                                <IconButton onClick={downloadResultPlotImage} color="primary" sx={{
                                                    backgroundColor: 'primary.main',
                                                    color: 'white',
                                                    '&:hover': {
                                                        backgroundColor: 'primary.dark'
                                                    }
                                                }}>
                                                    <Download />
                                                </IconButton>
                                            </Box>
                                            <Box sx={{ position: 'relative' }}>
                                                <Box
                                                    component="img"
                                                    src={resultPlotImage}
                                                    alt="ESPCN Comparison Plot"
                                                    onLoad={() => console.log('Result Plot image loaded successfully')}
                                                    onError={(e) => console.error('Result Plot image failed to load:', e)}
                                                    sx={{
                                                        width: '100%',
                                                        height: 'auto',
                                                        maxHeight: 400,
                                                        objectFit: 'contain',
                                                        borderRadius: 1,
                                                        opacity: parametersChanged ? 0.6 : 1,
                                                        filter: parametersChanged ? 'grayscale(30%)' : 'none',
                                                        transition: 'opacity 0.3s ease, filter 0.3s ease'
                                                    }}
                                                />
                                                {parametersChanged && (
                                                    <Box
                                                        sx={{
                                                            position: 'absolute',
                                                            top: 10,
                                                            right: 10,
                                                            backgroundColor: 'rgba(255, 152, 0, 0.9)',
                                                            color: 'white',
                                                            padding: '4px 8px',
                                                            borderRadius: 1,
                                                            fontSize: '0.75rem',
                                                            fontWeight: 'bold'
                                                        }}
                                                    >
                                                        Parameters Changed
                                                    </Box>
                                                )}
                                            </Box>
                                        </Box>
                                    )}

                                    {/* Center Cropped Image for Fariba */}
                                    {console.log('croppedImage state:', croppedImage, 'algorithmId:', algorithmId)}
                                    {croppedImage && algorithmId === 'fariba' && (
                                        <Box sx={{ mt: 3 }}>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                                                <Typography variant="h6" sx={{ fontWeight: 600, color: 'secondary.main' }}>
                                                    Center Cropped Input
                                                </Typography>
                                                <IconButton onClick={downloadCroppedImage} color="primary" sx={{
                                                    backgroundColor: 'primary.main',
                                                    color: 'white',
                                                    '&:hover': {
                                                        backgroundColor: 'primary.dark'
                                                    }
                                                }}>
                                                    <Download />
                                                </IconButton>
                                            </Box>
                                            <Box sx={{ position: 'relative' }}>
                                                <Box
                                                    component="img"
                                                    src={croppedImage}
                                                    alt="Center Cropped"
                                                    onLoad={() => console.log('Cropped image loaded successfully')}
                                                    onError={(e) => console.error('Cropped image failed to load:', e)}
                                                    sx={{
                                                        width: '100%',
                                                        height: 'auto',
                                                        maxHeight: 300,
                                                        objectFit: 'contain',
                                                        borderRadius: 1,
                                                        opacity: parametersChanged ? 0.6 : 1,
                                                        filter: parametersChanged ? 'grayscale(30%)' : 'none',
                                                        transition: 'opacity 0.3s ease, filter 0.3s ease'
                                                    }}
                                                />
                                                {parametersChanged && (
                                                    <Box
                                                        sx={{
                                                            position: 'absolute',
                                                            top: 10,
                                                            right: 10,
                                                            backgroundColor: 'rgba(255, 152, 0, 0.9)',
                                                            color: 'white',
                                                            padding: '4px 8px',
                                                            borderRadius: 1,
                                                            fontSize: '0.75rem',
                                                            fontWeight: 'bold'
                                                        }}
                                                    >
                                                        Parameters Changed
                                                    </Box>
                                                )}
                                            </Box>
                                        </Box>
                                    )}                        {/* Metrics Display */}
                                    {(psnrValue || qualityValue || processingTime || bitrate || predictedClass || decodeTime) && (
                                        <Box sx={{
                                            mt: 3,
                                            position: 'relative',
                                            opacity: parametersChanged ? 0.7 : 1,
                                            transition: 'opacity 0.3s ease'
                                        }}>
                                            <Typography variant="h6" gutterBottom>
                                                Processing Metrics
                                                {parametersChanged && (
                                                    <Chip
                                                        label="Parameters Changed - Results may be outdated"
                                                        color="warning"
                                                        size="small"
                                                        sx={{ ml: 2, fontSize: '0.75rem' }}
                                                    />
                                                )}
                                            </Typography>
                                            <Grid container spacing={2}>
                                                {psnrValue && (
                                                    <Grid item xs={6} md={3}>
                                                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                                                            <Typography variant="body2" color="text.secondary" gutterBottom>
                                                                PSNR (dB)
                                                            </Typography>
                                                            <Typography variant="h6" color="primary">
                                                                {psnrValue.toFixed(2)}
                                                            </Typography>
                                                        </Paper>
                                                    </Grid>
                                                )}
                                                {qualityValue && (
                                                    <Grid item xs={6} md={3}>
                                                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                                                            <Typography variant="body2" color="text.secondary" gutterBottom>
                                                                Image Quality Score
                                                            </Typography>
                                                            <Typography variant="h6" color="secondary">
                                                                {qualityValue.toFixed(3)}
                                                            </Typography>
                                                        </Paper>
                                                    </Grid>
                                                )}
                                                {bitrate && (
                                                    <Grid item xs={6} md={3}>
                                                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                                                            <Typography variant="body2" color="text.secondary" gutterBottom>
                                                                Bitrate (bpp)
                                                            </Typography>
                                                            <Typography variant="h6" color="warning.main">
                                                                {bitrate.toFixed(4)}
                                                            </Typography>
                                                        </Paper>
                                                    </Grid>
                                                )}
                                                {processingTime && (
                                                    <Grid item xs={6} md={3}>
                                                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                                                            <Typography variant="body2" color="text.secondary" gutterBottom>
                                                                Processing Time
                                                            </Typography>
                                                            <Typography variant="h6" color="info.main">
                                                                {processingTime.toFixed(2)}s
                                                            </Typography>
                                                        </Paper>
                                                    </Grid>
                                                )}
                                                {decodeTime && (
                                                    <Grid item xs={6} md={3}>
                                                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                                                            <Typography variant="body2" color="text.secondary" gutterBottom>
                                                                Decode Time
                                                            </Typography>
                                                            <Typography variant="h6" color="success.main">
                                                                {decodeTime.toFixed(1)} ms
                                                            </Typography>
                                                        </Paper>
                                                    </Grid>
                                                )}
                                                {predictedClass && (
                                                    <Grid item xs={12} md={6}>
                                                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                                                            <Typography variant="body2" color="text.secondary" gutterBottom>
                                                                Predicted Class
                                                            </Typography>
                                                            <Typography variant="h6" color="error.main">
                                                                {predictedClass}
                                                            </Typography>
                                                        </Paper>
                                                    </Grid>
                                                )}
                                            </Grid>
                                        </Box>
                                    )}
                                </>
                            )}
                        </Paper>
                    </Grid>
                </Grid>
            </Container>

            {/* Processing Loader Modal */}
            {processing && (
                <ProcessingLoader
                    algorithmName={algorithm?.name || 'Algorithm'}
                    estimatedTime={(algorithm?.processingTime || 2.5) * 1000} // Convert seconds to milliseconds
                />
            )}
        </>
    );
}

export default AlgorithmPage;
