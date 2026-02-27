import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    Container,
    Grid,
    Typography,
    Button,
    Box,
    Paper,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Chip,
    Alert,
    Card,
    CardContent,
    CircularProgress,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Divider
} from '@mui/material';
import { ArrowBack, Compare, PlayArrow, TrendingUp, Timer, HighQuality } from '@mui/icons-material';
import ALGORITHMS from '../constants/algorithms';

function ComparisonPage() {
    const navigate = useNavigate();

    // Group algorithms by category
    const algorithmsByCategory = ALGORITHMS.reduce((acc, algorithm) => {
        if (!acc[algorithm.category]) {
            acc[algorithm.category] = [];
        }
        acc[algorithm.category].push(algorithm);
        return acc;
    }, {});

    const [selectedCategory, setSelectedCategory] = useState('Super Resolution');
    const [selectedAlgorithms, setSelectedAlgorithms] = useState([]);
    const [algorithmParameters, setAlgorithmParameters] = useState({});
    const [selectedImage, setSelectedImage] = useState(null);
    const [results, setResults] = useState({});
    const [processing, setProcessing] = useState(false);

    const categories = Object.keys(algorithmsByCategory);
    const availableAlgorithms = algorithmsByCategory[selectedCategory] || [];

    const handleCategoryChange = (event) => {
        setSelectedCategory(event.target.value);
        setSelectedAlgorithms([]);
        setAlgorithmParameters({});
        setResults({});
    };

    const handleAlgorithmToggle = (algorithmId) => {
        if (selectedAlgorithms.includes(algorithmId)) {
            setSelectedAlgorithms(selectedAlgorithms.filter(id => id !== algorithmId));
            const newParams = { ...algorithmParameters };
            delete newParams[algorithmId];
            setAlgorithmParameters(newParams);
        } else {
            setSelectedAlgorithms([...selectedAlgorithms, algorithmId]);
            // Initialize default parameters
            const algorithm = ALGORITHMS.find(alg => alg.id === algorithmId);
            if (algorithm.parameters) {
                const defaultParams = {};
                Object.keys(algorithm.parameters).forEach(paramKey => {
                    defaultParams[paramKey] = algorithm.parameters[paramKey].default;
                });
                setAlgorithmParameters({
                    ...algorithmParameters,
                    [algorithmId]: defaultParams
                });
            }
        }
    };

    const handleParameterChange = (algorithmId, parameterName, value) => {
        setAlgorithmParameters({
            ...algorithmParameters,
            [algorithmId]: {
                ...algorithmParameters[algorithmId],
                [parameterName]: value
            }
        });
    };

    const handleImageUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
            setSelectedImage(file);
            setResults({});
        }
    };

    const runComparison = async () => {
        if (!selectedImage || selectedAlgorithms.length === 0) {
            return;
        }

        setProcessing(true);
        const newResults = {};

        try {
            for (const algorithmId of selectedAlgorithms) {
                const formData = new FormData();
                formData.append('image', selectedImage);
                formData.append('algorithm', algorithmId);

                // Add algorithm-specific parameters
                if (algorithmParameters[algorithmId]) {
                    Object.keys(algorithmParameters[algorithmId]).forEach(key => {
                        formData.append(key, algorithmParameters[algorithmId][key]);
                    });
                }

                const response = await fetch('http://localhost:8000/api/upload/', {
                    method: 'POST',
                    body: formData,
                });

                const data = await response.json();
                newResults[algorithmId] = data;
            }

            setResults(newResults);
        } catch (error) {
            console.error('Comparison failed:', error);
        } finally {
            setProcessing(false);
        }
    };

    return (
        <Container maxWidth="xl" sx={{ py: 4 }}>
            {/* Header */}
            <Box sx={{ mb: 4 }}>
                <Button
                    startIcon={<ArrowBack />}
                    onClick={() => navigate('/')}
                    sx={{ mb: 2 }}
                >
                    Back to Home
                </Button>
                <Typography variant="h3" component="h1" gutterBottom>
                    Algorithm Comparison
                </Typography>
                <Typography variant="h6" color="text.secondary">
                    Compare algorithms within the same category for fair evaluation
                </Typography>
            </Box>

            <Grid container spacing={4}>
                {/* Left Panel - Configuration */}
                <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 3 }}>
                        <Typography variant="h5" gutterBottom>
                            Configuration
                        </Typography>

                        {/* Category Selection */}
                        <FormControl fullWidth sx={{ mb: 3 }}>
                            <InputLabel>Category</InputLabel>
                            <Select
                                value={selectedCategory}
                                onChange={handleCategoryChange}
                                label="Category"
                            >
                                {categories.map(category => (
                                    <MenuItem key={category} value={category}>
                                        {category}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>

                        {/* Algorithm Selection */}
                        <Typography variant="h6" gutterBottom>
                            Select Algorithms (max 4)
                        </Typography>
                        <Box sx={{ mb: 3 }}>
                            {availableAlgorithms.map(algorithm => (
                                <Chip
                                    key={algorithm.id}
                                    label={algorithm.name}
                                    onClick={() => handleAlgorithmToggle(algorithm.id)}
                                    color={selectedAlgorithms.includes(algorithm.id) ? 'primary' : 'default'}
                                    variant={selectedAlgorithms.includes(algorithm.id) ? 'filled' : 'outlined'}
                                    sx={{ m: 0.5 }}
                                    disabled={!selectedAlgorithms.includes(algorithm.id) && selectedAlgorithms.length >= 4}
                                />
                            ))}
                        </Box>

                        {/* Parameters for each selected algorithm */}
                        {selectedAlgorithms.map(algorithmId => {
                            const algorithm = ALGORITHMS.find(alg => alg.id === algorithmId);
                            if (!algorithm.parameters) return null;

                            return (
                                <Card key={algorithmId} sx={{ mb: 2 }}>
                                    <CardContent>
                                        <Typography variant="h6" gutterBottom>
                                            {algorithm.name} Parameters
                                        </Typography>
                                        {Object.keys(algorithm.parameters).map(paramKey => {
                                            const param = algorithm.parameters[paramKey];
                                            return (
                                                <FormControl fullWidth sx={{ mb: 2 }} key={paramKey}>
                                                    <InputLabel>{param.label}</InputLabel>
                                                    <Select
                                                        value={algorithmParameters[algorithmId]?.[paramKey] || param.default}
                                                        onChange={(e) => handleParameterChange(algorithmId, paramKey, e.target.value)}
                                                        label={param.label}
                                                    >
                                                        {param.options.map(option => (
                                                            <MenuItem key={option.value} value={option.value}>
                                                                {option.label}
                                                            </MenuItem>
                                                        ))}
                                                    </Select>
                                                </FormControl>
                                            );
                                        })}
                                    </CardContent>
                                </Card>
                            );
                        })}

                        {/* Image Upload */}
                        <Box sx={{ mb: 3 }}>
                            <input
                                accept="image/*"
                                style={{ display: 'none' }}
                                id="upload-image"
                                type="file"
                                onChange={handleImageUpload}
                            />
                            <label htmlFor="upload-image">
                                <Button variant="outlined" component="span" fullWidth>
                                    {selectedImage ? selectedImage.name : 'Upload Test Image'}
                                </Button>
                            </label>
                        </Box>

                        {/* Run Comparison Button */}
                        <Button
                            variant="contained"
                            startIcon={processing ? <CircularProgress size={20} /> : <PlayArrow />}
                            onClick={runComparison}
                            disabled={!selectedImage || selectedAlgorithms.length === 0 || processing}
                            fullWidth
                        >
                            {processing ? 'Processing...' : 'Run Comparison'}
                        </Button>
                    </Paper>
                </Grid>

                {/* Right Panel - Results */}
                <Grid item xs={12} md={8}>
                    <Paper sx={{ p: 3 }}>
                        <Typography variant="h5" gutterBottom>
                            Comparison Results
                        </Typography>

                        {Object.keys(results).length === 0 ? (
                            <Alert severity="info">
                                Configure algorithms and upload an image to start comparison
                            </Alert>
                        ) : (
                            <Box>
                                {/* Comparison Table */}
                                <Box sx={{ mb: 4 }}>
                                    <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                                        <TrendingUp sx={{ mr: 1 }} />
                                        Performance Comparison
                                    </Typography>
                                    <TableContainer component={Paper} sx={{ mb: 3 }}>
                                        <Table size="small">
                                            <TableHead>
                                                <TableRow sx={{ backgroundColor: 'grey.800' }}>
                                                    <TableCell sx={{ fontWeight: 'bold', color: 'white' }}>
                                                        Algorithm
                                                    </TableCell>
                                                    <TableCell align="center" sx={{ fontWeight: 'bold', color: 'white' }}>
                                                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                                            <HighQuality sx={{ mr: 0.5, fontSize: 16 }} />
                                                            PSNR (dB)
                                                        </Box>
                                                    </TableCell>
                                                    <TableCell align="center" sx={{ fontWeight: 'bold', color: 'white' }}>
                                                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                                            <Timer sx={{ mr: 0.5, fontSize: 16 }} />
                                                            Time (s)
                                                        </Box>
                                                    </TableCell>
                                                    <TableCell align="center" sx={{ fontWeight: 'bold', color: 'white' }}>
                                                        Quality Score
                                                    </TableCell>
                                                    <TableCell align="center" sx={{ fontWeight: 'bold', color: 'white' }}>
                                                        Parameters
                                                    </TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                {selectedAlgorithms.map(algorithmId => {
                                                    const algorithm = ALGORITHMS.find(alg => alg.id === algorithmId);
                                                    const result = results[algorithmId];
                                                    const params = algorithmParameters[algorithmId] || {};

                                                    // Get best values for highlighting
                                                    const allPsnr = selectedAlgorithms.map(id => results[id]?.psnr).filter(Boolean);
                                                    const allTimes = selectedAlgorithms.map(id => results[id]?.processing_time).filter(Boolean);
                                                    const allQuality = selectedAlgorithms.map(id => results[id]?.image_quality).filter(Boolean);

                                                    const bestPsnr = Math.max(...allPsnr);
                                                    const bestTime = Math.min(...allTimes);
                                                    const bestQuality = Math.max(...allQuality);

                                                    return (
                                                        <TableRow key={algorithmId}>
                                                            <TableCell>
                                                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                                                    <Chip
                                                                        label={algorithm.name}
                                                                        color="primary"
                                                                        variant="outlined"
                                                                        size="small"
                                                                    />
                                                                </Box>
                                                            </TableCell>
                                                            <TableCell align="center">
                                                                {result?.psnr ? (
                                                                    <Typography
                                                                        variant="body2"
                                                                        sx={{
                                                                            fontWeight: result.psnr === bestPsnr ? 'bold' : 'normal',
                                                                            color: result.psnr === bestPsnr ? 'white' : 'text.primary',
                                                                            backgroundColor: result.psnr === bestPsnr ? 'success.main' : 'transparent',
                                                                            px: 1,
                                                                            py: 0.5,
                                                                            borderRadius: 1,
                                                                            display: 'inline-block'
                                                                        }}
                                                                    >
                                                                        {result.psnr.toFixed(2)}
                                                                    </Typography>
                                                                ) : '-'}
                                                            </TableCell>
                                                            <TableCell align="center">
                                                                {result?.processing_time ? (
                                                                    <Typography
                                                                        variant="body2"
                                                                        sx={{
                                                                            fontWeight: result.processing_time === bestTime ? 'bold' : 'normal',
                                                                            color: result.processing_time === bestTime ? 'white' : 'text.primary',
                                                                            backgroundColor: result.processing_time === bestTime ? 'success.main' : 'transparent',
                                                                            px: 1,
                                                                            py: 0.5,
                                                                            borderRadius: 1,
                                                                            display: 'inline-block'
                                                                        }}
                                                                    >
                                                                        {result.processing_time.toFixed(2)}
                                                                    </Typography>
                                                                ) : '-'}
                                                            </TableCell>
                                                            <TableCell align="center">
                                                                {result?.image_quality ? (
                                                                    <Typography
                                                                        variant="body2"
                                                                        sx={{
                                                                            fontWeight: result.image_quality === bestQuality ? 'bold' : 'normal',
                                                                            color: result.image_quality === bestQuality ? 'white' : 'text.primary',
                                                                            backgroundColor: result.image_quality === bestQuality ? 'success.main' : 'transparent',
                                                                            px: 1,
                                                                            py: 0.5,
                                                                            borderRadius: 1,
                                                                            display: 'inline-block'
                                                                        }}
                                                                    >
                                                                        {result.image_quality.toFixed(2)}
                                                                    </Typography>
                                                                ) : '-'}
                                                            </TableCell>
                                                            <TableCell align="center">
                                                                <Box sx={{ display: 'flex', gap: 0.5, justifyContent: 'center', flexWrap: 'wrap' }}>
                                                                    {Object.entries(params).map(([key, value]) => (
                                                                        <Chip
                                                                            key={key}
                                                                            label={`${key}: ${value}`}
                                                                            size="small"
                                                                            variant="outlined"
                                                                            sx={{ fontSize: '0.7rem' }}
                                                                        />
                                                                    ))}
                                                                </Box>
                                                            </TableCell>
                                                        </TableRow>
                                                    );
                                                })}
                                            </TableBody>
                                        </Table>
                                    </TableContainer>
                                </Box>

                                <Divider sx={{ my: 3 }} />

                                {/* Visual Results Grid */}
                                <Typography variant="h6" gutterBottom>
                                    Visual Results
                                </Typography>
                                <Grid container spacing={2}>
                                    {selectedAlgorithms.map(algorithmId => {
                                        const algorithm = ALGORITHMS.find(alg => alg.id === algorithmId);
                                        const result = results[algorithmId];

                                        return (
                                            <Grid item xs={12} sm={6} key={algorithmId}>
                                                <Card>
                                                    <CardContent>
                                                        <Typography variant="h6" gutterBottom>
                                                            {algorithm.name}
                                                        </Typography>

                                                        {result?.processed_image && (
                                                            <img
                                                                src={`http://localhost:8000${result.processed_image}`}
                                                                alt={`${algorithm.name} result`}
                                                                style={{ width: '100%', marginBottom: 16 }}
                                                            />
                                                        )}

                                                        {result && (
                                                            <Box>
                                                                {result.psnr && (
                                                                    <Typography variant="body2">
                                                                        PSNR: {result.psnr.toFixed(2)} dB
                                                                    </Typography>
                                                                )}
                                                                {result.processing_time && (
                                                                    <Typography variant="body2">
                                                                        Time: {result.processing_time.toFixed(2)}s
                                                                    </Typography>
                                                                )}
                                                                {result.image_quality && (
                                                                    <Typography variant="body2">
                                                                        Quality: {result.image_quality.toFixed(2)}
                                                                    </Typography>
                                                                )}
                                                            </Box>
                                                        )}
                                                    </CardContent>
                                                </Card>
                                            </Grid>
                                        );
                                    })}
                                </Grid>
                            </Box>
                        )}
                    </Paper>
                </Grid>
            </Grid>
        </Container>
    );
}

export default ComparisonPage;
