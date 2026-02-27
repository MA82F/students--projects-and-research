import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
    Container,
    Typography,
    Grid,
    Card,
    CardContent,
    CardActions,
    Button,
    Box,
    Chip,
    Paper
} from '@mui/material';
import { Speed, HighQuality, Category } from '@mui/icons-material';
import ALGORITHMS from '../constants/algorithms';

function HomePage() {
    const navigate = useNavigate();
    const location = useLocation();
    const [selectedCategory, setSelectedCategory] = useState(null);

    // Get category from URL parameters
    useEffect(() => {
        const params = new URLSearchParams(location.search);
        const category = params.get('category');
        setSelectedCategory(category);
    }, [location]);

    // Group algorithms by category
    const algorithmsByCategory = ALGORITHMS.reduce((acc, algorithm) => {
        const category = algorithm.category;
        if (!acc[category]) {
            acc[category] = [];
        }
        acc[category].push(algorithm);
        return acc;
    }, {});

    // Filter by selected category if specified
    const filteredCategories = selectedCategory
        ? { [selectedCategory]: algorithmsByCategory[selectedCategory] || [] }
        : algorithmsByCategory;

    const handleAlgorithmClick = (algorithmId) => {
        navigate(`/algorithm/${algorithmId}`);
    };

    const AlgorithmCard = ({ algorithm }) => (
        <Card
            sx={{
                height: 400, // Fixed height for all cards
                width: 300, // Fixed width for all cards
                display: 'flex',
                flexDirection: 'column',
                backgroundColor: 'background.paper',
                border: '1px solid',
                borderColor: 'divider',
                boxShadow: (theme) => theme.shadows[4],
                transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out, border-color 0.2s ease-in-out',
                '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: (theme) => theme.shadows[12],
                    borderColor: 'primary.main'
                }
            }}
        >
            <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', height: 'calc(100% - 52px)' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                    <Typography variant="h6" component="h3" gutterBottom>
                        {algorithm.name}
                    </Typography>
                    <Chip
                        label={algorithm.category}
                        size="small"
                        color="primary"
                        variant="outlined"
                    />
                </Box>

                <Typography
                    variant="body2"
                    color="text.secondary"
                    paragraph
                    sx={{
                        flexGrow: 1,
                        overflow: 'hidden',
                        display: '-webkit-box',
                        WebkitLineClamp: 3,
                        WebkitBoxOrient: 'vertical',
                        mb: 2
                    }}
                >
                    {algorithm.description}
                </Typography>

                {/* Speed Rating */}
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <Speed sx={{ mr: 1, fontSize: 20 }} color="primary" />
                    <Typography variant="body2" sx={{ mr: 2, fontWeight: 500 }}>
                        Speed:
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1, mr: 1 }}>
                        {[1, 2, 3, 4, 5].map((i) => (
                            <Box
                                key={i}
                                sx={{
                                    width: 18,
                                    height: 6,
                                    bgcolor: i <= algorithm.speed ? 'primary.main' : 'grey.300',
                                    borderRadius: 1,
                                    border: '1px solid',
                                    borderColor: i <= algorithm.speed ? 'primary.dark' : 'grey.400'
                                }}
                            />
                        ))}
                    </Box>
                    <Typography variant="body2" sx={{ fontWeight: 600, color: 'primary.main' }}>
                        {algorithm.speed}/5
                    </Typography>
                </Box>

                {/* Quality Rating */}
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <HighQuality sx={{ mr: 1, fontSize: 20 }} color="secondary" />
                    <Typography variant="body2" sx={{ mr: 2, fontWeight: 500 }}>
                        Quality:
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1, mr: 1 }}>
                        {[1, 2, 3, 4, 5].map((i) => (
                            <Box
                                key={i}
                                sx={{
                                    width: 18,
                                    height: 6,
                                    bgcolor: i <= algorithm.quality ? 'secondary.main' : 'grey.300',
                                    borderRadius: 1,
                                    border: '1px solid',
                                    borderColor: i <= algorithm.quality ? 'secondary.dark' : 'grey.400'
                                }}
                            />
                        ))}
                    </Box>
                    <Typography variant="body2" sx={{ fontWeight: 600, color: 'secondary.main' }}>
                        {algorithm.quality}/5
                    </Typography>
                </Box>                {/* Features */}
                <Box sx={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: 1,
                    mt: 'auto',
                    minHeight: 60, // Minimum space for features
                    alignItems: 'flex-start'
                }}>
                    {algorithm.features.slice(0, 4).map((feature, index) => (
                        <Chip
                            key={index}
                            label={feature}
                            size="small"
                            variant="outlined"
                            sx={{ fontSize: '0.7rem' }}
                        />
                    ))}
                    {algorithm.features.length > 4 && (
                        <Chip
                            label={`+${algorithm.features.length - 4} more`}
                            size="small"
                            variant="outlined"
                            sx={{ fontSize: '0.7rem' }}
                        />
                    )}
                </Box>
            </CardContent>

            <CardActions sx={{ p: 2 }}>
                <Button
                    size="small"
                    variant="contained"
                    onClick={() => handleAlgorithmClick(algorithm.id)}
                    fullWidth
                >
                    Start Processing
                </Button>
            </CardActions>
        </Card>
    );

    return (
        <Container maxWidth="xl" sx={{ py: 4 }}>
            {/* Header */}
            <Box sx={{ textAlign: 'center', mb: 6 }}>
                <Typography variant="h3" component="h1" gutterBottom>
                    AI Image Processing Platform
                </Typography>
                <Typography variant="h6" color="text.secondary" paragraph>
                    {selectedCategory
                        ? `${selectedCategory} Algorithms`
                        : 'Choose the right algorithm for your image processing needs'
                    }
                </Typography>

                {/* Compare and Category Navigation */}
                <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 3 }}>
                    <Button
                        variant="contained"
                        color="secondary"
                        onClick={() => navigate('/comparison')}
                        sx={{
                            px: 4,
                            py: 1.5,
                            fontSize: '1.1rem',
                            fontWeight: 600
                        }}
                    >
                        Compare Algorithms
                    </Button>

                    {selectedCategory && (
                        <Button
                            variant="outlined"
                            onClick={() => navigate('/')}
                        >
                            Show All Categories
                        </Button>
                    )}
                </Box>
            </Box>            {/* Algorithm Categories */}
            {Object.entries(filteredCategories).map(([category, algorithms]) => (
                <Box key={category} sx={{ mb: 6 }}>
                    <Paper sx={{ p: 3, mb: 3 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                            <Category sx={{ mr: 2 }} color="primary" />
                            <Typography variant="h4" component="h2">
                                {category}
                            </Typography>
                        </Box>

                        <Grid container spacing={3} sx={{ justifyContent: 'center' }}>
                            {algorithms.map((algorithm) => (
                                <Grid item key={algorithm.id} sx={{ display: 'flex', justifyContent: 'center' }}>
                                    <AlgorithmCard algorithm={algorithm} />
                                </Grid>
                            ))}
                        </Grid>
                    </Paper>
                </Box>
            ))}

        </Container>
    );
}

export default HomePage;
