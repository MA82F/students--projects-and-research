import React from 'react';
import { styled } from '@mui/material/styles';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  Tooltip, 
  IconButton, 
  Chip,
  CircularProgress
} from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import ImageIcon from '@mui/icons-material/Image';
import TimerIcon from '@mui/icons-material/Timer';

const StyledCard = styled(Card)(({ theme, active }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  transition: 'all 0.3s ease-in-out',
  cursor: 'pointer',
  border: active ? `2px solid ${theme.palette.primary.main}` : '2px solid transparent',
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: theme.shadows[8],
  },
}));

const CardHeader = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  borderBottom: `1px solid ${theme.palette.divider}`,
}));

const CardFooter = styled(Box)(({ theme }) => ({
  padding: theme.spacing(1.5, 2),
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  backgroundColor: theme.palette.background.paper,
  borderTop: `1px solid ${theme.palette.divider}`,
  borderBottomLeftRadius: theme.shape.borderRadius,
  borderBottomRightRadius: theme.shape.borderRadius,
}));

const AlgorithmImage = styled(Box)(({ theme }) => ({
  width: '100%',
  paddingTop: '56.25%', // 16:9 aspect ratio
  position: 'relative',
  backgroundColor: theme.palette.grey[100],
  borderRadius: theme.shape.borderRadius,
  overflow: 'hidden',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  '& img': {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    objectFit: 'cover',
  },
}));

const AlgorithmCard = ({ 
  algorithm,
  name,
  description,
  complexity,
  processingTime,
  image,
  active,
  onClick,
  loading,
  result
}) => {
  // Handle both object and string algorithm props for backward compatibility
  const algorithmName = algorithm?.name || name || (typeof algorithm === 'string' ? algorithm : '');
  const algorithmDescription = algorithm?.description || description || '';
  const algorithmComplexity = algorithm?.complexity || complexity || 'Medium';
  const algorithmProcessingTime = algorithm?.processingTime || processingTime || 'N/A';
  return (
    <StyledCard 
      elevation={3} 
      active={active}
      onClick={onClick}
    >
      <CardHeader>
        <Typography variant="h6" component="h3" noWrap>
          {algorithmName.replace(/_/g, ' ')}
        </Typography>
        <Tooltip title={algorithmDescription || 'No description available'} arrow>
          <IconButton size="small" onClick={(e) => e.stopPropagation()}>
            <InfoIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </CardHeader>
      
      <CardContent sx={{ flexGrow: 1, p: 2 }}>
        <AlgorithmImage>
          {loading ? (
            <CircularProgress size={40} />
          ) : result?.processed_image ? (
            <img 
              src={result.processed_image} 
              alt={`Processed with ${algorithmName}`} 
            />
          ) : (
            <ImageIcon sx={{ fontSize: 48, color: 'action.disabled' }} />
          )}
        </AlgorithmImage>

        <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {algorithmComplexity && (
            <Chip 
              size="small" 
              label={`Complexity: ${algorithmComplexity}`} 
              color="primary" 
              variant="outlined"
            />
          )}
          {algorithmProcessingTime && (
            <Chip 
              size="small" 
              icon={<TimerIcon fontSize="small" />} 
              label={`~${algorithmProcessingTime}`} 
              variant="outlined"
            />
          )}
        </Box>
      </CardContent>
      
      <CardFooter>
        <Typography variant="caption" color="textSecondary">
          {algorithmName}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {result && (
            <Chip 
              size="small" 
              label="Processed" 
              color="primary" 
              variant="filled"
              sx={{ ml: 1 }}
            />
          )}
        </Box>
      </CardFooter>
    </StyledCard>
  );
};

export default AlgorithmCard;
