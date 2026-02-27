import React, { useState, useEffect } from 'react';
import {
    Box,
    Typography,
    LinearProgress,
    CircularProgress,
    Stack,
    Fade,
    keyframes,
    Modal,
    Backdrop
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
    CloudUpload,
    Settings,
    Psychology,
    Download,
    CheckCircle
} from '@mui/icons-material';

// Animated gradient background
const pulseAnimation = keyframes`
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
`;

const AnimatedBox = styled(Box)(({ theme }) => ({
    background: 'linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab)',
    backgroundSize: '400% 400%',
    animation: `${pulseAnimation} 4s ease-in-out infinite`,
    borderRadius: theme.spacing(2),
    padding: theme.spacing(4),
    color: 'white',
    textAlign: 'center',
    minHeight: '400px',
    maxWidth: '500px',
    width: '90%',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    outline: 'none',
    boxShadow: '0 20px 40px rgba(0,0,0,0.3)',
}));

const StepContainer = styled(Box)(({ theme, active }) => ({
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(2),
    padding: theme.spacing(1.5),
    borderRadius: theme.spacing(1),
    backgroundColor: active ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.1)',
    transition: 'all 0.3s ease',
    transform: active ? 'scale(1.05)' : 'scale(1)',
    opacity: active ? 1 : 0.7,
}));

const IconContainer = styled(Box)(({ theme, active }) => ({
    width: 40,
    height: 40,
    borderRadius: '50%',
    backgroundColor: active ? 'white' : 'rgba(255,255,255,0.3)',
    color: active ? theme.palette.primary.main : 'white',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.3s ease',
}));

const ProcessingLoader = ({ algorithmName = 'Algorithm', estimatedTime = 8000 }) => {
    const [currentStep, setCurrentStep] = useState(0);
    const [progress, setProgress] = useState(0);
    const [steps, setSteps] = useState([]);

    useEffect(() => {
        // Calculate step durations based on estimated total time
        const uploadTime = Math.max(500, estimatedTime * 0.15); // 15% for upload
        const initTime = Math.max(800, estimatedTime * 0.25); // 25% for model init
        const processTime = Math.max(1000, estimatedTime * 0.45); // 45% for processing
        const resultTime = Math.max(500, estimatedTime * 0.1); // 10% for results
        const completeTime = Math.max(300, estimatedTime * 0.05); // 5% for complete

        // For very fast algorithms (< 2 seconds), use simplified steps
        let stepsArray;
        if (estimatedTime < 2000) {
            stepsArray = [
                {
                    icon: <CloudUpload />,
                    title: 'Uploading Image',
                    description: 'Preparing your image...',
                    duration: estimatedTime * 0.3
                },
                {
                    icon: <Psychology />,
                    title: 'Processing Image',
                    description: `Enhancing with ${algorithmName}...`,
                    duration: estimatedTime * 0.6
                },
                {
                    icon: <CheckCircle />,
                    title: 'Complete!',
                    description: 'Processing finished!',
                    duration: estimatedTime * 0.1
                }
            ];
        } else {
            stepsArray = [
                {
                    icon: <CloudUpload />,
                    title: 'Uploading Image',
                    description: 'Preparing your image for processing...',
                    duration: uploadTime
                },
                {
                    icon: <Settings />,
                    title: 'Initializing Model',
                    description: `Loading ${algorithmName} neural network...`,
                    duration: initTime
                },
                {
                    icon: <Psychology />,
                    title: 'Processing Image',
                    description: 'Enhancing image with AI technology...',
                    duration: processTime
                },
                {
                    icon: <Download />,
                    title: 'Generating Results',
                    description: 'Preparing enhanced image and metrics...',
                    duration: resultTime
                },
                {
                    icon: <CheckCircle />,
                    title: 'Complete!',
                    description: 'Processing finished successfully.',
                    duration: completeTime
                }
            ];
        }

        setSteps(stepsArray);

        let progressTimer;
        let stepTimer;

        const updateProgress = () => {
            // Calculate interval based on estimated time to reach 100% in time
            const interval = estimatedTime / 100; // milliseconds per percent

            progressTimer = setInterval(() => {
                setProgress((prev) => {
                    if (prev >= 100) {
                        clearInterval(progressTimer);
                        return 100;
                    }
                    return prev + 1;
                });
            }, interval);
        };

        const updateSteps = () => {
            let totalTime = 0;
            stepsArray.forEach((step, index) => {
                stepTimer = setTimeout(() => {
                    setCurrentStep(index);
                }, totalTime);
                totalTime += step.duration;
            });
        };

        updateProgress();
        updateSteps();

        return () => {
            clearInterval(progressTimer);
            clearTimeout(stepTimer);
        };
    }, [algorithmName, estimatedTime]);

    return (
        <Modal
            open={true}
            BackdropComponent={Backdrop}
            BackdropProps={{
                sx: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    backdropFilter: 'blur(5px)'
                }
            }}
        >
            <AnimatedBox>
                <Stack spacing={3} width="100%" maxWidth="500px">
                    {/* Header */}
                    <Box textAlign="center">
                        <Typography variant="h4" fontWeight="bold" gutterBottom>
                            Processing Image
                        </Typography>
                        <Typography variant="h6" sx={{ opacity: 0.9 }}>
                            Using {algorithmName} Algorithm
                        </Typography>
                    </Box>

                    {/* Progress Bar */}
                    <Box sx={{ width: '100%' }}>
                        <LinearProgress
                            variant="determinate"
                            value={progress}
                            sx={{
                                height: 8,
                                borderRadius: 4,
                                backgroundColor: 'rgba(255,255,255,0.3)',
                                '& .MuiLinearProgress-bar': {
                                    backgroundColor: 'white',
                                    borderRadius: 4,
                                }
                            }}
                        />
                        <Typography variant="body2" sx={{ mt: 1, opacity: 0.9 }}>
                            {progress}% Complete
                        </Typography>
                    </Box>

                    {/* Steps */}
                    <Stack spacing={1.5}>
                        {steps.map((step, index) => (
                            <Fade key={index} in={index <= currentStep} timeout={500}>
                                <StepContainer active={index === currentStep}>
                                    <IconContainer active={index === currentStep}>
                                        {index === currentStep ? (
                                            <CircularProgress size={20} sx={{ color: 'inherit' }} />
                                        ) : index < currentStep ? (
                                            <CheckCircle />
                                        ) : (
                                            step.icon
                                        )}
                                    </IconContainer>
                                    <Box flex={1}>
                                        <Typography variant="subtitle1" fontWeight="bold">
                                            {step.title}
                                        </Typography>
                                        <Typography variant="body2" sx={{ opacity: 0.9 }}>
                                            {step.description}
                                        </Typography>
                                    </Box>
                                </StepContainer>
                            </Fade>
                        ))}
                    </Stack>

                    {/* Footer */}
                    <Box textAlign="center" sx={{ mt: 2 }}>
                        <Typography variant="body2" sx={{ opacity: 0.8 }}>
                            Please wait while we enhance your image...
                        </Typography>
                    </Box>
                </Stack>
            </AnimatedBox>
        </Modal>
    );
};

export default ProcessingLoader;
