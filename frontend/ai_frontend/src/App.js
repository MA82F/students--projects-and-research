import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  ThemeProvider,
  CssBaseline,
  Menu,
  MenuItem,
  Box,
  createTheme
} from '@mui/material';
import ALGORITHMS from './constants/algorithms';
import HomePage from './components/HomePage';
import AlgorithmPage from './components/AlgorithmPage';
import ComparisonPage from './components/ComparisonPage';

// Dark Theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
  },
});

function NavBar() {
  const navigate = useNavigate();
  const location = useLocation();
  const [superResolutionAnchor, setSuperResolutionAnchor] = useState(null);
  const [compressionAnchor, setCompressionAnchor] = useState(null);

  const superResolutionAlgorithms = ALGORITHMS.filter(alg => alg.category === 'Super Resolution');
  const compressionAlgorithms = ALGORITHMS.filter(alg => alg.category === 'Compression');

  const handleSuperResolutionClick = (event) => {
    setSuperResolutionAnchor(event.currentTarget);
  };

  const handleCompressionClick = (event) => {
    setCompressionAnchor(event.currentTarget);
  };

  const handleClose = () => {
    setSuperResolutionAnchor(null);
    setCompressionAnchor(null);
  };

  const handleAlgorithmSelect = (algorithmId) => {
    navigate(`/algorithm/${algorithmId}`);
    handleClose();
  };

  const handleAllSelect = (category) => {
    navigate(`/?category=${category}`);
    handleClose();
  };

  return (
    <AppBar position="static">
      <Toolbar>
        <Box sx={{ flexGrow: 1 }}>
          <Typography
            variant="h6"
            component="span"
            sx={{
              cursor: 'pointer',
              display: 'inline-block',
              padding: '8px 12px',
              borderRadius: '8px',
              transition: 'all 0.2s ease-in-out',
              '&:hover': {
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                transform: 'scale(1.02)',
                color: 'primary.light'
              }
            }}
            onClick={() => navigate('/')}
          >
            AI Image Processing Platform
          </Typography>
        </Box>

        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            color="inherit"
            onClick={() => navigate('/comparison')}
            variant={location.pathname === '/comparison' ? 'outlined' : 'text'}
          >
            Compare Algorithms
          </Button>

          <Button
            color="inherit"
            onClick={handleSuperResolutionClick}
            variant={location.pathname.includes('/algorithm/') &&
              superResolutionAlgorithms.some(alg =>
                location.pathname.includes(alg.id)) ? 'outlined' : 'text'}
          >
            Super Resolution ▼
          </Button>

          <Button
            color="inherit"
            onClick={handleCompressionClick}
            variant={location.pathname.includes('/algorithm/fariba') ? 'outlined' : 'text'}
          >
            Compression ▼
          </Button>
        </Box>

        {/* Super Resolution Menu */}
        <Menu
          anchorEl={superResolutionAnchor}
          open={Boolean(superResolutionAnchor)}
          onClose={handleClose}
          PaperProps={{
            sx: {
              bgcolor: 'background.paper',
              border: '1px solid',
              borderColor: 'divider',
            }
          }}
        >
          <MenuItem
            onClick={() => handleAllSelect('Super Resolution')}
            sx={{ fontWeight: 600, color: 'primary.main' }}
          >
            All Super Resolution
          </MenuItem>
          {superResolutionAlgorithms.map((algorithm) => (
            <MenuItem
              key={algorithm.id}
              onClick={() => handleAlgorithmSelect(algorithm.id)}
            >
              {algorithm.name}
            </MenuItem>
          ))}
        </Menu>

        {/* Compression Menu */}
        <Menu
          anchorEl={compressionAnchor}
          open={Boolean(compressionAnchor)}
          onClose={handleClose}
          PaperProps={{
            sx: {
              bgcolor: 'background.paper',
              border: '1px solid',
              borderColor: 'divider',
            }
          }}
        >
          <MenuItem
            onClick={() => handleAllSelect('Compression')}
            sx={{ fontWeight: 600, color: 'primary.main' }}
          >
            All Compression
          </MenuItem>
          {compressionAlgorithms.map((algorithm) => (
            <MenuItem
              key={algorithm.id}
              onClick={() => handleAlgorithmSelect(algorithm.id)}
            >
              {algorithm.name}
            </MenuItem>
          ))}
        </Menu>
      </Toolbar>
    </AppBar>
  );
}

function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Router>
        <NavBar />
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/algorithm/:algorithmId" element={<AlgorithmPage />} />
          <Route path="/comparison" element={<ComparisonPage />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
