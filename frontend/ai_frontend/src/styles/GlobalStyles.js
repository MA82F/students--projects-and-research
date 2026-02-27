import { GlobalStyles as MuiGlobalStyles } from '@mui/material';

const GlobalStyles = () => {
  return (
    <MuiGlobalStyles
      styles={{
        '*': {
          margin: 0,
          padding: 0,
          boxSizing: 'border-box',
        },
        html: {
          WebkitFontSmoothing: 'antialiased',
          MozOsxFontSmoothing: 'grayscale',
          height: '100%',
          width: '100%',
        },
        body: {
          height: '100%',
          width: '100%',
          backgroundColor: '#f8f9fa',
          color: '#2b2d42',
        },
        '#root': {
          height: '100%',
          width: '100%',
        },
        '.MuiContainer-root': {
          paddingLeft: '24px',
          paddingRight: '24px',
          '@media (min-width: 600px)': {
            paddingLeft: '24px',
            paddingRight: '24px',
          },
        },
        a: {
          textDecoration: 'none',
          color: 'inherit',
        },
        '::-webkit-scrollbar': {
          width: '8px',
          height: '8px',
        },
        '::-webkit-scrollbar-track': {
          background: '#f1f1f1',
          borderRadius: '4px',
        },
        '::-webkit-scrollbar-thumb': {
          background: '#888',
          borderRadius: '4px',
          '&:hover': {
            background: '#555',
          },
        },
      }}
    />
  );
};

export default GlobalStyles;
