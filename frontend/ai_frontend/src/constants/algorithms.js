const ALGORITHMS = [
  {
    id: 'ninasr',
    name: 'NinaSR',
    description: 'Scalable Super-Resolution with configurable quality levels',
    englishDescription: 'Scalable Super-Resolution with configurable quality levels. NinaSR offers multiple variants (B0, B1, B2) with different trade-offs between speed and quality, and supports multiple scaling factors (2x, 3x, 4x, 8x).',
    category: 'Super Resolution',
    speed: 4,
    quality: 4,
    complexity: 'Medium',
    processingTime: 2.5,
    tags: ['scalable', 'configurable', 'efficient'],
    features: ['Multiple Variants', 'Flexible Scaling', 'Speed/Quality Trade-off'],
    parameters: {
      scale: {
        type: 'select',
        options: [
          { value: 2, label: '2x' },
          { value: 3, label: '3x' },
          { value: 4, label: '4x' },
          { value: 8, label: '8x' }
        ],
        default: 3,
        label: 'Scaling Factor'
      },
      variant: {
        type: 'select',
        options: [
          { value: 'b0', label: 'B0 (Fast - 0.1M params)', description: 'Fast but medium quality' },
          { value: 'b1', label: 'B1 (Balanced - 1M params)', description: 'Balance between speed and quality' },
          { value: 'b2', label: 'B2 (Quality - 10M params)', description: 'High quality but slower' }
        ],
        default: 'b1',
        label: 'Model Variant'
      }
    }
  },
  {
    id: 'edsr',
    name: 'EDSR',
    description: 'Enhanced Deep Super-Resolution full model',
    englishDescription: 'Enhanced Deep Super-Resolution (Full Version). Uses 32 residual blocks with 256 features for maximum quality. Larger model (40.7M parameters) with superior PSNR results but slower processing.',
    category: 'Super Resolution',
    speed: 2,
    quality: 5,
    complexity: 'High',
    processingTime: 4.0,
    tags: ['state-of-the-art', 'high-quality', 'accurate'],
    features: ['32 ResBlocks', '256 Features', 'Maximum Quality'],
    parameters: {
      scale: {
        type: 'select',
        options: [
          { value: 2, label: '2x', description: 'PSNR ~38.19 dB' },
          { value: 3, label: '3x', description: 'PSNR ~34.68 dB' },
          { value: 4, label: '4x', description: 'PSNR ~32.48 dB' }
        ],
        default: 2,
        label: 'Scaling Factor'
      }
    }
  },
  {
    id: 'edsr_baseline',
    name: 'EDSR Baseline',
    description: 'Enhanced Deep Super-Resolution baseline model',
    englishDescription: 'Enhanced Deep Super-Resolution (Baseline Version). Uses 16 residual blocks with 64 features for faster inference. Smaller model (1.37M parameters) with good balance between speed and quality.',
    category: 'Super Resolution',
    speed: 4,
    quality: 4,
    complexity: 'Medium',
    processingTime: 3.0,
    tags: ['baseline', 'efficient', 'balanced'],
    features: ['16 ResBlocks', '64 Features', 'Fast Processing'],
    parameters: {
      scale: {
        type: 'select',
        options: [
          { value: 2, label: '2x', description: 'PSNR ~37.98 dB' },
          { value: 3, label: '3x', description: 'PSNR ~34.37 dB' },
          { value: 4, label: '4x', description: 'PSNR ~32.09 dB' }
        ],
        default: 2,
        label: 'Scaling Factor'
      }
    }
  },
  {
    id: 'rdn',
    name: 'RDN',
    description: 'Residual Dense Network for super-resolution',
    englishDescription: 'Residual Dense Network for super-resolution. RDN makes full use of the hierarchical features from all the convolutional layers through dense connections, allowing efficient information flow and feature reuse.',
    category: 'Super Resolution',
    speed: 3,
    quality: 5,
    complexity: 'High',
    processingTime: 3.8,
    tags: ['residual', 'dense'],
    features: ['Dense Connections', 'High Quality', 'Detail Preservation'],
    parameters: {
      scale: {
        type: 'select',
        options: [
          { value: 2, label: '2x', description: 'PSNR ~38.12 dB' },
          { value: 3, label: '3x', description: 'PSNR ~33.98 dB' },
          { value: 4, label: '4x', description: 'PSNR ~32.35 dB' }
        ],
        default: 2,
        label: 'Scaling Factor'
      }
    }
  },
  {
    id: 'rcan',
    name: 'RCAN',
    description: 'Residual Channel Attention Network',
    englishDescription: 'Residual Channel Attention Network. RCAN introduces channel attention mechanism to adaptively rescale channel-wise features by considering interdependencies among channels, leading to more informative representations.',
    category: 'Super Resolution',
    speed: 2,
    quality: 5,
    complexity: 'Very High',
    processingTime: 5.5,
    tags: ['attention', 'residual'],
    features: ['Attention Mechanism', 'Exceptional Accuracy', 'Complex Images'],
    parameters: {
      scale: {
        type: 'select',
        options: [
          { value: 2, label: '2x', description: 'PSNR ~38.27 dB' },
          { value: 3, label: '3x', description: 'PSNR ~34.76 dB' },
          { value: 4, label: '4x', description: 'PSNR ~32.64 dB' }
        ],
        default: 2,
        label: 'Scaling Factor'
      }
    }
  },
  {
    id: 'carn',
    name: 'CARN',
    description: 'Cascading Residual Network',
    englishDescription: 'Cascading Residual Network. CARN efficiently uses multi-level representations through cascading connections and performs feature aggregation to improve both accuracy and efficiency.',
    category: 'Super Resolution',
    speed: 4,
    quality: 4,
    complexity: 'Medium',
    processingTime: 2.8,
    tags: ['cascading', 'efficient'],
    features: ['Cascading Aggregation', 'Efficient', 'Balanced'],
    parameters: {
      scale: {
        type: 'select',
        options: [
          { value: 2, label: '2x', description: 'PSNR ~37.76 dB' },
          { value: 3, label: '3x', description: 'PSNR ~34.29 dB' },
          { value: 4, label: '4x', description: 'PSNR ~32.13 dB' },
          { value: 8, label: '8x', description: 'PSNR ~26.54 dB' }
        ],
        default: 2,
        label: 'Scaling Factor'
      }
    }
  },
  {
    id: 'carn_m',
    name: 'CARN-M',
    description: 'Cascading Residual Network Mobile',
    englishDescription: 'Cascading Residual Network Mobile. CARN-M is a lightweight version of CARN designed for mobile devices, offering good quality with reduced computational requirements.',
    category: 'Super Resolution',
    speed: 5,
    quality: 3,
    complexity: 'Low',
    processingTime: 1.8,
    tags: ['mobile', 'lightweight', 'efficient'],
    features: ['Mobile Optimized', 'Fast Processing', 'Compact Model'],
    parameters: {
      scale: {
        type: 'select',
        options: [
          { value: 2, label: '2x', description: 'PSNR ~37.53 dB' },
          { value: 3, label: '3x', description: 'PSNR ~33.99 dB' },
          { value: 4, label: '4x', description: 'PSNR ~31.92 dB' },
          { value: 8, label: '8x', description: 'PSNR ~26.21 dB' }
        ],
        default: 2,
        label: 'Scaling Factor'
      }
    }
  },
  {
    id: 'vdsr',
    name: 'VDSR',
    description: 'Very Deep Super-Resolution network',
    englishDescription: 'Very Deep Super-Resolution network. VDSR uses very deep convolutional networks (up to 20 layers) with global residual learning to achieve accurate image super-resolution across multiple scales.',
    category: 'Super Resolution',
    speed: 3,
    quality: 4,
    complexity: 'Medium',
    processingTime: 3.2,
    tags: ['deep', 'accurate'],
    features: ['Deep Network', 'Accurate', 'Multi-Scale'],
    parameters: {
      scale: {
        type: 'select',
        options: [
          { value: 2, label: '2x', description: 'PSNR ~37.53 dB' },
          { value: 3, label: '3x', description: 'PSNR ~33.66 dB' },
          { value: 4, label: '4x', description: 'PSNR ~31.35 dB' },
          { value: 8, label: '8x', description: 'PSNR ~25.93 dB' }
        ],
        default: 2,
        label: 'Scaling Factor'
      }
    }
  },
  {
    id: 'direct',
    name: 'Direct Upscaling',
    description: 'Simple bicubic interpolation upscaling',
    englishDescription: 'Simple bicubic interpolation upscaling. This is a traditional computer vision approach that uses mathematical interpolation to increase image size. Fast but limited in quality enhancement.',
    category: 'Super Resolution',
    speed: 5,
    quality: 2,
    complexity: 'Very Low',
    processingTime: 0.5,
    tags: ['fast', 'basic'],
    features: ['Fast', 'Simple', 'No Complexity'],
    parameters: {
      scale: {
        type: 'select',
        options: [
          { value: 2, label: '2x' },
          { value: 3, label: '3x' },
          { value: 4, label: '4x' },
          { value: 8, label: '8x' }
        ],
        default: 2,
        label: 'Scaling Factor'
      }
    }
  },
  {
    id: 'espcn',
    name: 'ESPCN',
    description: 'Efficient Sub-Pixel Convolutional Neural Network',
    englishDescription: 'Efficient Sub-Pixel Convolutional Neural Network. ESPCN uses sub-pixel convolution layers to upscale images efficiently, operating on low-resolution space for most computations and performing upscaling at the final layer.',
    category: 'Super Resolution',
    speed: 4,
    quality: 3,
    complexity: 'Low',
    processingTime: 1.0,
    tags: ['super-resolution', 'neural-network', 'efficient'],
    features: ['Fast Processing', 'Sub-pixel Convolution', 'Memory Efficient']
  },
  {
    id: 'fariba',
    name: 'Classification and Compression by Learning',
    description: 'Advanced image compression with joint classification',
    englishDescription: 'Advanced image compression with joint classification. Fariba is a neural compression algorithm that simultaneously compresses images and performs classification, optimizing for both compression efficiency and classification accuracy. Supports 5 quality levels (1-5) where 1 is highest compression and 5 is highest quality.',
    category: 'Compression',
    speed: 3,
    quality: 4,
    complexity: 'High',
    processingTime: 4.5,
    tags: ['compression', 'classification', 'advanced'],
    features: ['Advanced Compression', 'Joint Classification', 'Adjustable Quality'],
    supportedQualities: [1, 2, 3, 4, 5],
    useQualityInsteadOfScale: true,
    qualityRange: { min: 1, max: 5 }
  }
];

export default ALGORITHMS;
