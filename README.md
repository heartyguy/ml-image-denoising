# ML Image Denoising

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Theano](https://img.shields.io/badge/Theano-0.7.0-orange.svg)](http://deeplearning.net/software/theano/)

A comprehensive suite of image denoising techniques, from classical statistical methods to state-of-the-art deep learning models. This repository provides implementations and comparisons of Principal Component Analysis (PCA), Non-negative Matrix Factorization (NMF), Singular Value Decomposition (SVD), Spectral Decomposition, Convolutional Neural Networks (CNN), and a Generative Adversarial Denoising Autoencoder (GAN-DAE).

![image](https://github.com/user-attachments/assets/b751f5d2-278b-4878-b98f-550bd250466f) -->
![image](https://github.com/user-attachments/assets/6561cabe-21a2-43dc-85bc-7c861cce56f1)


## 🎯 Overview

Image denoising is a fundamental problem in computer vision and image processing. This project serves as both a practical toolkit and an educational resource for understanding and comparing various denoising methodologies across traditional and modern approaches.

### Key Features

- **Classical Methods**: Implementation of foundational techniques including PCA, NMF, SVD, and Spectral Decomposition
- **Deep Learning**: Custom Convolutional Neural Network designed for effective noise reduction
- **State-of-the-Art**: Sophisticated Generative Adversarial Network-based Denoising Autoencoder for high-fidelity reconstruction
- **Comprehensive Evaluation**: Built-in PSNR metrics and visual comparison tools
- **Modular Design**: Extensible codebase for easy experimentation with new techniques
- **Interactive Notebooks**: Jupyter notebooks for each method with detailed explanations

## 🚀 Quick Start

### Prerequisites

- Python 3.6+
- Git
- pip
- virtualenv

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ml-image-denoising.git
   cd ml-image-denoising
   ```

2. **Set up environment**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```
   This creates a virtual environment and installs all necessary dependencies.

3. **Activate the virtual environment**
   ```bash
   source bin/activate  # On Windows: Scripts\activate
   ```

### Dataset Preparation

This project uses the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html):

1. **Download the dataset**
   - Download `img_align_celeba.zip` from the CelebA project website
   - Extract to the parent directory: `../img_align_celeba/`

2. **Process the data**
   ```bash
   python dataprocessing.py
   ```
   This script crops images to the required size and converts them to efficient HDF5 format.

## 📊 Usage

### Jupyter Notebooks

Launch Jupyter to explore the different denoising methods:

```bash
jupyter notebook
```

#### Available Notebooks

- **`adversarial_autoencoder_denoising.ipynb`** - GAN-based denoising autoencoder
- **`cnn_denoising.ipynb`** - Convolutional Neural Network approach
- **`pca_denoising.ipynb`** - Principal Component Analysis method
- **`nmf_denoising.ipynb`** - Non-negative Matrix Factorization technique

### Example Usage

```python
# Load and preprocess data
from helper import cmp_images_psnr
import numpy as np

# Compare denoising methods
psnr_score = cmp_images_psnr(clean_images, denoised_images, num_samples)
print(f"PSNR: {psnr_score:.2f} dB")
```

## 🏗️ Project Structure

```
ml-image-denoising/
├── adversarial_autoencoder_denoising.ipynb  # GAN-DAE implementation
├── cnn_denoising.ipynb                      # CNN-based denoising
├── pca_denoising.ipynb                      # PCA method
├── nmf_denoising.ipynb                      # NMF technique
├── dataprocessing.py                        # Data preprocessing script
├── helper.py                                # Utility functions (PSNR, etc.)
├── install.sh                               # Environment setup script
├── requirements.txt                         # Python dependencies
└── lib/                                     # Core library modules
    ├── activations.py                       # Activation functions
    ├── updates.py                           # Optimization algorithms
    ├── ops.py                               # Neural network operations
    ├── metrics.py                           # Evaluation metrics
    └── ...
```

## 🔬 Methods Implemented

### Classical Approaches

| Method | Description | Notebook |
|--------|-------------|----------|
| **PCA** | Dimensionality reduction via principal components | `pca_denoising.ipynb` |
| **NMF** | Non-negative matrix factorization | `nmf_denoising.ipynb` |
| **SVD** | Singular value decomposition | `pca_denoising.ipynb` |
| **Interpolation** | Bilinear and bicubic upsampling | `pca_denoising.ipynb` |

### Deep Learning Approaches

| Method | Description | Notebook |
|--------|-------------|----------|
| **CNN** | Custom convolutional neural network | `cnn_denoising.ipynb` |
| **GAN-DAE** | Generative adversarial denoising autoencoder | `adversarial_autoencoder_denoising.ipynb` |

## 📈 Performance

Typical PSNR improvements on CelebA dataset:

- **Bilinear Interpolation**: ~25.5 dB
- **Bicubic Interpolation**: ~26.2 dB
- **CNN Denoising**: ~28-30 dB
- **GAN-DAE**: ~30-35 dB

*Results may vary based on noise levels and training parameters.*

## 🛠️ Configuration

### Training Parameters

Key parameters can be modified in the notebook cells:

```python
# Training configuration
batch_size = 128
learning_rate = 0.0002
num_epochs = 200
l2_regularization = 1e-5
```

### Model Architecture

The GAN-DAE uses an encoder-decoder architecture:
- **Encoder**: Progressively downsamples from 128×128 to 8×8
- **Decoder**: Reconstructs high-resolution output
- **Discriminator**: Ensures realistic image generation

## 📋 Requirements

### Core Dependencies

- `Theano==0.7.0`
- `Pillow>=3.0.0`
- `numpy>=1.10.2`
- `h5py>=2.5.0`
- `tqdm>=3.4.0`
- `ipython>=4.0.0`

See `requirements.txt` for complete dependency list.

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Areas for Contribution

- Implementation of additional denoising algorithms
- Performance optimizations
- Better evaluation metrics
- Documentation improvements
- Bug fixes and code quality improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) for providing the facial image dataset
- [Theano](http://deeplearning.net/software/theano/) deep learning framework
- Original DCGAN implementation inspiration
- The open-source community for various utility functions

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- **Issues**: [GitHub Issues](https://github.com/your-username/ml-image-denoising/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/ml-image-denoising/discussions)

## 🔗 Related Work

- [Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior)
- [DnCNN: Beyond a Gaussian Denoiser](https://github.com/cszn/DnCNN)
- [Noise2Noise: Learning Image Restoration without Clean Data](https://github.com/NVlabs/noise2noise)

---

⭐ **Star this repository if you find it helpful!**
