# LungConVT: Hybrid CNN-Transformer for Lung Disease Classification

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.10%2B-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-Pattern%20Recognition-red)](https://doi.org/10.1016/j.patcog.2024.XXXXX)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.patcog.2024.XXXXX-blue)](https://doi.org/10.1016/j.patcog.2024.XXXXX)

## 🔬 Overview

LungConVT is a state-of-the-art deep learning architecture that combines Convolutional Neural Networks (CNNs) with Vision Transformers for accurate classification of lung diseases from chest X-ray images. The model achieves **95.2% accuracy** in distinguishing between COVID-19, Normal, Bacterial Pneumonia, and Viral Pneumonia cases.

<p align="center">
  <img src="assets/model_architecture.png" alt="LungConVT Architecture" width="800"/>
</p>

### ✨ Key Features

- 🏗️ **Hybrid Architecture**: Seamlessly integrates CNN and Transformer blocks
- 🎯 **High Accuracy**: 95.2% accuracy on multi-class lung disease classification
- 🔍 **Explainable AI**: Integrated Grad-CAM for visual explanations
- 📊 **Comprehensive Evaluation**: ROC curves, confusion matrices, and statistical tests
- 🔄 **Reproducible**: Fixed seeds and detailed environment specifications
- 📦 **Production Ready**: Modular code structure with CLI interface

## 📈 Results

<table>
<tr>
<td>

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| COVID-19 | 0.961 | 0.943 | 0.952 |
| Normal | 0.948 | 0.972 | 0.960 |
| Bacterial Pneumonia | 0.937 | 0.925 | 0.931 |
| Viral Pneumonia | 0.945 | 0.958 | 0.951 |
| **Overall** | **0.948** | **0.950** | **0.949** |

</td>
<td>
<img src="assets/confusion_matrix.png" alt="Confusion Matrix" width="400"/>
</td>
</tr>
</table>

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 16GB RAM minimum

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/lungconvt.git
cd lungconvt
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python validate_setup.py
```

### 🏃‍♂️ Training the Model

```bash
# For headless environments (servers, docker)
./run_headless.sh --mode train --config config.yaml

# Or directly
python main.py --mode train --config config.yaml
```

### 📊 Evaluating the Model

```bash
python main.py --mode evaluate --model_path path/to/model.h5 --config config.yaml
```

### 🔍 Generating Explanations

```bash
python main.py --mode explain --model_path path/to/model.h5 --image_path path/to/image.jpg
```

## 📁 Project Structure

```
lungconvt/
├── 📄 data_loader.py       # Data loading and preprocessing
├── 🧠 model.py            # LungConVT architecture
├── 🏋️ train.py            # Training pipeline
├── 📊 evaluate.py         # Model evaluation
├── 🔍 explain.py          # Grad-CAM explanations
├── 🛠️ utils.py            # Helper functions
├── 🎯 main.py             # CLI interface
├── 📓 demo.ipynb          # Interactive demo
├── ⚙️ config.yaml         # Configuration
└── 📋 requirements.txt    # Dependencies
```

## 🏗️ Model Architecture

LungConVT features a hierarchical architecture with four main components:

1. **Initial Feature Extraction**: Convolutional blocks for low-level features
2. **Depthwise Separable Convolutions**: Efficient feature extraction
3. **Dual-Head Convolutional Multi-Head Attention (DHC-MHA)**: Local-global feature integration
4. **Adaptive Multi-Grained Multi-Head Attention (AMG-MHA)**: Multi-scale feature analysis

<details>
<summary>📐 Architecture Details</summary>

```python
Input (256×256×3)
    ↓
Conv Block (32 filters)
    ↓
DC Layers (32→64→128)
    ↓
DHC-MHA (heads=4)
    ↓
DC Layer (128)
    ↓
DHC-MHA (heads=8)
    ↓
Reshape & AMG-MHA
    ↓
Global Average Pooling
    ↓
Dense (Softmax, 4 classes)
```

</details>

## 📊 Dataset

The model is trained on a curated dataset of chest X-ray images:

- **Total Images**: 8,000
- **Classes**: 4 (COVID-19, Normal, Bacterial Pneumonia, Viral Pneumonia)
- **Split**: 80% training, 20% testing
- **Augmentation**: Rotation, zoom, shift, shear

### Dataset Structure
```
data/
├── DB_modified_ImgDB.csv
└── images/
    ├── COVID-19/
    ├── Normal/
    ├── Pneumonia-Bacterial/
    └── Pneumonia-Viral/
```

## 🔬 Reproducibility

We ensure full reproducibility through:

- 🎲 Fixed random seeds (NumPy, TensorFlow, Python)
- 📦 Exact package versions in `requirements.txt`
- 🖥️ Hardware specifications documented
- 📝 Detailed training logs saved automatically

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for detailed instructions.

## 📈 Performance Metrics

<p align="center">
  <img src="assets/roc_curves.png" alt="ROC Curves" width="600"/>
</p>

- **Overall Accuracy**: 95.2%
- **Macro F1-Score**: 0.948
- **Average AUC**: 0.987

## 🔍 Explainability

LungConVT includes Grad-CAM visualizations to understand model decisions:

<p align="center">
  <img src="assets/gradcam_examples.png" alt="Grad-CAM Examples" width="800"/>
</p>

## 💻 Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- GPU: 4GB VRAM (optional)

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA RTX 2070+ (8GB VRAM)

## 📝 Configuration

Edit `config.yaml` to customize:

```yaml
model:
  input_size: [256, 256]
  
training:
  batch_size: 16
  epochs: 200
  learning_rate: 0.001
  
data:
  test_size: 0.2
  augmentation:
    rotation_range: 5
    zoom_range: 0.15
```

## 🐛 Troubleshooting

<details>
<summary>Common Issues & Solutions</summary>

### Display/Qt Errors
```bash
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
```

### Out of Memory
- Reduce `batch_size` in config.yaml
- Enable mixed precision training

### Installation Issues
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

</details>

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{lungconvt2024,
  title={LungConVT: A Hybrid CNN-Transformer Architecture for Lung Disease Classification},
  author={[Your Name]},
  journal={Pattern Recognition},
  year={2024},
  doi={10.1016/j.patcog.2024.XXXXX}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Medical imaging community for datasets
- TensorFlow and Keras teams
- Vision Transformer and ConvNeXt papers for inspiration

## 📧 Contact

- **Author**: [Your Name]
- **Email**: [your.email@institution.edu]
- **Lab**: [Your Lab Name]
- **Institution**: [Your Institution]

---

<p align="center">
  Made with ❤️ for the medical AI community
</p>

<p align="center">
  <a href="https://github.com/yourusername/lungconvt/stargazers">⭐ Star us on GitHub!</a>
</p>