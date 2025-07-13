# FETCH ONNX Edition

[![DOI](https://zenodo.org/badge/165734093.svg?style=flat-square)](https://zenodo.org/badge/latestdoi/165734093)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-green.svg)](https://onnxruntime.ai/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**FETCH (Fast Extragalactic Transient Candidate Hunter) - ONNX Edition** provides a lightweight, TensorFlow-free ONNX inference solution for fast radio burst (FRB) detection and classification.

This ONNX edition offers significant advantages over the original TensorFlow implementation:
- **Zero TensorFlow dependencies** - Pure ONNX Runtime for ultra-fast inference
- **Cross-platform compatibility** - Works on any ONNX-supported platform
- **GPU/CPU flexibility** - Automatic hardware detection with seamless fallback

## 🚀 Quick Start

### Model Storage Configuration

FETCH uses the `ONNX_HOME` environment variable to determine where to store downloaded ONNX models:

```bash
# Set custom model directory (optional)
export ONNX_HOME=/path/to/your/models

# If not set, models are stored in $HOME/onnx_models by default
# Example: /home/user/onnx_models/
```

The first time you run inference, models will be automatically downloaded from Zenodo and cached locally for future use.

### Installation

#### Option 1: Using uv (Recommended)
```bash
# Install with CPU-only inference
uv pip install "fetch[cpu] @ git+https://github.com/devanshkv/fetch.git@onnx"

# Install with GPU acceleration
uv pip install "fetch[gpu] @ git+https://github.com/devanshkv/fetch.git@onnx"
```

#### Option 2: Traditional pip
```bash
# Clone repository
git clone https://github.com/devanshkv/fetch.git
cd fetch
git checkout onnx

# CPU-only inference
pip install -r requirements_onnx.txt
pip install .

# GPU-accelerated inference  
pip install -r requirements_onnx_gpu.txt
pip install .
```

#### Option 3: Development Installation
```bash
git clone https://github.com/devanshkv/fetch.git
cd fetch
git checkout onnx

# Install in development mode
pip install -e .
```

## 📖 Usage

### Command Line Interface

The primary interface is through the `fetch-predict` command installed globally:

```bash
fetch-predict -m <model_id> -c <data_directory>
```

Alternatively, use the script directly:
```bash
python src/fetch/predict_onnx.py -m <model_id> -c <data_directory>
```

### Basic Examples

```bash
# CPU inference with model 'a' (models auto-downloaded to $ONNX_HOME)
fetch-predict -m a -c /path/to/candidates -g -1

# GPU inference (automatically detects available GPU)
fetch-predict -m a -c /path/to/candidates

# Specific GPU with custom parameters
fetch-predict -m a -c /path/to/candidates -g 1 -b 16 -p 0.7

# High-throughput batch processing with custom model directory
fetch-predict -m j -c /path/to/large_dataset -b 32 --onnx_dir ./models

# Set custom model storage location
export ONNX_HOME=/data/models
fetch-predict -m a -c /path/to/candidates
```

### Advanced Usage

```bash
# Multiple model ensemble prediction
for model in {a..k}; do
    fetch-predict -m $model -c /data/candidates -o results_$model.csv
done

# Custom threshold tuning
fetch-predict -m a -c /data/test_set -p 0.3  # High sensitivity
fetch-predict -m a -c /data/prod_set -p 0.8  # High precision
```

### Command Line Arguments

| Argument | Short | Description | Default | Examples |
|----------|-------|-------------|---------|----------|
| `--model` | `-m` | Model identifier (a-k) | Required | `a`, `j`, `k` |
| `--data_dir` | `-c` | Directory containing H5 candidate files | Required | `/data/candidates/` |
| `--gpu_id` | `-g` | GPU device ID (-1 for CPU) | 0 | `-1`, `0`, `1` |
| `--batch_size` | `-b` | Inference batch size | 8 | `16`, `32`, `64` |
| `--probability` | `-p` | Classification threshold | 0.5 | `0.3`, `0.7`, `0.9` |
| `--onnx_dir` | | Model storage directory | `$ONNX_HOME` or `$HOME/onnx_models` | `./models/` |
| `--output` | `-o` | Output CSV filename | `results_{model}.csv` | `predictions.csv` |

## 📊 Input/Output Format

### Input Requirements

Candidate files must be HDF5 format with the following structure:
```
candidate.h5
├── data_freq_time    # 2D array: frequency vs time
├── data_dm_time      # 2D array: DM vs time  
└── metadata          # Optional: SNR, DM, width, etc.
```

**Expected dimensions**: 
- Frequency-time: `(256, 256)` or `(512, 512)`
- DM-time: `(256, 256)` or `(512, 512)`

### Output Format

Results are saved as CSV with the following columns:

```csv
candidate,probability,label
cand_001.h5,0.9234,1.0
cand_002.h5,0.1456,0.0
cand_003.h5,0.8901,1.0
```

- **candidate**: Input filename
- **probability**: Model confidence score (0.0-1.0)
- **label**: Binary classification (>= threshold = 1.0, < threshold = 0.0)

## 🛠️ Development & Building

### Building from Source

```bash
# Clone and setup
git clone https://github.com/devanshkv/fetch.git
cd fetch
git checkout onnx

# Install build dependencies
pip install uv

# Build package
uv build

# Install locally
pip install dist/fetch-*.whl
```

### Development Setup

```bash
# Install development dependencies
pip install -r requirements_onnx.txt
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
black .
ruff check .

# Type checking
mypy fetch/
```

### Testing ONNX Models

```bash
# Test all models
python bin/test_onnx_models.py

# Test specific model
python bin/test_onnx_models.py --model a

# Benchmark performance
python bin/test_onnx_models.py --benchmark
```

## 🎓 Scientific Background

FETCH implements deep learning models for Fast Radio Burst (FRB) detection based on the methodology described in:

> **"FETCH: A deep-learning based classifier for fast transient classification"**  
> Agarwal et al., Monthly Notices of the Royal Astronomical Society, 2020  
> DOI: [10.1093/mnras/staa1856](https://doi.org/10.1093/mnras/staa1856)

## 📄 License & Citation

### License
This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

### Citing FETCH

If you use FETCH in your research, please cite:

```bibtex
@article{Agarwal2020,
  doi = {10.1093/mnras/staa1856},
  url = {https://doi.org/10.1093/mnras/staa1856},
  year = {2020},
  month = jun,
  publisher = {Oxford University Press ({OUP})},
  volume = {497},
  number = {2}, 
  pages = {1661--1674},
  author = {Devansh Agarwal and Kshitij Aggarwal and Sarah Burke-Spolaor and Duncan R Lorimer and Nathaniel Garver-Daniels},
  title = {{FETCH}: A deep-learning based classifier for fast transient classification},
  journal = {Monthly Notices of the Royal Astronomical Society}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution

- **Performance optimization**: ONNX model quantization, pruning
- **Hardware support**: Apple Silicon, Intel GPU, AMD ROCm
- **Data formats**: Support for additional input formats
- **Visualization**: Real-time monitoring dashboards
- **Integration**: Pulsar search pipeline connectors

## 📞 Support & Community

- **Issues**: [GitHub Issues](https://github.com/devanshkv/fetch/issues)
- **Discussions**: [GitHub Discussions](https://github.com/devanshkv/fetch/discussions)
- **Email**: [devansh.kv@gmail.com](mailto:devansh.kv@gmail.com)

---

**FETCH ONNX Edition** - Bringing state-of-the-art FRB detection to the masses with unprecedented efficiency and ease of use.