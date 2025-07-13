# Sambuca Core

A Python package for semi-analytical bio-optical modeling and inversion of aquatic remote sensing data.

## Overview

Sambuca Core provides tools for:
- Forward modeling of aquatic reflectance spectra
- Management of Spectral Inherent Optical Properties (SIOPs)
- Lookup table generation for fast inversion
- Image-based parameter retrieval from remote sensing data

## Installation

```bash
git clone https://github.com/lmschwenger/sambuca_core.git
cd sambuca_core
pip install .
```

## Quick Start

### 1. Basic Forward Modeling

```python
from sambuca.core.forward_model import forward_model

# Define parameters
wavelengths = [400, 450, 500, 550, 600, 650, 700]
a_water = [0.00635, 0.0145, 0.0596, 0.2885, 0.5318, 1.011, 1.692]
a_ph_star = [0.0624, 0.0436, 0.0201, 0.0123, 0.0082, 0.0065, 0.0058]
substrate1 = [0.15, 0.18, 0.22, 0.25, 0.28, 0.30, 0.32]

# Run forward model
result = forward_model(
    chl=1.0,              # Chlorophyll concentration (mg/m³)
    cdom=0.1,             # CDOM absorption (1/m)
    nap=0.5,              # NAP concentration (mg/L)
    depth=10.0,           # Water depth (m)
    substrate1=substrate1,
    wavelengths=wavelengths,
    a_water=a_water,
    a_ph_star=a_ph_star,
    num_bands=len(wavelengths)
)

print(f"Modeled reflectance: {result.rrs}")
```

### 2. SIOP Management

```python
from sambuca.core.siop_manager import SIOPManager

# Load SIOP data from directory
siop_manager = SIOPManager('path/to/siop/data')

# Get SIOPs for specific wavelengths
wavelengths = [400, 500, 600]
siops = siop_manager.get_siops_for_wavelengths(wavelengths)

print(f"Available libraries: {siop_manager.list_available_libraries()}")
```

### 3. Lookup Table Generation

```python
from sambuca.core.lookup_table import LookUpTable

# Define parameter ranges for lookup table
parameter_ranges = {
    'chl': (0.1, 5.0),
    'cdom': (0.01, 1.0),
}

fixed_parameters = {
    'nap': 0.1,
    'depth': 10.0,
    'substrate_fraction': 1.0
}

# Create and build lookup table
lut = LookUpTable(
    siop_manager=siop_manager,
    wavelengths=wavelengths,
    parameter_ranges=parameter_ranges,
    fixed_parameters=fixed_parameters
)

lut.build_table(grid_size=20)
lut.save('my_lookup_table.pkl')
```

### 4. Image Inversion

```python
from sambuca.core.inversion_handler import InversionHandler
import numpy as np

# Load lookup table
lut = LookUpTable.load('my_lookup_table.pkl', siop_manager)

# Create test image (height x width x bands)
test_image = np.random.rand(100, 100, 3) * 0.1

# Invert image
inverter = InversionHandler()
result = inverter.invert_image_from_lookup_table(
    lookup_table=lut,
    observed_image=test_image,
    metric='euclidean'
)

# Access results
chl_map = result.parameters['chl']
cdom_map = result.parameters['cdom']
error_map = result.errors

print(f"Processed {result.metadata['n_valid_pixels']} pixels")
```

## Features

### Forward Model
- Semi-analytical Lee/Sambuca bio-optical model
- Supports multiple substrates
- Configurable optical parameters
- Comprehensive output (reflectance, absorption, backscatter, etc.)

### SIOP Manager
- Load spectral libraries from CSV files
- Automatic wavelength interpolation
- Support for multi-column data formats
- Lightweight implementation without pandas dependency

### Lookup Tables
- Fast parameter space sampling
- Compressed storage options
- Memory-efficient processing
- Flexible parameter configuration

### Inversion Handler
- Image-based parameter retrieval
- Multiple distance metrics (Euclidean, RMSE, SAM)
- KD-tree acceleration for large datasets
- Chunked processing for memory efficiency
- Masking support for land/water discrimination

## Requirements

- Python ≥ 3.8
- numpy ≥ 1.20
- scipy ≥ 1.6
- tqdm ≥ 4.0

## Development

```bash
# Clone repository
git clone https://github.com/yourusername/sambuca_core.git
cd sambuca_core

# Install in development mode
pip install -e ".[dev]"

# Run tests
python -m pytest tests/
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.