# fastGRAINS: Grain Shape Analysis Tool

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camedinak24/fastGRAINS25/blob/main/FastGRAINS.ipynb)

## Overview

This repository contains the implementation of the manuscript titled **"Using a Smartphone Device to Quantify Particle Size and Shape Descriptors"** (November 2024). fastGRAINS provides an open-source, automatic procedure to obtain particle size and shape descriptors through 2D image processing using a smartphone device.

## Key Features

- Automatic grain shape analysis using smartphone images
- Requires only a simple initial calibration of the camera lens
- Generates spreadsheets with detailed particle size and shape descriptors
- Supports geotechnical applications through morphological characterization
- Validated with known critical state parameter samples

## Implementation

The code was developed by **Carlos Kuncar Medina** as part of his MSc research at Universidad de Concepción, Chile. The research was conducted under the supervision of Dr. Daniella Escribano and Dr. Gonzalo Montalva Alvarado.

## Authors

**Daniella Escribano, PhD**
- Department of Civil Engineering, Universidad de Concepción, Chile
- Fondecyt Iniciación Nº11241067, ANID, Chile
- Email: describano@udec.cl
- ORCID: 0000-0003-2014-9008

**Carlos Kuncar Medina, MSc. Candidate**
- Department of Civil Engineering, Universidad de Concepción, Chile
- Email: camedina2017@udec.cl
- *Lead developer of the fastGRAINS codebase*

**Gonzalo Montalva Alvarado, PhD**
- Department of Civil Engineering, Universidad de Concepción, Chile
- EASER Project ACT240044, ANID, Chile
- Email: gmontalva@udec.cl
- ORCID: 0000-0001-8598-7120

## Usage

The repository includes a Jupyter notebook that can be run locally or in Google Colab. The notebook guides users through the process of:

1. Loading and preprocessing grain images
2. Calibrating the analysis based on camera parameters
3. Extracting size and shape descriptors for each grain
4. Generating comprehensive result reports

## Output Description

The analysis produces an Excel file with the following parameters for each grain:

### Identification and Images
- Unique grain identifiers
- Visual representations of analysis results

### Size Measurements (in pixel units)
- Diameter of largest inscribed circle
- Diameter of smallest circumscribed circle
- Minimum and maximum Feret diameters
- Major and minor axis lengths

### Shape Descriptors
- Equivalent diameter
- Roundness (Zheng & Hryciw, 2015)
- Aspect Ratio
- Convexity
- Various sphericity measures (Area, Circular ratio, Diameter, Perimeter, Width-to-Length ratio)

## Applications

The tool benefits the geotechnical community by:
- Providing rapid assessment of grain morphology
- Establishing connections between particle shape and mechanical properties
- Complementing traditional experimental programs
- Offering reliable estimates with as few as 30 particles per sample

## Citation

If you use this tool in your research, please cite:
```
Escribano, D., Kuncar Medina, C., & Montalva Alvarado, G. (2024). Using a Smartphone Device to Quantify Particle Size and Shape Descriptors. [Journal information pending]
```

## License

[Include license information here]

## Acknowledgments

This research was supported by Fondecyt Iniciación Nº11241067 and EASER Project ACT240044, ANID, Chile.
