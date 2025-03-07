# Grain Shape Analysis Notebook

**Implementation of the Manuscript:**

This notebook corresponds to the implementation of the manuscript titled **"Using a Smartphone Device to Quantify Particle Size and Shape Descriptors"** by **Daniella Escribano**, **Carlos Kuncar Medina**, and **Gonzalo Montalva Alvarado** (November 2024). As authors, we plan to post the code on GitHub after the review process.

---

## Author Information

**Daniella Escribano, PhD**

- Department of Civil Engineering, Universidad de Concepción, Chile
- Fondecyt Iniciación Nº11241067, ANID, Chile
- Email: describano@udec.cl
- ORCID: 0000-0003-2014-9008

**Carlos Kuncar Medina, MSc. Candidate**

- Department of Civil Engineering, Universidad de Concepción, Chile
- Email: camedina2017@udec.cl

**Gonzalo Montalva Alvarado, PhD**

- Department of Civil Engineering, Universidad de Concepción, Chile
- EASER Project ACT240044, ANID, Chile
- Email: gmontalva@udec.cl
- ORCID: 0000-0001-8598-7120

---

## Abstract Summary

This notebook provides a procedure to obtain particle size and shape descriptors through a 2D image processing algorithm using a smartphone device. The open-source tool is automatic and requires only a simple initial calibration of the camera lens. The output consists of a spreadsheet with particle size and shape descriptors for each grain, as well as their average values for the entire sample. The tool benefits the geotechnical community due to the strong link between particle morphology and mechanical properties. An application is included where the script is used to obtain particle shape descriptors of a sand with known critical state parameters, evaluating them through particle shape predictive models. The results indicate that analyzing images with a minimum of 30 particles provides a good match with experimental data, highlighting the advantages of the tool as a first estimate and as a complement to a full experimental program.

---
## 11. Explanation of Excel Results

The Excel file generated from the grain shape analysis contains several columns with detailed information for each grain. Below is an explanation of each column:

### **Identification and Images**

- **`grain_id`**: Unique identifier assigned to each grain analyzed.
- **`grain_results`**: Image of the grain with circles overlaid, showing the results of the shape analysis (e.g., inscribed circle, circumscribed circle).
- **`grain_rs_PCD`**: Resized binary image of the grain adjusted to the Particle Characteristic Dimension (PCD), used for consistent scale in analysis.

### **Size Measurements** *(in pixel units)*

The following columns represent size measurements and are **in pixel units**:

- **`dins`**: Diameter of the largest inscribed circle within the grain.
- **`dcir`**: Diameter of the smallest circumscribed circle that encloses the grain.
- **`minf`**: Minimum Feret diameter (smallest distance between two parallel tangents on opposite sides of the grain).
- **`maxf`**: Maximum Feret diameter (largest distance between two parallel tangents on opposite sides of the grain).
- **`d1`**: Length of the major axis of the grain (maximum dimension).
- **`d2`**: Length of the minor axis of the grain (minimum dimension).

### **Shape Descriptors**

- **`de`**: Equivalent diameter.
- **`Roundness`**: Measure of the grain's roundness based on the curvature of its edges, following the method by Zheng and Hryciw (2015).
- **`AR`**: Aspect Ratio, calculated as the ratio of the major axis length to the minor axis length (\( AR = \frac{d1}{d2} \)).
- **`Cx`**: Convexity.
- **`Sa`**: Area Sphericity.
- **`Sc`**: Circular ratio sphericity.
- **`Sd`**: Diameter-Sphericity.
- **`Sp`**: Perimeter Sphericity based.
- **`Swl`**: Width to Length ratio Shpericity.

### **Notes on Units**

- The size measurements (**`dins`**, **`dcir`**, **`minf`**, **`maxf`**, **`d1`**, **`d2`**) are in **pixel units**. These values are derived from the images and represent relative sizes based on the image resolution.
- The shape descriptors are **dimensionless** quantities that describe the grain's shape characteristics independent of size.

### **Understanding the Measurements**

- **Size Measurements**: Useful for understanding the physical dimensions and size distribution of the grains within the sample. While provided in pixel units, these can be converted to real-world units (e.g., millimeters) if the scale of the images is known.
- **Shape Descriptors**: Provide insights into the morphological characteristics of the grains, which can influence the material's mechanical properties, such as strength and permeability.

### **Reference**

The methods and calculations for the roundness and sphericity parameters are based on:

- **Zheng, J., & Hryciw, R. D. (2015).** Traditional soil particle sphericity, roundness, and surface roughness by computational geometry. *Géotechnique*, 65(6), 494-506.

---

**Tip**: To convert pixel units to actual measurements, you need to know the scale of your images (e.g., pixels per millimeter). Once the scale is determined through calibration, you can multiply the pixel measurements by the scale factor to obtain real-world dimensions.

---

**Understanding these parameters will help in assessing the grain morphology and predicting the behavior of granular materials in geotechnical applications.**


<a target="_blank" href="https://colab.research.google.com/github/camedinak24/fastGRAINS25/blob/main/FastGRAINS.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
