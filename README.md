# Underwater Image Enhancement Using Adaptive Color Correction and Improved Retinex Algorithm

![Underwater Enhancement Banner](./sample_images/banner.jpg)

---

## 📊 Overview

This project addresses the enhancement of underwater images, which often suffer from low contrast, blurring, and severe color distortion due to light attenuation and scattering in water. The proposed solution combines **adaptive color correction**, **wavelet-based decomposition**, and an **improved Retinex algorithm**, fused using **Non-Subsampled Shearlet Transform (NSST)** to produce visually appealing and quantitatively superior results.

> This method is developed as a final-year B.Tech project under the guidance of Mrs. Y. Sravani, Assistant Professor, Department of ECE, SVCE.

---

## 🔖 Key Features

- ✅ Adaptive color correction for natural color tone balancing
- ✅ Retinex enhancement for edge and contrast preservation
- ✅ Wavelet transform for multi-scale detail extraction
- ✅ NSST-based image fusion for superior image quality
- ✅ Quantitative evaluation using PCQI, UIQM, UCIQE, and Entropy

---

## 📚 Abstract

Underwater images suffer from degradation due to light absorption and scattering in water. This project proposes a novel fusion-based algorithm combining **adaptive color correction** and **improved Retinex processing**, followed by fusion using **NSST**. The technique enhances both the details and color fidelity of the input image, yielding better results in terms of both visual quality and statistical image quality metrics.

---

## 🔹 Table of Contents

1. [Demo & Results](#-demo--results)
2. [Methodology](#-methodology)
3. [Software Setup](#-software-setup)
4. [Project Structure](#-project-structure)
5. [Evaluation Metrics](#-evaluation-metrics)
6. [Documentation](#-documentation)
7. [Authors & Mentors](#-authors--mentors)
8. [Future Scope](#-future-scope)
9. [References](#-references)

---

## 🚀 Demo & Results

<table>
<tr>
<td><img src="./sample_images/original.jpg" width="300"></td>
<td><img src="./sample_images/enhanced.jpg" width="300"></td>
</tr>
<tr>
<td align="center">Original Underwater Image</td>
<td align="center">Enhanced Image using our Method</td>
</tr>
</table>

---

## 💡 Methodology

### 1. Adaptive Color Correction
- Balances red, green, and blue channels using underwater image statistics
- Dynamically adjusts color cast based on ambient light conditions

### 2. Wavelet Decomposition (DWT)
- Decomposes the image into multiple frequency bands
- Extracts high-intensity detail from high-frequency components

### 3. Improved Retinex Algorithm
- Enhances local contrast and edge information
- Separates illumination and reflectance components for better clarity

### 4. NSST Fusion
- Merges detail-enhanced and Retinex-enhanced images
- Retains both fine textures and global color consistency

---

## 🛠️ Software Setup

- **MATLAB** R2021a or later
- Toolboxes Required:
  - Image Processing Toolbox
  - Signal Processing Toolbox

#### To Run:
```matlab
input = imread('sample_images/input.jpg');
enhanced = enhanceUnderwaterImage(input);
imshow(enhanced);
```

---

## 📂 Project Structure

```
underwater-retinex-project/
├── README.md
├── code/                      # MATLAB source files
│   └── enhanceUnderwaterImage.m
├── documentation/            # PDF and PPT documents
│   ├── B12_Project_Report.pdf
│   └── viva_presentation.pptx
├── sample_images/            # Input and result images
├── results/                  # Quantitative metrics outputs
└── requirements.txt
```

---

## 📊 Evaluation Metrics

| Metric | Description | Sample Value |
|--------|-------------|--------------|
| PCQI   | Perceptual Color Quality Index | 0.92 |
| UIQM   | Underwater Image Quality Measure | 0.88 |
| UCIQE  | Underwater Color Image Quality Evaluation | 0.75 |
| IE     | Image Entropy | 7.28 |

These metrics confirm our method's superiority over existing techniques.

---

## 📄 Documentation

- [📄 Full Project Report (PDF)](./documentation/B12_Project_Report.pdf)
- [📝 Final Viva Presentation (PPTX)](./documentation/viva_presentation.pptx)

Includes:
- Introduction to Digital Image Processing
- Review of Existing Methods
- Literature Survey
- Proposed Methodology
- Results and Discussion
- Conclusion and Future Work

---

## 👩‍💼 Authors & Mentors

**Contributors**  
- S. Raiyan (22KH5A0417)  
- K. Kishore (22KH5A0410)  
- S. Reshma Begum (21KH1A0475)  
- S. Fathima Arfa (21KH1A0483)

**Supervisor**  
Mrs. Y. Sravani, M.Tech., Assistant Professor, Dept. of ECE

**Institution**  
Sri Venkateswara College of Engineering, Kadapa  
Affiliated to JNTUA, Ananthapuramu

---

## 🌐 Future Scope

- 🤖 **AI Integration**: Deep learning for end-to-end enhancement (GANs, CNNs)
- ⏱️ **Real-Time Processing**: For underwater drones or AUVs
- 🌊 **Multispectral Imaging**: Expanding beyond RGB to explore depth and clarity
- 🌿 **Marine Research**: Feature extraction for species classification and tracking

---

## 📓 References

1. Tang Zhongqiang et al., "Vision enhancement of underwater vehicle based on improved DCP algorithm"
2. He Xiao et al., "Underwater image enhancement using Retinex and Gamma correction"
3. Yang Fuhao et al., "Color compensation and Retinex for image clarity"
4. Li Shelei et al., "Dark primary color prior model for underwater image correction"
5. [Dataset and baseline](https://github.com/lin9393/underwater-image-enhance)

---

> © 2025 Sri Venkateswara College of Engineering. For academic use only.
