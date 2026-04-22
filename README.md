# Spectrum Reconstruction via Tikhonov Regularization and GCV

This repository contains a MATLAB implementation for reconstructing continuous optical spectra from measured photocurrents and a pre-calibrated sensor responsivity matrix. The algorithm dynamically adapts to different light sources (e.g., monochromatic lasers, LEDs, broadband lamps) using an optimized dictionary of Gaussian basis functions.

This code is adapted from the methodology described by **Yoon et al.** in their *Science* publication: [https://doi.org/10.1126/science.add8544](https://doi.org/10.1126/science.add8544).

## 📌 Overview

Reconstructing a high-resolution spectrum from a limited number of sensor channels is an ill-posed inverse problem. This script solves this by employing **Tikhonov Regularization** combined with **Generalized Cross-Validation (GCV)** to automatically find the optimal balance between data fidelity (matching the measured currents) and solution smoothness (preventing noise-induced overfitting).

### Mathematical Formulation
The core optimization problem is formulated as a Non-Negative Least Squares (NNLS) minimization:

$$\min_{x \ge 0} \left( \| Ax - b \|_2^2 + \gamma^2 \| Lx \|_2^2 \right)$$

Where:
* $A$ is the forward matrix (responsivity matrix mapped to Gaussian basis functions).
* $x$ is the vector of unknown spectral coefficients to be solved.
* $b$ is the measured photocurrent vector.
* $L$ is the second-difference (Laplacian) operator matrix to penalize roughness.
* $\gamma$ is the regularization parameter, optimized automatically via GCV.

## ✨ Key Features

* **Dynamic Source Classification:** The algorithm performs a preliminary diagnostic pass to estimate the standard deviation of the light source. It then automatically classifies the source as *Monochromatic*, *LED / Narrow Broad*, or *True Broadband*.
* **Adaptive Gaussian Dictionaries:** Based on the classification, the script selects the optimal Full Width at Half Maximum (FWHM) pool for its Gaussian basis functions, ensuring precise reconstruction for any type of light.
* **Automated GCV Optimization:** Eliminates the need to manually guess the smoothing parameter ($\gamma$) by calculating the Generalized Cross-Validation score across a wide parameter space.
* **High-Resolution Output:** Upsamples and interpolates responsivity data using piecewise cubic Hermite polynomials (`pchip`) to produce smooth, high-resolution spectral curves.

## 📂 Required Input Files

To run the script, ensure the following CSV files are in the same directory:

1.  `ResponsivityMatrix.csv`: The calibrated responsivity data of the sensor array (Rows = Sensor Channels/Voltages, Columns = Wavelengths).
2.  `MeasuredSignals.csv`: The raw photocurrent measurements (Rows = Sensor Channels, Columns = Distinct Samples).
3.  `SpectrumMatrix.csv`: (Optional for ground-truth comparison) The actual spectra measured by a reference spectrometer.

## 🚀 Usage

1. Clone the repository and place your input `.csv` data files in the working directory.
2. Open the main script in MATLAB.
3. Adjust the **User Settings** section if your wavelength bounds, step sizes, or FWHM pools differ from the default (400 nm to 900 nm).
4. Run the script.

### Outputs
Upon completion, the script generates the following files:
* `reconstructed_spectrum_matrix_hires.csv`: The final, high-resolution reconstructed spectra.
* `simulated_current_matrix.csv`: The theoretical currents calculated from the reconstructed spectra (useful for verifying data fidelity).
* `reconstruction_summary.csv`: A comprehensive metadata table detailing the classification, chosen FWHM, optimum $\gamma$, and error metrics for each sample.
* **Visualization:** A tiled figure will automatically display comparing the reconstructed spectra against the ground-truth measurements.

## 📖 Citation & Acknowledgments

If you use or build upon this code, please refer to the original mathematical framework:
> Yoon, et al. *"Miniaturized spectrometers with a tunable van der Waals junction."* Science (2022). DOI: 10.1126/science.add8544
