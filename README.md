# Separability of photon triplets generated through third-order spontaneous parametric downconversion in waveguides

This repository contains a Python class and a collection of Jupyter notebooks developed to generate the results for the paper:  ["Strategies for generating separable photon triplets in waveguides and ring resonators"](https://arxiv.org/abs/2506.15810).

## Contents

### Core Class: topdc_calc.py

A class for computing various physical quantities related to degenerate third-order parametric down-conversion (TOPDC) in optical fibers or waveguides. It uses the pump and triplet modes refractive index data to model the waveguide dispersion parameters and compute the joint spectral amplitude (JSA) of the triplets, the single photon reduced density matrix and its mode decomposition, and spectral separability.

Refer to the class and methods docstrings for detailed usage information.

### Jupyter notebooks - Numerical results

The following notebooks contain simulations for each section of the paper. They include the full code used to generate all figures and tables in the publication.

#### Section II –  Ideal Case and Separability Conditions: 

* **`II_1_test_JSA_ideal.ipynb`- JSA, reduced density matrix and mode decomposition-Ideal case:**   Simulates a scenario where all separability conditions are satisfied to calculate the idealized JSA, the reduced density matrix and its mode decomposition.  This code was used to obtain Figures 2(a)–2(d).

* **`II_2_test_JSA_ideal_optim_pump_bw.ipynb` - Separability vs. Pump Bandwidth:** 
Computes the separability parameter for an ideal waveguide as a function of the pump bandwidth. This code was used to obtain Figure 1.

* **`II_3_test_JSA_ideal_convergence.ipynb` – Convergence Test:** 
Tests simulation convergence with respect to frequency window width.

#### Section III – Sample calculations: Optimized Fiber Taper

* **`III_1_test_JSA_calc_auto.ipynb` - Scaled JSA, reduced density matrix and Mode Decomposition (Automatic):**
Computes the JSA, single-photon reduced density matrix, mode decomposition, separability parameter, and triplet generation rate for an optimized GeO2-doped silica fiber taper.
Uses the _topdc_calc_ class and its _topdc_scaled_auto_ method with units scaled by the phase-matching bandwidth.
This code was used to obtain Figures 6(a)–6(d).

* **`III_2_test_JSA_calc_manual.ipynb` - JSA, reduced density matrix and Mode Decomposition (manual):** Performs the same calculation as `III_1_test_JSA_calc_auto.ipynb`, but separates and visualizes each output manually for greater control over plotting. This code also reproduces Figures 6(a)–6(d).

* **`III_3_nonlinear_parameter.ipynb` - Nonlinear Parameter Calculation:** Computes the nonlinear parameter based on the mode field distributions of the pump and triplet modes. This code was used to generate Figure 3. 

* **`III_4_phase-matching.ipynb` - Phase-Matching Curve:** Calculates the phase-matching curve for the degenerate TOPDC process using the dispersion data from the waveguide’s pump and triplet modes.  This code was used to generate Figure 4. 

* **`III_5_test_JSA_calc_manual_unfiltered.ipynb` -  Unfiltered JSA Calculation:** Computes the unfiltered joint spectral amplitude using dispersion data from the pump and triplet modes. This code was used to generate Figures 5(a)–5(b).

* **`III_6_test_JSA_calc_auto_lambdap_sweep.ipynb` - Separability vs. Pump Wavelength:** Calculates the separability parameter as a function of the pump central wavelength for the optimized fiber taper. This code was used to generate Figure 7.

* **`III_7_test_JSA_calc_manual_pump_bw_sweep.ipynb` - Separability vs. Pump Bandwidth:** Computes the separability parameter as a function of the pump spectral bandwidth for the optimized fiber taper. 

#### Section IV – Detection

* **`IV_1_a_test_JSA_ideal_local_oscillator_optimization.ipynb`** -  Optimization of the local oscillator (LO) for homodyne detection of photon triplets in an ideal waveguide where all separability conditions are satisfied. The LO is optimized using a gradient descent method and basin-hopping algorithm with real-valued random seeds.  
  _Results summarized in Table 1 of the paper._

* **`IV_1_b_test_JSA_ideal_local_oscillator_optimization_complex_seed.ipynb`** -  Same as `IV_1_a_test_JSA_ideal_local_oscillator_optimization.ipynb`, but using generalized complex-valued random seeds for the LO initialization.  
  _Results summarized in Table 1 of the paper._ 

* **`IV_2_a_test_JSA_calc_local_oscillator_optimization.ipynb`** - LO optimization for a realistic case using the data from optimized GeO2-doped silica fiber tapers. Real-valued seeds are used.  
  _Results summarized in Table 2 of the paper._

* **`IV_2_b_test_JSA_calc_local_oscillator_optimization_complex_seed.ipynb`** -   Same as `IV_2_a_test_JSA_calc_local_oscillator_optimization.ipynb`, but using complex-valued random seeds.  
  _Results summarized in Table 2 of the paper._


The dispersion data for the pump and triplet modes is available in the file _disp_data.npy_ of this repository, and the mode electric field intensities are available in the folder _mode_data.npy_. 
