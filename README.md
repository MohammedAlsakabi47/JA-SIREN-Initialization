# Official implementation of JA-SIREN initialization

Implementation of **JA-SIREN**, a deterministic initialization scheme for sinusoidal implicit neural representations (INRs), introduced in *"Jacobi-Anger Method for Deterministic Initialization in Implicit Neural Representation"* (MLSP 2026 - Oral).

## Overview

Sinusoidal INRs (e.g., SIREN) rely on stochastic weight initialization, which offers no guarantee on the network's initial spectral response and causes run-to-run PSNR variance of up to 2.5 dB. JA-SIREN replaces this with a **closed-form, deterministic initialization** derived from the target signal's own spectrum — no random seed, no hyperparameter tuning.

## Method

Given a signal to fit:

1. Compute the **DST-II** of the target signal and retain the top-*M* frequency components {k_m} with amplitudes {α_m}.
2. Construct a two-layer sinusoidal MLP of width *M* with **diagonal second-layer weights**, so each neuron corresponds to exactly one frequency.
3. Set first-layer weights: `w¹_m = k_m`
4. Match each neuron's output to its DST term via the **Jacobi-Anger expansion**, solving for the second-layer weight: `w²_m = J₁⁻¹(|α_m| / 2)` with the sign absorbed into the output weight: `w_out_m = sign(α_m)`.

This yields a fully deterministic initialization that analytically matches the network's initial output spectrum to the target signal, requiring zero random seeds or tuning.
