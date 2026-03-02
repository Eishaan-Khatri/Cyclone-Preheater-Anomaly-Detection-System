# Cyclone Preheater Anomaly Detection System

**Project Title:** Advanced Anomaly Detection for Cyclone Preheater (Cement Plant)  
**Author:** Eishaan Khatri  
**Date:** March 2026  
**Type:** Unsupervised Machine Learning + Domain-Driven Feature Engineering  

---

## 📋 Overview

This project detects abnormal operating periods in a **cyclone preheater** system using real-time sensor data.  

This is an advanced Isolation Forest model which includes:
- Cleaning noisy industrial data (error strings like "unit down", "timeout", "not connect")
- Engineering **physics-informed** features (pressure drops, heat transfer efficiency, temperature gradients)
- Adding **temporal** features (rate-of-change, 1-hour volatility, deviation from moving average)
- Training an enhanced Isolation Forest (28+ features instead of 6)
- Applying temporal smoothing + period grouping to eliminate noise
- Performing root-cause analysis and generating professional visualisations

**Result:** Highly accurate, explainable anomaly periods that can be directly used for maintenance alerts.

---

## 🗂️ Dataset

- **File:** `data(internship-data-1).csv`
- **Sampling:** ~5-minute intervals
- **Time Range:** 2019–2022 (3+ years)
- **Sensors (6 raw):**
  - `Cyclone_Inlet_Draft`
  - `Cyclone_Outlet_Gas_draft`
  - `Cyclone_cone_draft`
  - `Cyclone_Inlet_Gas_Temp`
  - `Cyclone_Material_Temp`
  - `Cyclone_Gas_Outlet_Temp`

---

## 🛠️ Requirements

```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
