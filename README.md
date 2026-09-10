# P-M Interaction Behavior Predictor for FRP-Strengthened RC Columns

This repository contains the interactive **Streamlit GUI application** developed for the research paper:

> **"Proposing a Novel Hybrid Machine Learning Framework for Predicting the Interaction Behavior of Reinforced Concrete Columns under Axial Compression and Bending"**  
> *M. Chellapandian, S.P. Murali Kannan, and K. Muthamil Sudar (2026)*  
> *Centre for Disaster Mitigation and Management, Vellore Institute of Technology & Mepco Schlenk Engineering College*

---

## 🌟 Key Features
1. **18 Design Parameters**: Covers cross-sectional dimensions, concrete strength, internal steel reinforcement, NSM FRP laminates, EB FRP wraps, and load eccentricity.
2. **Presets for Retrofit Schemes**: One-click configuration for:
   - Control RC Column (Unstrengthened)
   - NSM FRP Strengthened Only
   - EB FRP Confinement Wrap Only
   - Hybrid NSM + EB FRP System
3. **Dual Output Predictions**:
   - Ultimate Axial Peak Load ($P_u$ in kN)
   - Ultimate Bending Moment ($M_u$ in $\text{kN}\cdot\text{m}$)
   - Retrofit Strength Gain percentage vs. unstrengthened control column.
4. **Uncertainty Quantification**: 95% Confidence / Predictive Interval calibrated from the testing RMSE ($155.65\text{ kN}$).
5. **Dynamic $P\text{–}M$ Interaction Envelope**: Plots the continuous interaction failure curve ($0 \le e \le 1.05d$) and highlights the current design state.
6. **Data Export**: One-click download of the complete design calculation in `.csv` format.

---
