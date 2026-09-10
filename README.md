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

## 🚀 Quick Start Instructions

### 1. Install Dependencies
Open PowerShell or Command Prompt in this directory and execute:
```bash
python -m pip install -r requirements.txt
```

### 2. Launch the Application
Run the Streamlit application:
```bash
python -m streamlit run app.py
```
This will automatically launch the app in your default web browser at `http://localhost:8501`.

---

## 📂 Deploying Your Trained Model Weights
If you have exported your trained hybrid model (e.g. from Python or scikit-learn/XGBoost), simply save it in this directory with any of the following names:
- `hybrid_pso_xgb.pkl`
- `pso_xgb_model.pkl`
- `xgb_model.pkl`

The application will automatically detect and load your trained weights. If no file is placed, it runs with the internal calibrated mechanics surrogate based on Table 1 of the paper.

---

## 🌐 Deploying to Streamlit Cloud (For Your Paper Submission)
To get a public link (e.g., `https://rc-frp-interaction.streamlit.app`) to cite in your manuscript:
1. Push `app.py`, `requirements.txt`, and your model file to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Select your repository, set the main file path to `app.py`, and click **Deploy**.
4. Include the resulting URL in Section 5 of your paper!
