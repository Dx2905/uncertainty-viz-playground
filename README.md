

# 🎛️ **Uncertainty Visualization Playground**

### *An interactive Streamlit playground for exploring prediction uncertainty, calibration, and SHAP explanations for clinical AI models.*

---

## 🔖 Badges


[![Streamlit](https://img.shields.io/badge/App-Streamlit-red)]()
[![Uncertainty](https://img.shields.io/badge/Visualization-Uncertainty-orange)]()
[![Explainability](https://img.shields.io/badge/XAI-SHAP-blue)]()
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()


---

## 📌 **Overview**

The **Uncertainty Viz Playground** is an interactive dashboard for experimenting with how clinical prediction uncertainty can be shown to users.
It allows you to load model outputs (from Project 1), visualize uncertainty distributions, inspect misclassifications, and explore local explanations using SHAP.

This project is intentionally **hands-on and exploratory** — a sandbox for testing visual encodings and preparing material for the **clinical-uncertainty-design-space** mini-paper.

---

# 🎯 **Core Purpose**

Modern clinical ML systems often output just a single probability — but clinicians require richer information:

* How uncertain is this prediction?
* What range of outcomes are plausible?
* What features drove the decision?
* Does this patient belong to a high-variance or high-risk cluster?
* Are we at risk of false positives or false negatives?

This playground lets you **test visualizations**, compare encoding styles, and rapidly iterate on design ideas without retraining the entire ML pipeline.

---

## 🚀 **Key Features**

### 🔹 **Load Model Outputs**

* Supports CSV files containing:

  * predicted probabilities
  * labels
  * bootstrap distributions
  * SHAP values
* Default sample file included:
  `./data/sample_predictions_heart.csv`

### 🔹 **Uncertainty Visualization**

* Histogram/density of bootstrap samples
* Uncertainty range indicators
* Distribution spread → stability indicator
* Identify high-variance predictions

### 🔹 **Calibration View**

* Mini reliability plot
* Binned calibration error
* Quick visual sense of model trustworthiness

### 🔹 **Local Explanations (SHAP)**

* Inspect individual patient explanations
* Local feature contributions
* Toggle between explanation and uncertainty

### 🔹 **Misclassification Explorer**

* Top false positive cases
* Top false negative cases
* Sort by model confidence or uncertainty
* Understand model failure patterns

### 🔹 **Streamlit UI**

* Sidebar controls
* Case selector
* Responsive visualization layout
* Simple and expandable interface

---

## 🧱 **Architecture**

```
uncertainty-viz-playground/
├─ data/
│   ├─ sample_predictions_heart.csv   # example input
│
├─ app/
│   ├─ playground.py                  # Streamlit app
│   ├─ views.py                       # modular visualization logic
│   └─ components/                    # optional UI helpers
│
├─ utils/
│   ├─ load_data.py                   # CSV loader + validation
│   ├─ uncertainty.py                 # density, histograms, intervals
│   ├─ shap_utils.py                  # load/plot SHAP values
│   └─ errors.py                      # FP/FN summary helpers
│
└─ README.md
```

---

## 🛠 **Installation**

```bash
git clone https://github.com/Dx2905/uncertainty-viz-playground.git
cd uncertainty-viz-playground
pip install -r requirements.txt
```

---

## ▶️ **Run the App**

```bash
streamlit run app/playground.py
```

Streamlit will launch the interactive dashboard in your browser.

---

## 📊 **Understanding the Visuals**

### **1. Uncertainty Distribution Plot**

Shows a histogram + KDE of bootstrap samples.

Questions answered:

* “How stable is this prediction?”
* “Is the model guessing or confident?”

### **2. Calibration Snapshot**

Mini-version of reliability curves:

* Overconfidence / underconfidence
* Spread of calibration bins

### **3. SHAP Local Explanation**

Explains:

* “Which features drove THIS specific prediction?”
* “Is uncertainty correlated with certain features?”

### **4. Misclassification Explorer**

Useful for:

* Failure understanding
* Error-aware triage
* Identifying where explanations and uncertainty disagree

---

## 🧪 **Example Input File Requirements**

Minimum columns (for full functionality):

```
prediction
label
bootstrap_samples (JSON or list)
shap_values (JSON or list)
features... (optional)
```

Example included in:

```
data/sample_predictions_heart.csv
```

---

## 📷 **Screenshot Placeholders**

Add screenshots like:

```
![Uncertainty Distribution](./screenshots/uncertainty_distribution.png)
![SHAP Local Explanation](./screenshots/shap_local.png)
![Misclassification Explorer](./screenshots/misclassification_view.png)
```

---

## 🔮 **Future Extensions**

* Add conformal intervals
* Allow uploading multiple models
* Add visual encoding variants for research comparison
* Multi-patient summary view
* Interactive cohort clustering
* Export prototype panels for papers

---

## 📚 **Relation to Project 2 (Design Space Mini-Paper)**

This playground provides **live, interactive versions** of several prototypes described in the design space:

* dotplots
* distributions
* reliability slices
* feature explanations
* FP/FN analysis

It acts as your **experimental sandbox** prior to formalizing the design space into the paper.

---

## ✉️ **Contact**

**Fnu Gaurav**
Email: [yadav.gaurav2905@gmail.com](mailto:yadav.gaurav2905@gmail.com)
LinkedIn: [https://www.linkedin.com/in/fnu-gaurav-653355252/](https://www.linkedin.com/in/fnu-gaurav-653355252/)
GitHub: [https://github.com/Dx2905](https://github.com/Dx2905)

---

