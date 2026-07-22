# 🏦 Banking Intent Classification

## 1. Project Overview

**Banking Intent Classification** is a Natural Language Processing (NLP) project for automatic classification of banking customer support requests into predefined intent categories.

The application predicts one of **77 banking intents** from a customer's message and displays the three most probable predictions together with calibrated probabilities.

🌐 **Live Demo:** https://banking-intent-classification.onrender.com/

---

# 2. Business Problem and Objective

Financial institutions receive thousands of customer requests every day through mobile applications, chatbots, and support centers.

Manual routing of these requests is time-consuming and expensive.

The objective of this project is to build a machine learning model capable of automatically recognizing customer intent, enabling:

- faster request processing;
- intelligent ticket routing;
- chatbot automation;
- reduced operational costs;
- improved customer experience.

---

# 3. Dataset

**Dataset:** Banking77

Source:

https://huggingface.co/datasets/PolyAI/banking77

Dataset characteristics:

- **Language:** English
- **Number of classes:** 77 banking intents
- **Training samples:** 10,003
- **Test samples:** 3,080
- **Total samples:** 13,083

Each record consists of:

- customer request (text)
- numerical intent label
- textual intent label

Example:

| Text | Intent |
|------|---------|
| My card has not arrived yet | card_arrival |
| My transfer is still pending | pending_transfer |

---

# 4. Evaluation Metric

Since this is a **multi-class classification** problem with an imbalanced class distribution, several evaluation metrics were used:

- Accuracy
- Precision
- Recall
- **Macro F1-score** (primary metric)
- Weighted F1-score

The **Macro F1-score** was selected as the primary evaluation metric because it assigns equal importance to every class regardless of its frequency.

---

# 5. Solution Approach

The project followed a complete machine learning workflow.

## Data preprocessing

- text cleaning
- lowercasing
- punctuation removal
- train/validation split

## Feature Engineering

Traditional machine learning:

- TF-IDF vectorization

Deep learning:

- DistilBERT tokenizer

## Models

The following models were trained and compared:

- Dummy Classifier
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- Fine-tuned DistilBERT

For deployment, the Linear SVM model was calibrated using **CalibratedClassifierCV** to obtain probability estimates.

## Technologies

- Python
- Scikit-learn
- Streamlit
- NumPy
- SciPy
- Joblib
- Docker

---

# 6. Experimental Results

| Model | Test Accuracy | Test Macro F1 | Deployment |
|--------|--------------:|--------------:|:----------:|
| Dummy Classifier | Baseline | Baseline | ❌ |
| TF-IDF + Logistic Regression | 0.8818 | 0.8819 | ❌ |
| TF-IDF + Linear SVM | 0.8880 | 0.8879 | ❌ |
| **TF-IDF + Calibrated Linear SVM** | **≈0.8880*** | **≈0.8879*** | ✅ |
| Fine-tuned DistilBERT | **0.9094** | **0.9091** | ❌ |

\* The calibrated Linear SVM uses the same underlying classifier as the original Linear SVM but applies **CalibratedClassifierCV (sigmoid calibration)** to produce reliable probability estimates (`predict_proba()`). 
This makes the model suitable for production deployment while maintaining performance very close to the original Linear SVM.

Although DistilBERT achieved the highest predictive performance, the deployed application uses a **TF-IDF + Calibrated Linear SVM** model because it offers:

- significantly lower memory consumption;
- much faster inference;
- probability estimates;
- lightweight deployment suitable for Render Free Tier.

---

# 7. Conclusions

The project demonstrates that both traditional machine learning and transformer-based models are effective for banking intent classification.

Key findings:

- DistilBERT achieved the highest overall performance.
- Linear SVM delivered competitive results while requiring considerably fewer computational resources.
- CalibratedClassifierCV enabled reliable probability estimation for the deployed SVM model.
- The deployed application provides real-time intent prediction with Top-3 recommendations.

The developed solution can be integrated into customer support systems or chatbot platforms to automate request routing and improve operational efficiency.

---

# 8. Installation & Usage

## Clone repository

```bash
git clone <repository-url>
cd banking-intent-classification
```

## Install dependencies

```bash
pip install -r requirements-app.txt
```

## Run the application

```bash
streamlit run app/streamlit_app.py
```

Open your browser:

```
http://localhost:8501
```

Or use the deployed version:

https://banking-intent-classification.onrender.com/

---

# 9. Requirements

Main project dependencies:

```text
streamlit>=1.46,<2.0
scikit-learn
numpy
scipy
joblib
```

Python version:

```
Python 3.11+
```

---

## Project Structure

```
.
├── app/
│   ├── streamlit_app.py
│   └── assets/
├── models/
│   └── svm_deployment.joblib
├── notebooks/
├── src/
├── Dockerfile
├── requirements-app.txt
└── README.md
```

---

## Author

**Oleh F.**

Master's Degree Project

2026