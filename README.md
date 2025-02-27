#  Fact-Checking Model (LIAR Dataset)

This repository contains a **fact-checking model** trained on the **LIAR dataset** (~12.8K labeled statements). The model classifies political statements as either **true , false , barely true, mostly-true, half-true, "pants-fire**.

## 📌 Features
✔️ **Fine-tuned Transformers** 'roberta-base'**
✔️ **Binary Classification for High Accuracy (~92.5%)**
✔️ **Interactive Web UI** via **Streamlit**  

## 1️⃣ Prerequisites
Before starting, ensure you have:
✔ Python 3.8 or higher installed (Check using python --version)

## 🛠 Installation
### Clone the Repository
git clone https://github.com/Leenalhadid/fact_checking_system.git

### Navigate to the Project Directory
cd fact_checking_system

### install dependencies

pip install torch transformers datasets evaluate pandas scikit-learn streamlit

## 🚀 Using the Model
## **to classify a statement from the terminal**:
python leen_test.py --query "your statement"

## Web-Based UI (Streamlit)
streamlit run leen_test.py
