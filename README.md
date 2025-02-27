#  Fact-Checking Model (LIAR Dataset)

This repository contains a **fact-checking model** trained on the **LIAR dataset** (~12.8K labeled statements). The model classifies political statements as either **true or false**.

## 📌 Features
✔️ **Fine-tuned Transformers** 'roberta-base'
✔️ **Binary Classification for High Accuracy (~92.5%)**
✔️ **Interactive Web UI** via **Streamlit**  

## 🛠 Installation
### **1️⃣ Clone the Repository**
git clone https://github.com/Leenalhadid/fact_checking_system.git

cd fact_checking_system

### **1️⃣install dependencies**

pip install torch transformers datasets evaluate pandas scikit-learn streamlit

## **to classify a statement from the terminal**:
python leen_test.py --query "your statement"

## **Web-Based UI (Streamlit)**
streamlit run leen_test.py
