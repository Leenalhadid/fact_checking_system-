import os
import shutil
import torch
import subprocess
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define model path
model_path = "my_saved_model"
backup_zip = "my_saved_model_backup.zip"

# Step 1: Train the model if not found
if not os.path.exists(model_path):
    print("Model not found. Training...")
    
    # Run the training script ( main code)
    subprocess.run(["python", "main.py"], check=True)

# Step 2: Unzip the saved model if necessary
if os.path.exists(backup_zip):
    print("Extracting model backup...")
    shutil.unpack_archive(backup_zip, model_path)
    print("Model extracted successfully!")

# Step 3: Load the trained model
print("Loading model...")
loaded_model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Model loaded successfully!")

# Step 4: Function to Predict Labels
def predict_label(text):
    with torch.no_grad():
        inputs = loaded_tokenizer(text, return_tensors="pt").to(device)
        logits = loaded_model(**inputs).logits
        probs = softmax(logits.cpu().numpy(), axis=1)
        return ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"][np.argmax(probs)]

# Step 5: Streamlit UI for Query Input
st.title('Leen Test')
query = st.text_input("Enter your query:")

if st.button('Get Answer') and query:
    st.subheader("Answer:")
    st.write(predict_label(query))
