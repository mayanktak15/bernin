import json
import pandas as pd
from datasets import Dataset

# Synthetic Q&A pairs for Docify Online
synthetic_data = [
    {
        "question": "What is Docify Online?",
        "answer": "Docify Online is a platform for filling out medical certificates and consultation forms, with support from a chatbot to answer your questions."
    },
    {
        "question": "How do I submit a consultation form on Docify Online?",
        "answer": "Log in, navigate to the dashboard, and fill out the form with your symptoms. You can edit past submissions if needed."
    },
    {
        "question": "Is my data secure on Docify Online?",
        "answer": "Yes, we use password hashing and store data securely in a database. User details are exported to a CSV file, excluding sensitive information."
    },
    {
        "question": "How should I describe my symptoms in the consultation form?",
        "answer": "Include details like the type of symptoms (e.g., fever, cough), their duration, severity, and any relevant medical history."
    },
    {
        "question": "What should I do if I have a fever?",
        "answer": "Stay hydrated, rest, and take acetaminophen or ibuprofen as directed. Consult a doctor if the fever lasts more than 48 hours or exceeds 103°F."
    },
    {
        "question": "Can I update my consultation details?",
        "answer": "Yes, go to the dashboard, select the consultation, and edit the symptoms. The updated details will be saved with a new timestamp."
    },
    {
        "question": "What are common symptoms of a cold?",
        "answer": "Common cold symptoms include runny nose, sore throat, cough, and mild fever. Rest and hydration can help with recovery."
    },
    {
        "question": "How can I contact support for Docify Online?",
        "answer": "Use the chatbot on the dashboard or email support@docify.online for assistance."
    },
    {
        "question": "What does a sore throat indicate?",
        "answer": "A sore throat may indicate a viral infection, bacterial infection like strep throat, or allergies. Seek medical advice if it persists or worsens."
    },
    {
        "question": "How do I manage a headache?",
        "answer": "Rest in a quiet, dark room, stay hydrated, and take over-the-counter pain relievers like ibuprofen. Consult a doctor if it’s severe or frequent."
    },
    {
        "question": "What should I do if I have a cough?",
        "answer": "Stay hydrated, use a humidifier, and consider over-the-counter cough remedies. If the cough lasts more than a week, consult a doctor."
    },
    {
        "question": "How do I register on Docify Online?",
        "answer": "Go to the register page, provide your name, phone, email, and password, then submit. You’ll be prompted to log in afterward."
    },
    {
        "question": "What is the purpose of the consultation form?",
        "answer": "The consultation form collects your symptoms to help doctors provide accurate medical advice or certificates."
    },
    {
        "question": "Can I ask the chatbot about my symptoms?",
        "answer": "Yes, the chatbot can provide general advice based on your symptoms and answer questions about using Docify Online."
    },
    {
        "question": "What if I forget my password?",
        "answer": "Contact support@docify.online to reset your password, as password recovery is not currently automated."
    },
    {
        "question": "How long should I wait for a response from support?",
        "answer": "Support typically responds within 24-48 hours via email or the chatbot."
    },
    {
        "question": "What are symptoms of allergies?",
        "answer": "Allergy symptoms include sneezing, itchy eyes, runny nose, and skin rashes. Avoid triggers and consider antihistamines."
    },
    {
        "question": "How do I log out of Docify Online?",
        "answer": "Click the ‘Logout’ link in the navigation bar to end your session."
    },
    {
        "question": "What if my symptoms worsen?",
        "answer": "Update your consultation form with new symptoms and contact a doctor immediately if symptoms are severe."
    },
    {
        "question": "Can the chatbot diagnose my condition?",
        "answer": "The chatbot provides general advice based on symptoms but cannot diagnose conditions. Consult a doctor for a diagnosis."
    }
]

# Load Kaggle dataset (assumes train.json is downloaded)
def load_kaggle_dataset(file_path="train.json"):
    with open(file_path, 'r') as f:
        kaggle_data = json.load(f)
    # Extract question and answer, clean entries
    cleaned_data = []
    for entry in kaggle_data:
        question = entry.get("question", "").strip()
        answer = entry.get("answer", "").strip()
        if question and answer:  # Skip empty or invalid entries
            cleaned_data.append({"question": question, "answer": answer})
    return cleaned_data

# Combine datasets
kaggle_data = load_kaggle_dataset()  # Update path if needed
combined_data = kaggle_data + synthetic_data

# Clean and deduplicate
seen_questions = set()
deduped_data = []
for item in combined_data:
    if item["question"] not in seen_questions:
        seen_questions.add(item["question"])
        deduped_data.append(item)

# Convert to Dataset
df = pd.DataFrame(deduped_data)
dataset = Dataset.from_pandas(df)

# Save dataset for fine-tuning
dataset.save_to_disk("medical_dataset")

print(f"Dataset prepared with {len(deduped_data)} Q&A pairs. Saved to 'medical_dataset'.")