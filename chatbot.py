import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Initialize Flask app
app = Flask(__name__)

# FAQ Dataset (sample, replace with your actual FAQ data)
faq_data = """
What is Docify Online?
Docify Online is a platform for filling out medical certificates and consultation forms, with support from our chatbot.

How do I submit a consultation form?
Log in, go to the dashboard, and fill out the form with your symptoms. You can also update past submissions.

Is my data secure?
Yes, we use password hashing and store data securely in a database. User details are also exported to a CSV file.

How can I contact support?
You can reach our support team via the chatbot or email at support@docify.online.

What should I include in the symptoms field?
Describe your symptoms in detail, including duration, severity, and any relevant medical history.

=====================================
Disease: Diabetes
=====================================

🩺 Description:
A chronic condition that affects how your body processes blood sugar (glucose).

🔍 Common Symptoms:
- Frequent urination
- Excessive thirst
- Fatigue
- Blurred vision

👨‍⚕️ Recommended Specialist:
Endocrinologist

📝 Diagnosis:
- Fasting Blood Sugar Test
- HbA1c Test
- Glucose Tolerance Test

💊 Treatment / Management:
- Lifestyle changes (diet, exercise)
- Oral medications (e.g., Metformin)
- Insulin therapy (for Type 1 or advanced Type 2)

---

=====================================
Disease: Hypertension (High Blood Pressure)
=====================================

🩺 Description:
A condition in which the force of the blood against the artery walls is too high.

🔍 Common Symptoms:
- Often asymptomatic
- Headaches
- Dizziness
- Nosebleeds

👨‍⚕️ Recommended Specialist:
Cardiologist / General Physician

📝 Diagnosis:
- Blood Pressure Monitoring
- ECG, ECHO, and blood tests if needed

💊 Treatment / Management:
- Lifestyle modification
- Antihypertensive medications
- Stress management

---

=====================================
Disease: Asthma
=====================================

🩺 Description:
A respiratory condition marked by spasms in the bronchi of the lungs.

🔍 Common Symptoms:
- Wheezing
- Shortness of breath
- Chest tightness
- Persistent coughing

👨‍⚕️ Recommended Specialist:
Pulmonologist

📝 Diagnosis:
- Spirometry
- Peak flow test
- Allergy testing

💊 Treatment / Management:
- Inhalers (bronchodilators, corticosteroids)
- Avoidance of allergens and irritants

---

=====================================
Disease: Depression
=====================================

🩺 Description:
A mental health disorder characterized by persistently low mood and loss of interest.

🔍 Common Symptoms:
- Persistent sadness
- Loss of interest
- Fatigue
- Sleep/appetite changes
- Thoughts of self-harm

👨‍⚕️ Recommended Specialist:
Psychiatrist / Psychologist

📝 Diagnosis:
- Psychological evaluation
- DSM-5 criteria

💊 Treatment / Management:
- Psychotherapy (CBT, talk therapy)
- Antidepressant medication
- Lifestyle changes, support systems

---

=====================================
Disease: Migraine
=====================================

🩺 Description:
A neurological condition causing intense, throbbing headaches often on one side.

🔍 Common Symptoms:
- Severe headache
- Nausea and vomiting
- Sensitivity to light and sound

👨‍⚕️ Recommended Specialist:
Neurologist

📝 Diagnosis:
- Clinical history and symptom pattern
- MRI/CT scan to rule out other conditions

💊 Treatment / Management:
- Migraine-specific medications (Triptans)
- Preventive therapy
- Trigger management

---

=====================================
Disease: Arthritis
=====================================

🩺 Description:
Inflammation of joints that causes pain and stiffness.

🔍 Common Symptoms:
- Joint pain and swelling
- Morning stiffness
- Reduced range of motion

👨‍⚕️ Recommended Specialist:
Rheumatologist

📝 Diagnosis:
- Blood tests for inflammation markers
- X-rays or MRI of joints

💊 Treatment / Management:
- Anti-inflammatory drugs
- Physical therapy
- Joint protection techniques

---

=====================================
Disease: Tuberculosis (TB)
=====================================

🩺 Description:
A serious infectious disease that mainly affects the lungs.

🔍 Common Symptoms:
- Chronic cough with blood
- Night sweats
- Weight loss
- Fever

👨‍⚕️ Recommended Specialist:
Pulmonologist / Infectious Disease Specialist

📝 Diagnosis:
- Chest X-ray
- Sputum test
- Tuberculin skin test

💊 Treatment / Management:
- Long-term antibiotics (6 months)
- Directly Observed Therapy (DOT)

---

=====================================
Disease: PCOS (Polycystic Ovary Syndrome)
=====================================

🩺 Description:
A hormonal disorder causing enlarged ovaries with small cysts.

🔍 Common Symptoms:
- Irregular periods
- Acne
- Weight gain
- Excess facial/body hair

👨‍⚕️ Recommended Specialist:
Gynecologist / Endocrinologist

📝 Diagnosis:
- Hormonal blood tests
- Pelvic ultrasound

💊 Treatment / Management:
- Hormone therapy
- Lifestyle modifications
- Metformin for insulin resistance

---

=====================================
Disease: Thyroid Disorder
=====================================

🩺 Description:
Imbalance in thyroid hormone production (hypo or hyperthyroidism).

🔍 Common Symptoms:
- Weight changes
- Fatigue
- Hair thinning
- Cold or heat intolerance

👨‍⚕️ Recommended Specialist:
Endocrinologist

📝 Diagnosis:
- TSH, T3, T4 blood tests
- Thyroid ultrasound

💊 Treatment / Management:
- Thyroid hormone replacement
- Anti-thyroid medications
- Regular hormone monitoring

---

"""


# Step 1: Data Collection and Preprocessing
def preprocess_faq_data(faq_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_text(faq_text)
    return chunks


faq_chunks = preprocess_faq_data(faq_data)

# Step 2: Embedding and Vector Store Setup
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Create FAISS vector store
vector_store = FAISS.from_texts(faq_chunks, embedding_model)
vector_store.save_local("faiss_index")

# Load vector store (for subsequent runs)
vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Step 3: Initialize Small LLM (google/flan-t5-small)
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
text2text_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=200,
    device=-1  # CPU
)

llm = HuggingFacePipeline(pipeline=text2text_pipeline)

# Step 4: RAG Setup with LangChain
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print(retriever)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


# Step 5: Process Query and Generate Structured Response
def process_query(user_query, symptoms=None):
    # Retrieve relevant FAQ documents
    context = qa_chain({"query": user_query,"Symptoms":symptoms})['result']

    # Construct prompt with symptoms (if provided) and FAQ context
    prompt = f"""
    You are a medical chatbot for Docify Online. Answer the user's query in a structured, clear, and concise manner.
    Use the following FAQ context to inform your response:
    {context}

    User Query: {user_query}
    """
    if symptoms:
        prompt += f"\nUser Symptoms: {symptoms}\nPlease incorporate the symptoms into your response if relevant."

    prompt += """
    Provide the response in the following format:
    **Answer**: [Your answer here]
    **Additional Info**: [Any relevant details or suggestions]
    """

    # Generate response
    response = text2text_pipeline(prompt)[0]["generated_text"]
    print(response)
    return response


# Flask Route for Chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_query = data.get('message')
    symptoms = data.get('symptoms')

    if not user_query:
        return jsonify({"reply": "Please provide a query."}), 400

    response = process_query(user_query, symptoms)
    if response is None:
        return jsonify("hello how are you please ask a relevant query regarding the site")
    return jsonify({"reply": response})


if __name__ == '__main__':
    app.run(debug=True, port=5001)