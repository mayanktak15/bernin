import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask, request, jsonify
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel, PeftConfig
import torch

# Initialize Flask app
app = Flask(__name__)

# FAQ Dataset
faq_data = """

**1. What should I do if I have a fever?**
Stay hydrated, rest, and take paracetamol if necessary. If fever persists for more than 3 days, consult a doctor. (Recommended doctor: General Physician)

**2. How do I book an online doctor consultation on Docify?**
Visit the 'Consult a Doctor' page, fill in your symptoms, select a specialty, and book an available time slot.

**3. Can I get a medical certificate online?**
Yes, fill out the medical certificate form on Docify with valid symptoms and upload a valid ID. A doctor will review and issue the certificate.

**4. What should I do if I have a sore throat?**
Gargle with warm salt water, stay hydrated, and avoid cold drinks. Consult an ENT specialist if it persists beyond 5 days.

**5. What is the validity of an online medical certificate?**
Most online medical certificates are valid for 1 to 7 days, depending on the doctor's assessment.

**6. Do I need a prescription to buy antibiotics?**
Yes, antibiotics should only be taken with a valid doctor's prescription. Self-medication is harmful.

**7. I need rest for 2 days due to body pain. Can I get a certificate?**
Yes, submit your symptoms in the medical certificate form and select your desired duration. A doctor will review and issue it if appropriate.

**8. What should I do in case of a skin rash?**
Avoid scratching, use soothing lotion, and consult a dermatologist if the rash spreads or lasts longer than 3 days.

**9. Can I consult a gynecologist online?**
Yes, Docify allows confidential gynecologist consultations through video or chat. Appointments are private and secure.

**10. How long does it take to get a medical certificate?**
Once submitted, most certificates are processed and delivered digitally within 1 to 4 hours.

**11. I have back pain. Which doctor should I consult?**
For general back pain, consult a General Physician. For chronic or severe pain, consider an Orthopedic specialist.

**12. How do I update incorrect details in the submitted form?**
Login to your account, go to 'My Submissions', and click 'Edit' next to the form to update details.

**13. Is my information safe on Docify?**
Yes, all data is encrypted and only accessible to you and the assigned medical professional.

**14. I have frequent headaches. What can I do?**
Stay hydrated, manage screen time, and track triggers. If headaches persist, consult a Neurologist.

**15. What should I do if I feel dizzy often?**
Sit down immediately, hydrate, and rest. If dizziness continues, consult a General Physician or ENT.

**16. Can I get a fitness certificate for joining a gym?**
Yes, book a consultation with a doctor and mention 'fitness clearance' in your reason.

**17. What if I miss my consultation slot?**
You can reschedule your consultation from the dashboard or request support through live chat.

**18. Do online doctors on Docify issue prescriptions?**
Yes, after your consultation, a valid digital prescription is sent to your registered email and dashboard.

**19. Can I consult a pediatrician for my child online?**
Yes, Docify offers pediatric consultations for children under 18 with parental consent.

**20. What documents are required for a medical certificate?**
Typically, a valid government ID and symptom declaration are required. Doctors may request more if needed.

**21. What should I do if I have loose motion or diarrhea?**
Stay hydrated with ORS. Eat light food. If symptoms persist, consult a Gastroenterologist.

**22. Can I get a certificate for sick leave due to stress?**
Yes, stress-related issues are valid. Book a consultation with a Psychiatrist or General Physician.

**23. I have cold and body ache. Do I need to visit physically?**
No, most common symptoms can be diagnosed online. You can consult a doctor from home via Docify.

**24. How much does a doctor consultation cost?**
Prices vary by specialty. General consultations typically start from ₹199. Check the pricing on the consultation page.

**25. Can I use Docify from outside India?**
Yes, but medical certificates issued are valid within India only. Consultations are accessible globally.

**26. Can I talk to the doctor on call instead of chat?**
Yes, you can choose between voice or video call options while booking the consultation.

**27. I need a certificate for COVID leave. What do I do?**
Select 'COVID symptoms' in the certificate form and upload test results if available. A doctor will review your case.

**28. How long are prescriptions from online doctors valid?**
Prescriptions are generally valid for 7 to 30 days depending on the medication and the doctor's discretion.

**29. I’m experiencing hair loss. Who should I consult?**
Consult a Dermatologist to check for causes like stress, nutrition, or medical conditions.

**30. Can I get a second opinion on Docify?**
Yes, you can consult multiple doctors for a second opinion. Mention previous treatment in the form.

**31. Can I cancel a consultation?**
Yes, cancellations are allowed up to 1 hour before the consultation. Check refund policy for details.

**32. What’s the minimum age for online consultation?**
There is no minimum age, but minors need parental supervision or consent.

**33. I have pain while urinating. What should I do?**
Drink plenty of water and avoid spicy food. Consult a Urologist if symptoms persist.

**34. Is it possible to download my medical certificate?**
Yes, you can download it from your dashboard after approval by the doctor.

**35. Can I get a certificate for travel fitness?**
Yes, select 'Travel Fitness Certificate' from the dropdown while submitting the form.

**36. What if I enter the wrong email or contact number?**
You can update these from your account settings or contact support for assistance.

**37. I feel anxious all the time. Can I talk to a mental health professional?**
Yes, Docify offers confidential consultations with certified Psychologists and Psychiatrists.

**38. Is video consultation necessary for a certificate?**
Not always. Depending on the case, some certificates can be issued via chat consultation.

**39. Can I access my old prescriptions?**
Yes, all prescriptions are stored in your account history for future reference.

**40. What is a general physician consultation good for?**
It’s ideal for fever, cold, cough, fatigue, weakness, and most day-to-day issues.

**41. How long does each consultation last?**
Consultations typically last between 10 to 20 minutes depending on the case.

**42. Can I use insurance for online consultations?**
Some insurance providers may cover teleconsultation. Please check with your provider.

**43. I need a certificate for attending college. What do I choose?**
Select 'Educational Leave Certificate' and mention your reason in the form.

**44. What is the refund policy for consultations?**
Refunds are available if you cancel 1 hour before or if the doctor fails to attend.

**45. Is Docify available 24x7?**
Yes, you can book consultations anytime. Doctors are available based on slots.

**46. What should I do for acidity or gas?**
Eat on time, avoid spicy food, and consult a General Physician or Gastroenterologist if symptoms repeat often.

**47. Can I get an invoice for my consultation?**
Yes, an invoice is generated automatically and can be downloaded from your dashboard.

**48. Can I resubmit a form with corrections?**
Yes, go to 'My Submissions' → 'Edit' → Make changes → Save. The updated form will be reviewed again.



"""


# Preprocess FAQ data
def preprocess_faq_data(faq_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_text(faq_text)
    return chunks


faq_chunks = preprocess_faq_data(faq_data)

# Embedding and Vector Store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

if not os.path.exists("faiss_index"):
    vector_store = FAISS.from_texts(faq_chunks, embedding_model)
    vector_store.save_local("faiss_index")
else:
    vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Load Fine-Tuned Model
model_name = "google/flan-t5-small"
finetuned_path = "fine_tuning/lora_flan_t5_small/finetuned"
tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, finetuned_path)
text2text_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=200,
    device=-1
)

llm = HuggingFacePipeline(pipeline=text2text_pipeline)

# RAG Setup
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
print(retriever.metadata)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


# Process Query
def process_query(user_query, symptoms=None):
    context = qa_chain({"query": user_query})['result']
    prompt = f"""
    You are a medical chatbot for Docify Online. Answer the user's query in a structured, clear, and concise manner.
    Use the following FAQ context to inform your response:
    your role is to answer information about what to do in fever
    {context}

    User Query: {user_query}
    """
    if symptoms:
        prompt += f"\nUser Symptoms: {symptoms}\nPlease incorporate the symptoms into your response if relevant."

    prompt += """
    understand the question and situation of a person then answer:
    **Answer**: [Your answer here]
    **Additional Info**: [Any relevant details or suggestions]
    Do not speculate or provide unverified medical advice.
    """

    response = text2text_pipeline(prompt)[0]["generated_text"]
    return response


# Flask Route
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_query = data.get('message')
    symptoms = data.get('symptoms')
    if not user_query:
        return jsonify({"reply": "Please provide a query."}), 400

    response = process_query(user_query, symptoms)
    return jsonify({"reply": response})


# Manual Evaluation (run separately or comment out after testing)
def manual_evaluation():
    test_queries = [
        {"query": "How do I manage a fever?", "symptoms": "Fever for 2 days, 101°F"},
        {"query": "What is Docify Online?", "symptoms": None},
        {"query": "What does a sore throat mean?", "symptoms": "Sore throat and cough"},
        {"query": "How do I update my consultation?", "symptoms": None},
        {"query": "how can i get my certificat?","symptoms":None}
    ]
    print("Manual Evaluation Results:")
    for test in test_queries:
        response = process_query(test["query"], test["symptoms"])
        print(f"\nQuery: {test['query']}")
        if test["symptoms"]:
            print(f"Symptoms: {test['symptoms']}")
        print(f"Response: {response}")
        print("-" * 50)


if __name__ == '__main__':
    # Run manual evaluation (comment out after testing)
    #manual_evaluation()
    print(retriever.metadata)
    app.run(debug=True, port=5002)