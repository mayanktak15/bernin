import os
from flask import Flask, request, jsonify
from vector_creator import get_vector_store

# Suppress TensorFlow and duplicate library issues
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)

# ======== FAQ Dataset ========


vector_store=get_vector_store(r"D:\bernin\Assignment\faq.txt")
# ======== Retriever ========
retriever = vector_store.as_retriever(search_kwargs={"k": 3})



# ======== Query Processor ========
def process_query(user_query, symptoms=None):
    # Prepare the symptoms section if provided
    symptoms_section = f"User Symptoms: {symptoms}\nIncorporate these symptoms into your response if relevant." if symptoms else ""

    # Retrieve the top 3 relevant documents
    top_docs = retriever.get_relevant_documents(user_query)[:3]

    # Debug: Print the retrieved documents
    result = ""
    print("--- Retrieved Documents ---")
    for i, doc in enumerate(top_docs):
        doc_text = f"Doc {i + 1}: {doc.page_content}\n" + "-" * 50 + "\n"
        print(doc_text)
        result += doc_text

    return result


# ======== Flask API Route ========
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_query = data.get('message')
    symptoms = data.get('symptoms')

    if not user_query:
        return jsonify({"reply": "Please provide a query."}), 400

    response = process_query(user_query, symptoms)
    return jsonify({"reply": response})

# ======== Manual Evaluation ========
def manual_evaluation():
    test_queries = [
        {"query": "How do I manage a fever?", "symptoms": "Fever for 2 days, 101°F"},
        {"query": "What is Docify Online?", "symptoms": None},
        {"query": "What does a sore throat mean?", "symptoms": "Sore throat and cough"},
        {"query": "How do I update my consultation?", "symptoms": None},
        {"query": "What doctor should I see for a headache?", "symptoms": "Frequent headaches for a week"}
    ]
    print("Manual Evaluation Results:")
    for test in test_queries:
        response = process_query(test["query"], test["symptoms"])
        print(f"\nQuery: {test['query']}")
        if test["symptoms"]:
            print(f"Symptoms: {test['symptoms']}")
        print(f"Response: {response}")
        print("-" * 50)

# ======== Main Entrypoint ========
if __name__ == '__main__':
    # Comment this out in production
    manual_evaluation()
    app.run(debug=True, port=5003)
