import os
from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import StuffDocumentsChain


# Suppress TensorFlow and duplicate library issues
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)

# ======== FAQ Dataset ========
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
"""

# ======== Preprocess FAQ into Chunks ========
def preprocess_faq_data(faq_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    return text_splitter.split_text(faq_text)

faq_chunks = preprocess_faq_data(faq_data)

# ======== Embeddings & VectorStore ========
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

if not os.path.exists("faiss_index"):
    vector_store = FAISS.from_texts(faq_chunks, embedding_model)
    vector_store.save_local("faiss_index")
else:
    vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# ======== Retriever ========
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ======== LLM & Prompt Setup ========
llm = OllamaLLM(model="docify", base_url="http://localhost:11434")

prompt_template = PromptTemplate(
    input_variables=["context", "question", "symptoms_section"],
    template="""
You are a helpful AI medical assistant. Use the information from the context below and the symptoms (if provided)
to answer the question as clearly and accurately as possible.

{symptoms_section}

Context:
{context}

Question: {question}

Answer:"""
)

llm_chain = LLMChain(llm=llm, prompt=prompt_template)
combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context"
)

# ======== Query Processor ========
def process_query(user_query, symptoms=None):
    # Prepare the symptoms section if provided
    symptoms_section = f"User Symptoms: {symptoms}\nIncorporate these symptoms into your response if relevant." if symptoms else ""

    # Retrieve the top 3 relevant documents
    top_docs = retriever.get_relevant_documents(user_query)[:3]

    # Debug: Print the retrieved documents
    print("--- Retrieved Documents ---")
    for i, doc in enumerate(top_docs):
        print(f"Doc {i+1}: {doc.page_content}")
        print("-" * 50)

    # Pass the relevant documents to the chain for processing
    from langchain.llms import Ollama
    ollama = Ollama(base_url='http://localhost:11434', model="docify")
    result=ollama(f"answer user query base on retrived information{user_query}+{top_docs} give short and summerized answer"
                  f"do not recomand and medication ask them to fill the form and consult a doc")
    print(result)
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
    #manual_evaluation()
    app.run(debug=True, port=5003)
