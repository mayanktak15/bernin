import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from vector_creator import get_vector_store
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()  # load variables from .env file

api_key = os.getenv("API_KEY")  # get API key

genai.configure(api_key=api_key)


# Suppress TensorFlow and duplicate library issues


# ======== Vector Store Initialization ========
vector_store = get_vector_store(r"D:\bernin\Assignment\faq.txt")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ======== Query Processor Function ========
def process_query(user_query, symptoms=None):
    symptoms_section = f"User Symptoms: {symptoms}\nIncorporate these symptoms into your response if relevant." if symptoms else ""
    top_docs = retriever.get_relevant_documents(user_query)[:3]

    result = ""
    for i, doc in enumerate(top_docs):
        doc_text = f"Doc {i + 1}: {doc.page_content}\n" + "-" * 50 + "\n"
        result += doc_text
    print(result)
    return result
def process_query2(user_query, symptoms=None):
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
# Optional: Manual evaluation function

# Step 5: Process Query and Generate Structured Response
def process_query3(user_query, symptoms=None):
    model_id = "tiiuae/falcon-7b"

    text_generation_pipeline = pipeline(
        "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, max_new_tokens=400, device=0)

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    prompt_template = """
    <|system|>
    Answer the question based on your knowledge. Use the following context to help:

    {context}

    </s>
    <|user|>
    {question}
    </s>
    <|assistant|>

     """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    llm_chain = prompt | llm | StrOutputParser()
    from langchain_core.runnables import RunnablePassthrough

    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

    # Generate and return response
    try:
        response = rag_chain.invoke(prompt)
        response = response.replace("</s>", "").strip()
        print("Model response:", response)
        return response
    except Exception as e:
        print("Model generation error:", e)
        return "Sorry, there was an error generating a response."

# Process Query
def process_query4(user_query, symptoms=None):
    model_name = "google/flan-t5-base"
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

def process_query5(user_query,symptom:None):
    generation_config = {

        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 15000,
        "response_mime_type": "text/plain",
    }
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
    ]
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
    )
    # Generate summary using Gemini model
    top_docs = retriever.get_relevant_documents(user_query)[:3]
    summary = model.generate_content(contents=(
        f"U are a chatbot for docify answer in minmum words about the faq user ask "
        f"Docify is an online platform that allows users to consult certified doctors from the comfort of their home. Whether it's a minor health concern or the need for a medical certificate"
        f"now user can ask unreleveant question make sure not to answer them"
        f"do not provide any medical consultation form your side"
        f"strictly follow the context provide to you"
        f"query={user_query},extracted_content={top_docs}"

    ))


    return summary.text


def manual_evaluation():
    test_queries = [
        {"query": "How do I manage a fever?", "symptoms": "Fever for 2 days, 101°F"},
        {"query": "What is Docify Online?", "symptoms": None},
    ]
    for test in test_queries:
        response = process_query(test["query"], test["symptoms"])
        print(f"\nQuery: {test['query']}")
        if test["symptoms"]:
            print(f"Symptoms: {test['symptoms']}")
        print(f"Response: {response}")
