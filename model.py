from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA #This is just a retrieval chain, for chat history use conversational retrieval chain
from langchain.chains import ConversationalRetrievalChain

from typing import Dict, Any
import torch

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import ingest
import os

# LLMChain: This chain uses a Language Model for generating responses to queries or prompts. 
# It can be used for various tasks such as chatbots, summarization, and more

# StuffDocumentsChain: This is the default QA chain used in the RetrievalQAChain. It processes the retrieved documents 
# and generates answers to questions based on the content of the documents

#RetrievalQAChain: This chain combines a Retriever and a QA chain. 
#It is used to retrieve documents from a Retriever and then use a QA chain to answer a question based on the retrieved documents

# the RetrievalQAChain is used with a VectorStore as the Retriever and the default StuffDocumentsChain as the QA chain.

#From what I understand:
# RetrievalQAChain: retrieves the vector based on the prompt
# StuffDocumentsChain: Converts the vector into the answer
# LLMChain: Uses the answer to generate a response to the prompt

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up an answer.
Be empathetic, sympathetic, and kind in your responses.


Context: {chat_history} 
{context}

Question: {question}

Only returns the helpful answer below and nothing else.
Helpful answer:
"""

custom_template = """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question. 
Preserve the original question in the answer sentiment during rephrasing.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT_CUSTOM = PromptTemplate(template=custom_template, input_variables=["question", "chat_history"])


def setCustomPrompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question", "chat_history"])
    return prompt

def loadLLM():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama", max_new_tokens=512, temperature=0)
    return llm

def retrievalQAChain(llm, prompt, db):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type="stuff", 
        retriever= db.as_retriever(search_kwargs={"k": 1}), 
        combine_docs_chain_kwargs={"prompt": prompt}, 
        return_source_documents = True, 
        verbose=True,
        # condense_question_prompt=CONDENSE_QUESTION_PROMPT_CUSTOM,
        rephrase_question = False
        )
    #search_kwargs={"k": 2} means 2 searches
    #return_source_documents = True means don't use base knowledge use only knowledge we provided
    return qa_chain

def qaBot():
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs={"device": DEVICE})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)

    llm = loadLLM()
    qa_prompt = setCustomPrompt()
    qa = retrievalQAChain(llm, qa_prompt, db)

    return qa

def finalResult(query):
    qa_result = qaBot()
    chat_history = []
    # Will be query if using RetrievalQA, question for ConversationalQA
    response = qa_result({'chat_history': chat_history, 'question': query})
    print()
    return response


def askBot(query, qa_result, chat_history):
    # Will be query if using RetrievalQA, question for ConversationalQA
    response = qa_result({'chat_history': chat_history, 'question': query})
    print()
    return response

app = Flask(__name__)
app.static_folder = "static"
CORS(app)


qa_result = None
if os.path.exists(DB_FAISS_PATH):
    qa_result = qaBot()
chat_history = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    global qa_result
    prompt = request.args.get("msg")
    answer = askBot(prompt, qa_result, chat_history)
    data = {
        "answer" : answer["answer"],
        "source" : answer["source_documents"][0].metadata["source"]
    }
    return jsonify(data)

@app.route("/get2")
def get_bot_response2():
    global qa_result
    link = request.args.get("msg")
    data = {}
    if ingest.addLink(link):
        ingest.createVectorDB(link)
        data = {
            "answer" : "Success!"
        }
    else:
        data = {
            "answer" : "Failed :("
        }
    qa_result = qaBot()
    return data

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

    #Need to do Add button before send if the vector db path does not exist