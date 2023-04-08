"""Ask a question to the devbase asistant."""

from langchain.document_loaders import PyPDFLoader  # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings  # for creating embeddings
from langchain.vectorstores import Chroma  # for the vectorization part
from langchain.chains import ConversationalRetrievalChain  # for chatting with the pdf
from langchain.llms import OpenAI  # the LLM model we'll use (CHatGPT)
# the chat model we'll use (ChatGPT)
from langchain.chat_models import ChatOpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask("app")
CORS(app)
openai_api_key = "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"
persist_directory = 'db_bluon'
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)
pdf_qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key),
                                               vectordb.as_retriever(), return_source_documents=True)


def process_input(user_input, chat_history):
    result = pdf_qa({"question": user_input, "chat_history": chat_history})
    return result["answer"]


@app.post('/query')
def query():
    data = request.get_json()

    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid input"}), 400

    history = data.get("history", [])

    chat_history = []
    for item in history:
        chat_history.append((item["message"], item["response"]))

    response = process_input(data["message"], chat_history)

    # Do something with the data here, for example:
    processed_data = {"response": response}

    return jsonify(processed_data)
