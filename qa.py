"""Ask a question to the notion database."""
# import faiss
# from langchain import OpenAI
# from langchain.chains import VectorDBQAWithSourcesChain
# import pickle
# import argparse

# parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
# parser.add_argument('question', type=str, help='The question to ask the notion DB')
# args = parser.parse_args()

# # Load the LangChain.
# index = faiss.read_index("docs.index")

# with open("faiss_store.pkl", "rb") as f:
#     store = pickle.load(f)

# store.index = index
# chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)
# result = chain({"question": args.question})
# print(f"Answer: {result['answer']}")
# print(f"Sources: {result['sources']}")

from langchain.document_loaders import PyPDFLoader  # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings  # for creating embeddings
from langchain.vectorstores import Chroma  # for the vectorization part
from langchain.chains import ConversationalRetrievalChain  # for chatting with the pdf
from langchain.llms import OpenAI  # the LLM model we'll use (CHatGPT)
# the chat model we'll use (ChatGPT)
from langchain.chat_models import ChatOpenAI

persist_directory = 'db'

embedding = OpenAIEmbeddings()

vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

pdf_qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
                                               vectordb.as_retriever(), return_source_documents=True)
query = "What happens if I feel sick? Must I to work?"
result = pdf_qa({"question": query, "chat_history": ""})
print("Answer:")
print(result["answer"])
