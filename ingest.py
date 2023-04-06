"""This is the logic for ingesting Notion data into LangChain."""
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

# Here we load in the data in the format that Notion exports it in.
ps = list(Path("Devbase_Coda/").glob("**/*.pdf"))

data = []
sources = []

# Iterate through each PDF file
for i, p in enumerate(ps):
    # Load the PDF file using the PyPDFLoader
    loader = PyPDFLoader(str(p))
    pages = loader.load_and_split()
    # Combine the text from all the pages and append it to the data list
    data.extend(pages)

    # Append the file path to the sources list
    # sources.extend([{"source": sources[i]}] * len(p))

print(len(data), len(sources))
# # Here we split the documents, as needed, into smaller chunks.
# # We do this due to the context limits of the LLMs.
# docs = []
# metadatas = []
# for i, d in enumerate(data):
#     docs.extend(d)
#     metadatas.extend([{"source": sources[i]}] * len(d))
# print(len(docs), len(metadatas))

# # Here we create a vector store from the documents and save it to disk.
# store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
# faiss.write_index(store.index, "devbase.index")
# store.index = None
# with open("faiss_store_devbase.pkl", "wb") as f:
#     pickle.dump(store, f)


embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(data, embedding=embeddings,
                                 persist_directory=".")
vectordb.persist()
