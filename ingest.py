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
# ps = list(Path("bluon/").glob("**/*.pdf"))

loader = PyPDFLoader("bluon/38BRC-14PD.pdf")
data = loader.load()

# # Iterate through each PDF file
# for i, p in enumerate(ps):
#     # Load the PDF file using the PyPDFLoader
#     loader = PyPDFLoader(str(p))
#     pages = loader.load()
#     # Combine the text from all the pages and append it to the data list
#     data.append(pages)
# # Here we split the documents, as needed, into smaller chunks.
# # We do this due to the context limits of the LLMs.
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(data)

# # Here we create a vector store from the documents and save it to disk.
# store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
# faiss.write_index(store.index, "devbase.index")
# store.index = None
# with open("faiss_store_devbase.pkl", "wb") as f:
#     pickle.dump(store, f)

persist_directory = 'db_bluon'
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents, embedding=embeddings,
                                 persist_directory=persist_directory)
vectordb.persist()
