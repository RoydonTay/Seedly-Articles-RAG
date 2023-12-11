"""
Script to create vectorstore for webscrapped documents and write to pickle file.
"""
from langchain.vectorstores import FAISS
from langchain.embeddings.fastembed import FastEmbedEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

embeddings = FastEmbedEmbeddings()

path = '/home/cowboygarage/seedly_scrape/Scraped'
loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
)
docs = text_splitter.split_documents(documents)

db = FAISS.from_documents(docs, embeddings)

# Export index to pickle file
pkl = db.serialize_to_bytes()  # serializes the faiss

with open('store', 'wb') as f:
    pickle.dump(pkl, f)