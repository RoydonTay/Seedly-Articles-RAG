"""
Script to create vectorstore for webscrapped documents and write to pickle file.
"""
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

embedder = SpacyEmbeddings(model_name="en_core_web_sm")

path = 'Scraped'
loader = DirectoryLoader(
    path,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'},
    show_progress=True
)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
)
docs = text_splitter.split_documents(documents)

db = FAISS.from_documents(docs, embedder)

# Export index to pickle file
pkl = db.serialize_to_bytes()  # serializes the faiss

with open('store', 'wb') as f:
    pickle.dump(pkl, f)