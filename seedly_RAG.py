'''
Stack:
Langchain framework for LLM development. FAISS as vectorstore and FastEmbed model as embedding function. Pickle to store vectorstore.
'''
import pickle
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# initialize LLM
llm =  ChatGroq(
    temperature=0,
    model="llama3-70b-8192"
)

# Loading vector store and retrieving docs
with open("store", 'rb') as f:
    pkl = pickle.load(f)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.deserialize_from_bytes(
    embeddings=embeddings, serialized=pkl, allow_dangerous_deserialization=True
) 
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# LLM Prompts
CONDENSE_CONTEXT_PROMPT = PromptTemplate.from_template("""Summarise the context provided to be shorter in length, but remain understadable to you. 
Context : {context}
""")

RAG_PROMPT_CUSTOM = PromptTemplate.from_template(
"""Use the information in context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {improved_context}
Question: {question}
Helpful Answer:"""
)

# Function to join retireved docs
def format_docs(docs):
    context = "\n\n".join(doc.page_content for doc in docs)
    print('Context:', context, '\n\n')
    return context

# Langchain LLM Chains
condense_chain = (
    {"context": retriever | format_docs} 
    | CONDENSE_CONTEXT_PROMPT 
    | llm 
    | StrOutputParser())

rag_chain = (
    {"question": RunnablePassthrough(), "improved_context": condense_chain}
    | RAG_PROMPT_CUSTOM
    | llm
    | StrOutputParser()
)

# Model Output
print(rag_chain.invoke("What should I consider when buying insurance?"))