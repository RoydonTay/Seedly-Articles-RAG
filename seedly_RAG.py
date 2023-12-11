'''
Download quantized llama2 model from cmd-line:
>>> pip3 install huggingface-hub>=0.17.1
>>> huggingface-cli download TheBloke/Llama-2-7b-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
source: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/README.md

Stack:
Langchain framework for LLM development. FAISS as vectorstore and FastEmbed model as embedding function. Pickle to store vectorstore.
'''
import langchain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pickle
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
from langchain.embeddings.fastembed import FastEmbedEmbeddings
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

# User input:
question = str(input("Write your question: "))

# enable logging of processes
langchain.debug = True
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# initialize LLM
llm = LlamaCpp(
    model_path="../llama-2-7b-chat.Q4_K_M.gguf", # Download model into directory first
    temperature=0.1,
    max_tokens=2000,
    top_p=1,
    n_ctx = 800,
    callback_manager=callback_manager,
    verbose=True, 
)

# Loading vector store and retrieving docs
with open("store", 'rb') as f:
    pkl = pickle.load(f)

embeddings = FastEmbedEmbeddings()
db = FAISS.deserialize_from_bytes(
    embeddings=embeddings, serialized=pkl
) 
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# LLM Prompts
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Summarise the context provided to be shorter in length, but remain understadable to you. 
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
    return "\n\n".join(doc.page_content for doc in docs)

# Langchain LLM Chains
condense_chain = {"context": retriever | format_docs} | CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()

rag_chain = (
    {"question": RunnablePassthrough(), "improved_context": condense_chain}
    | RAG_PROMPT_CUSTOM
    | llm
    | StrOutputParser()
)

# Model Output
rag_chain.invoke("What should I consider when getting insurance?")