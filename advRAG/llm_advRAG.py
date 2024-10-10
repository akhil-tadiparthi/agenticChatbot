import os
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

file_path = "/Users/akhiltadiparthi/Documents/GitHub/agenticChatbot/advRAG/Enhancing_Lip_Reading_Techniques_with_LipNet_for_Improved_Sentence_Recognition.pdf"
loader = UnstructuredFileLoader(file_path)
docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=2000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(texts, embeddings)

llm = Ollama(model="llama3.1")

chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever()
)

question = "Can you please summarize the document"
result = chain.invoke({"query": question})

print(result['result'])