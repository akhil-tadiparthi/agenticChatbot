import os
from langchain.chains import RetrievalQA
from langchain_unstructured import UnstructuredLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema import Document


def filter_complex_metadata(document):
    # Process the metadata dictionary
    simplified_metadata = {}
    for key, value in document.metadata.items():
        # Simplify complex metadata structures
        if isinstance(value, (int, float, str)):
            simplified_metadata[key] = value
        # Add more conditions as needed
    # Return a new Document with simplified metadata
    return Document(page_content=document.page_content, metadata=simplified_metadata)


def rag(user_question):
    # Load the document
    file_path = "/Users/akhiltadiparthi/Documents/GitHub/agenticChatbot/advRAG/Enhancing_Lip_Reading_Techniques_with_LipNet_for_Improved_Sentence_Recognition.pdf"
    # Load the document
    loader = UnstructuredLoader(file_path)
    docs = loader.load()

    # Split the document into manageable chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(docs)

    # Filter out complex metadata and create Document instances
    cleaned_texts = [filter_complex_metadata(text) for text in texts]

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings()

    # Set up the vector store
    persist_directory = "db"
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    vector_store.add_documents(cleaned_texts)

    # Initialize the language model
    llm = Ollama(model="llama3.1")

    # Create the RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever()
    )

    # Ask a question
    question = user_question
    result = chain.invoke({"query": question})
    return result
