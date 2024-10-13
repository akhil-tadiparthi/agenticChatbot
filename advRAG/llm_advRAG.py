import os
from langchain.chains import RetrievalQA
from langchain_unstructured import UnstructuredLoader
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema import Document

class RAG:
    def __init__(self):
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model="llama3.1")
        #self.embeddings = HuggingFaceEmbeddings()

        # Set up the vector store
        persist_directory = "chromaDB"
        self.vector_store = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)

        # Initialize the language model
        self.llm = Ollama(model="llama3.1")

        # Initialize the files and urls
        self.file_paths = ["/Users/akhiltadiparthi/Documents/GitHub/agenticChatbot/advRAG/Enhancing_Lip_Reading_Techniques_with_LipNet_for_Improved_Sentence_Recognition.pdf"]
        self.urls = ["https://python.langchain.com/docs/integrations/document_loaders/web_base/"]

        # Add documents and urls to vector store
        self.add_documents_to_vectorstore()
        self.add_urls_to_vectorstore()

    def add_documents_to_vectorstore(self):
        # Load documents
        # docs = [UnstructuredLoader(file_path).load() for file_path in self.file_paths]
        # docs_list = [item for sublist in docs for item in sublist]

        # # Split documents
        # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        #     chunk_size=1000, chunk_overlap=200
        # )
        # doc_splits = text_splitter.split_documents(docs_list)

        # # Add to vector store
        # self.vector_store.add_documents(doc_splits)
        # print("Files added to ChromaDB")

        def filter_complex_metadata(document):
            simplified_metadata = {}
            for key, value in document.metadata.items():
                if isinstance(value, (int, float, str)):
                    simplified_metadata[key] = value
            return Document(page_content=document.page_content, metadata=simplified_metadata)
        
        docs = [UnstructuredLoader(file_path).load() for file_path in self.file_paths]
        docs_list = [item for sublist in docs for item in sublist]

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        texts = text_splitter.split_documents(docs_list)

        # Filter out complex metadata and create Document instances
        cleaned_texts = [filter_complex_metadata(text) for text in texts]

        # Add to vector store
        self.vector_store.add_documents(cleaned_texts)
        print("Files added to ChromaDB")

        
    def add_urls_to_vectorstore(self):
        # Load urls
        docs = [WebBaseLoader(url).load() for url in self.urls]
        docs_list = [item for sublist in docs for item in sublist]

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Add to vector store
        self.vector_store.add_documents(doc_splits)
        print("Urls added to ChromaDB")

    def perform_RAG(self, user_prompt):
        # Create the RetrievalQA chain
        chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.vector_store.as_retriever()
        )
        llm_response = chain.invoke({"query": user_prompt})
        return llm_response


if __name__ == "__main__":
    question = input("Enter your question (or type 'exit' to quit): ")
    rag_agent = RAG()
    rag_response = rag_agent.perform_RAG(question)
    print(rag_response)