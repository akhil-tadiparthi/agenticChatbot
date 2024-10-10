from langchain_ollama import ChatOllama
import os
import getpass
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage


class RAGagent:
    def __init__(self) -> None:
        self.local_llm = "llama3.1"
        self.llm = ChatOllama(model=self.local_llm, temperature=0)
        self.llm_json_mode = ChatOllama(model=self.local_llm, temperature=0, format="json")
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]
    
        # Load documents
        docs = [WebBaseLoader(url).load() for url in self.urls]
        docs_list = [item for sublist in docs for item in sublist]

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Add to vectorDB
        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=HuggingFaceEmbeddings(),
        )

        # Create retriever
        self.retriever = vectorstore.as_retriever(k=3)
        self.question = "What is Chain of thought prompting?"



    def retrieval_grader(self):
        doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

        # Grader prompt
        doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

        This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

        Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

        # Test
        docs = self.retriever.invoke(question)
        doc_txt = docs[1].page_content
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=doc_txt, question=question
        )
        result = self.llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        json.loads(result.content)


    def generator(self):
        rag_prompt = """You are an assistant for question-answering tasks. 

        Here is the context to use to answer the question:

        {context} 

        Think carefully about the above context. 

        Now, review the user question:

        {question}

        Provide an answer to this questions using only the above context. 

        Use three sentences maximum and keep the answer concise.

        Answer:"""


        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        # Test
        docs = self.retriever.invoke(self.question)
        docs_txt = format_docs(docs)
        rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=self.question)
        generation = self.llm.invoke([HumanMessage(content=rag_prompt_formatted)])
        print(generation.content)



