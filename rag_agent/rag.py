from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import json
from langchain_core.messages import HumanMessage, SystemMessage

# Your existing code for loading and processing documents
urls = ["https://python.langchain.com/docs/integrations/document_loaders/web_base/"]

# Load documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
)
doc_splits = text_splitter.split_documents(docs_list)

# Initialize embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings = OllamaEmbeddings(model="llama3.1")

# Set up the vector store
persist_directory = "db"
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
vector_store.add_documents(doc_splits)

# LLM
local_llm = "llama3.1"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_store.as_retriever()
)

# Router
# Prompt
router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

If the vectorstore contains relevant documents ('Yes'), then choose 'vectorstore' as the datasource.
If the vectorstore does not contain relevant documents ('No'), then choose 'websearch' as the datasource.

Return JSON with a single key, 'datasource', that is either 'websearch' or 'vectorstore'. No other information should be provided.

Examples:

1. Vectorstore contains relevant documents: Yes
   Response: {"datasource": "vectorstore"}

2. Vectorstore contains relevant documents: No
   Response: {"datasource": "websearch"}

Do not include any additional text in your response."""

# Function to check the vectorstore for relevant documents
def check_vectorstore_for_question(question, vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.4}
    )
    docs = retriever.get_relevant_documents(question)
    if docs:
        return "Yes"
    else:
        return "No"

# Router agent function
def router_agent(question):
    vectorstore_info = check_vectorstore_for_question(question, vector_store)
    new_router_instructions = (
        router_instructions + f"\n\nVectorstore contains relevant documents: {vectorstore_info}"
    )
    prompt = [
        SystemMessage(content=new_router_instructions),
        HumanMessage(content="Based on the above, which datasource should be used?")
    ]
    result = llm_json_mode.invoke(prompt)
    
    # Parse the result to ensure it is valid JSON
    try:
        result_content = json.loads(result.content)
        return result_content
    except json.JSONDecodeError:
        return {"datasource": "None"}

# Test the router agent
test_web_search = router_agent("Who is favored to win the NFC Championship game in the 2024 season?")
test_web_search2 = router_agent("Who is our current president?")
test_web_search3 = router_agent("What is google?")

test_vector_store = router_agent("How do I load web pages in langchain?")
test_vector_store2 = router_agent("how to load pdfs in langchain")
test_vector_store3 = router_agent("document loading in langchain")

print("Test Web Search Output:", test_web_search, test_web_search2, test_web_search3)
print("Test Vector Store Output:", test_vector_store, test_vector_store2, test_vector_store3)
