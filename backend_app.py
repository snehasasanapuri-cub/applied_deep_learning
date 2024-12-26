from dotenv import load_dotenv
import os
import nltk
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain



# Load environment variables
load_dotenv(dotenv_path="D:/Semester-3/Applied Deep Learning/Langchain/RAG_PROJECT/.env")
nltk.download('punkt')  # Ensure punkt tokenizer is downloaded

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")


# Initialize Pinecone
index_name = "hybrid-search-langchain-pinecone"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create Pinecone index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of dense vectors
        metric='dotproduct',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(index_name)

# Define embeddings and BM25 encoder
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
bm25_encoder = BM25Encoder().default()

# Initialize retriever
retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)

# Initialize the LLM (using GPT-4 or GPT-3.5)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  # Adjust temperature as needed

prompt_template = """
{context}

Question: {question}

Answer the question based on the context above as accurately as possible.
"""

prompt = PromptTemplate(
    input_variables=["context", "question"], 
    template=prompt_template
)

# Extract text from a research paper using PyPDF2
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Split text into sentences
def split_into_sentences(text):
    return sent_tokenize(text)

# Index research paper into Pinecone
def index_research_paper(pdf_path):
    # Clear existing index
    index.delete(delete_all=True)

    # Process the uploaded file
    research_text = extract_text_from_pdf(pdf_path)
    sentences = split_into_sentences(research_text)

    # Fit BM25 encoder and add sentences to retriever
    bm25_encoder.fit(sentences)
    retriever.add_texts(sentences)

# Chatbot query function
# Chatbot query function
def chatbot_query(question):
    query = question.lower().strip()

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        return "No relevant context found for the query."

    # Deduplicate and prepare context
    unique_docs = list(set(doc.page_content for doc in retrieved_docs))
    context = "\n".join([f"• {doc}" for doc in unique_docs])
    context = "\n".join([line.strip() for line in context.splitlines() if line.strip()])

    if not context.strip():
        return "The retrieved context is empty. Cannot generate an answer."

    # Construct and run the LLM chain
    try:
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(context=context, question=question)
    except Exception as e:
        return f"An error occurred: {str(e)}"
