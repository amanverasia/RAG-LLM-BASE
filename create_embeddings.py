import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = (add your own)

# Load and process the document
loader = UnstructuredMarkdownLoader("data/random_stuff.md")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create and store the embeddings
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

print("Embeddings created and stored successfully.")