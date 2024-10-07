import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = (add your own)

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini")

# Load the stored vectorstore
vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=OpenAIEmbeddings()
)

# Set up the retriever
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Function to query the RAG system
def query_rag(question):
    return rag_chain.invoke(question)

# Example usage
if __name__ == "__main__":
    question = input("Ask your query?\n")
    answer = query_rag(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")