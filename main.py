import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def build_coffee_rag():
    print("Loading documents from ./knowledge ...")
    loader = DirectoryLoader('./knowledge/', glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./coffee_vector_db"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    system_prompt = (
        "You are an expert Coffee Roasting Assistant.\n"
        "Answer the user's question using ONLY the provided context below.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

if __name__ == "__main__":
    coffee_bot = build_coffee_rag()
    query = "What drop temperature works best for Brazil beans?"
    response = coffee_bot.invoke({"input": query})
    print("\n--- Answer ---")
    print(response["answer"])
