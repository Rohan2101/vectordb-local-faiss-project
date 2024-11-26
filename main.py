import os
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
load_dotenv()
if __name__=="__main__":
    print("hi")
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama-3.1-70b-versatile",
    )
    pdf_path="/Users/rajeevsingh/PycharmProjects/pythonProject1/reAct.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=10,separator="\n")
    docs = text_splitter.split_documents(documents=documents)
    embeddings = PineconeEmbeddings(model="multilingual-e5-large")
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectorstore.save_local("faiss_index_react")
    new_vectorstore = FAISS.load_local(
        "faiss_index_react",embeddings,allow_dangerous_deserialization=True
    )
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm,retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(),combine_docs_chain
    )
    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(res["answer"])
