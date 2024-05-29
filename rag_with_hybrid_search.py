import os
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from dotenv import load_dotenv


load_dotenv()


HF_TOKEN = os.getenv('HF_TOKEN')

file_path = "deeplearningwithpython.pdf"
data_file = UnstructuredPDFLoader(file_path)
docs = data_file.load()

# create chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=2500,
                                          chunk_overlap=250)
chunks = splitter.split_documents(docs)

embedding=HuggingFaceEmbeddings()

# Vector store with the selected embedding model
vectorstore = Chroma.from_documents(chunks, embedding)

vectorstore_retreiver = vectorstore.as_retriever(search_kwargs={"k": 3})

keyword_retriever = BM25Retriever.from_documents(chunks)
keyword_retriever.k =  3

ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver,
                                                   keyword_retriever],
                                       weights=[0.5, 0.5])



llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    model_kwargs={"temperature": 0.3,"max_new_tokens":1024},
    huggingfacehub_api_token=HF_TOKEN,
)

template = """
<|system|>>
You are a helpful AI Assistant that follows instructions extremely well.
Use the following context to answer user question.

Think step by step before answering the question. You will get a $100 tip if you provide correct answer.

CONTEXT: {context}
</s>
<|user|>
{query}
</s>
<|assistant|>
"""

query ="The engine of neural networks: gradient-based optimization"

prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

chain = (
    {"context": ensemble_retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)

print(chain.invoke(query))

