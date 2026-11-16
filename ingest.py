import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from rag import DATA_PATH, CHROMA_PATH, get_embeddings
def main():
docs = PyPDFDirectoryLoader(DATA_PATH).load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)
if os.path.exists(CHROMA_PATH):
shutil.rmtree(CHROMA_PATH) # rebuild from scratch each time (simple and safe)
Chroma.from_documents(chunks, get_embeddings(), persist_directory=CHROMA_PATH)
print(f"Indexed {len(docs)} pages as {len(chunks)} chunks.")
if __name__ == "__main__":
main()