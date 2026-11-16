import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
load_dotenv()
DATA_PATH = "data"
CHROMA_PATH = "chroma"
LLM_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "models/gemini-embedding-001"
PROMPT = ChatPromptTemplate.from_template(
"""You are an assistant for construction project documents.
Answer using ONLY the context below.
If the answer is not in the context, reply exactly: "I can't find that in the documents."
After each fact, cite its source like [filename, page X].
Context:
{context}
Question: {question}"""
)
def get_embeddings():
return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
def get_llm():
return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
def get_db():
return Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())
def answer(question: str, k: int = 5):
"""Retrieve the top-k chunks and ask the LLM. Returns (answer_text, source_docs)."""
docs = get_db().similarity_search(question, k=k)
context = "\n\n---\n\n".join(
f"[{os.path.basename(d.metadata.get('source', '?'))}, "
f"page {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
for d in docs
)
response = get_llm().invoke(PROMPT.format_messages(context=context, question=question))
return response.content, docs