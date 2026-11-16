import argparse
from rag import answer
parser = argparse.ArgumentParser()
parser.add_argument("question")
args = parser.parse_args()
text, docs = answer(args.question)
print(text)
print("\nRetrieved chunks:")
for d in docs:
print(" -", d.metadata.get("source"), "page", d.metadata.get("page", 0) + 1)
