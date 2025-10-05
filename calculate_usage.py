# calculate_usage.py
import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredHTMLLoader

def calculate_stats():
    """Calculates the number of documents and total characters to be processed."""
    filings_dir = './data/unstructured/10k/'
    if not os.path.exists(filings_dir):
        print(f"Directory not found: {filings_dir}")
        return

    loader = DirectoryLoader(
        filings_dir,
        glob="**/*.html",
        loader_cls=UnstructuredHTMLLoader,
        silent_errors=True
    )
    documents = loader.load()

    target_years = [str(y) for y in range(2020, 2026)]
    docs_to_process = []
    for doc in documents:
        filename = os.path.basename(doc.metadata.get('source', ''))
        if any(year in filename for year in target_years):
            docs_to_process.append(doc)

    num_docs = len(docs_to_process)
    total_chars = sum(len(doc.page_content) for doc in docs_to_process)

    print("--- Usage Estimation Stats ---")
    print(f"Number of documents to process: {num_docs}")
    print(f"Total characters to embed: {total_chars:,}")
    print("\n--- LLM Extraction Cost Estimate ---")
    print(f"Total Input Tokens: {num_docs * 5250:,}")
    print(f"Total Output Tokens: {num_docs * 250:,}")
    print("\nPlug these numbers into your cloud provider's pricing calculator.")

if __name__ == "__main__":
    calculate_stats()