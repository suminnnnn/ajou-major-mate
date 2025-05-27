import os
import re
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from app.domains.base_ingestor import BaseIngestor

class CourseIngestor(BaseIngestor):
    def ingest(self, data_path: str) -> list[Document]:
        docs: list[Document] = []
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        full_path = os.path.join(base_dir, data_path)
        
        code_pattern = r"(?m)(?=^[A-Z]{3,4}\d{3,4})" 

        for filename in os.listdir(full_path):
            if not filename.lower().endswith(".pdf"):
                continue

            path = os.path.join(full_path, filename)
            department = os.path.splitext(filename)[0]
            print(f"📄 Loading {filename} (department={department})")

            loader = PDFPlumberLoader(path)
            pages = loader.load()
            full_text = "\n".join(page.page_content for page in pages)

            chunks = re.split(code_pattern, full_text)
            print(f"  ▶ Split into {len(chunks)} chunks")

            for idx, chunk in enumerate(chunks):
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source_file": filename,
                        "department": department,
                        "chunk_index": idx
                    }
                ))

        return docs
