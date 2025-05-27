import os
from typing import List
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.domains.base_ingestor import BaseIngestor

class DepartmentIntroIngestor(BaseIngestor):
    def ingest(self, data_path: str) -> List[Document]:
        docs: List[Document] = []
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        full_path = os.path.join(base_dir, data_path)
        
        chunk_index = 0

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=200,
            length_function=len
        )

        for filename in os.listdir(full_path):
            if not filename.lower().endswith(".pdf"):
                continue

            path = os.path.join(full_path, filename)
            department = os.path.splitext(filename)[0]
            print(f"📄 Loading {filename} (department={department})")

            loader = PDFPlumberLoader(path)
            pages = loader.load()
            full_text = "\n".join(p.page_content for p in pages)

            chunks = splitter.split_text(full_text)
            for idx, chunk in enumerate(chunks):
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "department": department,
                        "source_file": filename,
                        "chunk_index": chunk_index
                    }
                ))
                chunk_index += 1

        return docs