import os
import pdfplumber
import tempfile
from typing import List
from uuid import uuid4
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.domains.base_ingestor import BaseIngestor
from app.utils.s3_uploader import upload_file_to_s3

class CurriculumIngestor(BaseIngestor):
    def ingest(self, data_path: str) -> List[Document]:
        docs: List[Document] = []
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        full_path = os.path.join(base_dir, data_path)
        
        chunk_index = 0

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )

        for filename in os.listdir(full_path):
            if not filename.lower().endswith(".pdf"):
                continue

            path = os.path.join(full_path, filename)
            department = os.path.splitext(filename)[0]
            print(f"📄 Loading {filename} (department={department})")

            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    
                    tables = page.extract_tables()
                    for tbl_idx, table in enumerate(tables):
                        rows = ["\t".join([cell or "" for cell in row]) for row in table]
                        table_text = "\n".join(rows)

                        image = page.to_image(resolution=300)
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                            image.save(tmp.name)
                            s3_key = f"{department}/{filename}_p{page_num}_t{tbl_idx}_{uuid4().hex}.png"
                            image_url = upload_file_to_s3(tmp.name, s3_key)
                        os.remove(tmp.name)

                        docs.append(Document(
                            page_content=table_text,
                            metadata={
                                "type": "table",
                                "department": department,
                                "source_file": filename,
                                "chunk_index": chunk_index,
                                "page": page_num,
                                "table_index": tbl_idx,
                                "image_url": image_url,
                                # "table_summary": "",
                            }
                        ))
                        chunk_index += 1

                    text = page.extract_text() or ""
                    if text.strip():
                        text_chunks = splitter.split_text(text)
                        for idx, chunk in enumerate(text_chunks):
                            docs.append(Document(
                                page_content=chunk,
                                metadata={
                                    "type": "text",
                                    "department": department,
                                    "source_file": filename,
                                    "chunk_index": chunk_index,
                                    "page": page_num,
                                    "text_index": idx
                                }
                            ))
                            chunk_index += 1

        return docs