import os
import tempfile
import re
from typing import List, Dict, Any
from uuid import uuid4
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageDocumentParseLoader
from app.domains.base_ingestor import BaseIngestor
from app.utils.s3_uploader import upload_file_to_s3
from bs4 import BeautifulSoup

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

            file_path = os.path.join(full_path, filename)
            department = os.path.splitext(filename)[0]
            print(f"📄 Loading {filename} (department={department})")

            # UpstageDocumentParseLoader 사용
            loader = UpstageDocumentParseLoader(
                file_path=file_path,
                split="page"  # 페이지별로 분할
            )
            
            parsed_docs = loader.load()
            
            for page_num, doc in enumerate(parsed_docs):
                page_content = doc.page_content
                
                # 순차적으로 내용 처리 (테이블과 텍스트를 원본 순서대로)
                sequential_data = self._process_content_sequentially(
                    page_content, department, filename, page_num, chunk_index, splitter
                )
                docs.extend(sequential_data["documents"])
                chunk_index = sequential_data["next_chunk_index"]

        return docs

    def _process_content_sequentially(self, html_content: str, department: str, filename: str, 
                                    page_num: int, start_chunk_index: int, splitter) -> dict:
        """HTML 내용을 원본 순서대로 처리하고 테이블 관련 텍스트를 합쳐서 저장"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 디버깅용 플래그
        debug_mode = (department == "국방디지털융합학과")
        
        if debug_mode:
            print(f"\n🔍 [DEBUG] Processing page {page_num} of {department}")
        
        # 1단계: 모든 요소를 순서대로 추출하고 분류
        elements = self._extract_elements_sequentially(soup, debug_mode)
        
        # 2단계: 테이블 주변 텍스트 병합
        merged_chunks = self._merge_table_with_context(elements, debug_mode)
        
        # 3단계: Document 객체 생성
        documents = []
        chunk_index = start_chunk_index
        
        for chunk_data in merged_chunks:
            if chunk_data["type"] == "table_with_context":
                # 테이블 + 컨텍스트 청크
                combined_content = ""
                
                # 이전 텍스트 추가
                if chunk_data["prev_text"]:
                    combined_content += f"[제목/설명]\n{chunk_data['prev_text']}\n\n"
                
                # 테이블 추가
                combined_content += f"[테이블]\n{chunk_data['table_content']}\n\n"
                
                # 다음 텍스트 추가
                if chunk_data["next_text"]:
                    combined_content += f"[부가설명]\n{chunk_data['next_text']}"
                
                if debug_mode:
                    print(f"📊 [DEBUG] Table chunk {chunk_index}:")
                    print(f"   - Prev text: {bool(chunk_data['prev_text'])}")
                    print(f"   - Table: ✓")
                    print(f"   - Next text: {bool(chunk_data['next_text'])}")
                    print(f"   - Combined length: {len(combined_content)}")
                
                documents.append(Document(
                    page_content=combined_content.strip(),
                    metadata={
                        "type": "table_with_context",
                        "department": department,
                        "source_file": filename,
                        "chunk_index": chunk_index,
                        "page": page_num,
                        "table_index": chunk_data["table_index"],
                        "image_url": chunk_data["image_url"],
                        "has_prev_text": bool(chunk_data["prev_text"]),
                        "has_next_text": bool(chunk_data["next_text"])
                    }
                ))
                chunk_index += 1
                
            elif chunk_data["type"] == "text":
                # 일반 텍스트 청크 (테이블과 병합되지 않은 것)
                text_chunks = splitter.split_text(chunk_data["content"])
                
                for idx, chunk in enumerate(text_chunks):
                    if chunk.strip():
                        if debug_mode:
                            print(f"📝 [DEBUG] Text chunk {chunk_index}: {len(chunk)} chars")
                        
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                "type": "text",
                                "department": department,
                                "source_file": filename,
                                "chunk_index": chunk_index,
                                "page": page_num,
                                "text_index": idx,
                                "element_tag": chunk_data.get("element_tag", "unknown")
                            }
                        ))
                        chunk_index += 1
        
        if debug_mode:
            print(f"✅ [DEBUG] Page {page_num} processed: {len(documents)} chunks created")
        
        return {
            "documents": documents,
            "next_chunk_index": chunk_index
        }

    def _extract_elements_sequentially(self, soup: BeautifulSoup, debug_mode: bool) -> List[Dict[str, Any]]:
        """HTML에서 요소들을 순서대로 추출하여 리스트로 반환"""
        elements = []
        
        for element in soup.find_all(['table', 'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if element.name == 'table':
                markdown_table = self._convert_table_to_markdown(element)
                if markdown_table.strip():
                    elements.append({
                        "type": "table",
                        "content": markdown_table,
                        "element": element,
                        "original_index": len(elements)
                    })
                element.decompose()
            else:
                text_content = element.get_text(strip=True)
                if text_content and len(text_content) > 10:
                    elements.append({
                        "type": "text",
                        "content": text_content,
                        "element_tag": element.name,
                        "original_index": len(elements)
                    })
                element.decompose()
        
        # 남은 텍스트 처리
        remaining_text = soup.get_text(separator='\n', strip=True)
        remaining_text = re.sub(r'\n\s*\n', '\n\n', remaining_text)
        if remaining_text.strip():
            elements.append({
                "type": "text",
                "content": remaining_text,
                "element_tag": "remaining",
                "original_index": len(elements)
            })
        
        if debug_mode:
            print(f"🔄 [DEBUG] Extracted {len(elements)} elements:")
            for i, elem in enumerate(elements):
                print(f"   {i}: {elem['type']} ({len(elem['content'])} chars)")
        
        return elements

    def _merge_table_with_context(self, elements: List[Dict[str, Any]], debug_mode: bool) -> List[Dict[str, Any]]:
        """테이블과 주변 텍스트를 병합"""
        merged_chunks = []
        used_indices = set()
        
        for i, element in enumerate(elements):
            if element["type"] == "table" and i not in used_indices:
                # 이전 텍스트 찾기
                prev_text = ""
                prev_idx = i - 1
                if prev_idx >= 0 and prev_idx not in used_indices and elements[prev_idx]["type"] == "text":
                    prev_text = elements[prev_idx]["content"]
                    used_indices.add(prev_idx)
                
                # 다음 텍스트 찾기
                next_text = ""
                next_idx = i + 1
                if next_idx < len(elements) and next_idx not in used_indices and elements[next_idx]["type"] == "text":
                    next_text = elements[next_idx]["content"]
                    used_indices.add(next_idx)
                
                # 테이블 이미지 생성
                image_url = ""
                if "element" in element:
                    image_url = self._create_table_image(
                        element["element"], "", "", 0, len(merged_chunks)
                    )
                
                merged_chunks.append({
                    "type": "table_with_context",
                    "table_content": element["content"],
                    "prev_text": prev_text,
                    "next_text": next_text,
                    "table_index": len([c for c in merged_chunks if c["type"] == "table_with_context"]),
                    "image_url": image_url
                })
                
                used_indices.add(i)
                
                if debug_mode:
                    print(f"🔗 [DEBUG] Merged table {i}:")
                    print(f"   - Prev text ({prev_idx}): {len(prev_text)} chars")
                    print(f"   - Table: {len(element['content'])} chars")
                    print(f"   - Next text ({next_idx}): {len(next_text)} chars")
        
        # 사용되지 않은 텍스트 요소들 추가
        for i, element in enumerate(elements):
            if i not in used_indices and element["type"] == "text":
                merged_chunks.append({
                    "type": "text",
                    "content": element["content"],
                    "element_tag": element.get("element_tag", "unknown")
                })
                
                if debug_mode:
                    print(f"📄 [DEBUG] Standalone text {i}: {len(element['content'])} chars")
        
        return merged_chunks

    def _convert_table_to_markdown(self, table) -> str:
        """HTML 테이블을 마크다운 형식으로 변환"""
        rows = []
        
        # 헤더 처리
        thead = table.find('thead')
        if thead:
            header_rows = thead.find_all('tr')
            for row in header_rows:
                cells = row.find_all(['th', 'td'])
                row_data = []
                for cell in cells:
                    colspan = int(cell.get('colspan', 1))
                    rowspan = int(cell.get('rowspan', 1))
                    cell_text = cell.get_text(strip=True)
                    
                    # colspan 처리
                    row_data.append(cell_text)
                    for _ in range(colspan - 1):
                        row_data.append("")
                
                rows.append("| " + " | ".join(row_data) + " |")
            
            # 헤더 구분선 추가
            if rows:
                separator = "| " + " | ".join(["---"] * len(row_data)) + " |"
                rows.append(separator)
        
        # 바디 처리
        tbody = table.find('tbody')
        if tbody:
            body_rows = tbody.find_all('tr')
        else:
            # tbody가 없는 경우 table에서 직접 tr 찾기
            body_rows = table.find_all('tr')
            if thead:  # 헤더가 있었다면 헤더 행들은 제외
                header_row_count = len(thead.find_all('tr'))
                body_rows = body_rows[header_row_count:]
        
        for row in body_rows:
            cells = row.find_all(['td', 'th'])
            row_data = []
            for cell in cells:
                colspan = int(cell.get('colspan', 1))
                cell_text = cell.get_text(strip=True)
                
                row_data.append(cell_text)
                for _ in range(colspan - 1):
                    row_data.append("")
            
            if row_data:  # 빈 행이 아닌 경우만 추가
                rows.append("| " + " | ".join(row_data) + " |")
        
        return "\n".join(rows)

    def _create_table_image(self, table, department: str, filename: str, page_num: int, tbl_idx: int) -> str:
        """테이블 이미지를 생성하고 S3에 업로드 (옵션 - 필요시 구현)"""
        # 현재는 빈 URL 반환, 필요시 실제 이미지 생성 로직 구현
        # 예: HTML to Image 라이브러리 사용하여 테이블 이미지 생성
        try:
            # 간단한 구현 예시 (실제로는 더 정교한 이미지 생성 필요)
            s3_key = f"{department}/{filename}_p{page_num}_t{tbl_idx}_{uuid4().hex}.png"
            
            # 실제 이미지 생성 로직은 생략하고 placeholder URL 반환
            # 필요시 wkhtmltopdf, playwright, selenium 등을 사용하여 구현
            image_url = f"https://placeholder-url/{s3_key}"
            
            return image_url
        except Exception as e:
            print(f"⚠️ Failed to create table image: {e}")
            return ""