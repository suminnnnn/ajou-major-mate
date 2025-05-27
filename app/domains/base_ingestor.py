from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class BaseIngestor(ABC):
    """
    PDF/TXT 등에서 Document 객체 리스트를 추출
    """

    @abstractmethod
    def ingest(self, data_path: str) -> List[Document]:
        """
        지정된 경로에서 데이터를 읽고 Document 리스트를 반환
        """
        pass