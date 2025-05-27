from fastapi import APIRouter, Query, HTTPException
from app.vectorstore.qdrant import add_documents, delete_documents
from app.domains.course.ingestor import CourseIngestor
from app.domains.curriculum.ingestor import CurriculumIngestor
from app.domains.department_intro.ingestor import DepartmentIntroIngestor
from app.domains.employment_status.ingestor import EmploymentStatusIngestor

router = APIRouter()

domain_map = {
    "course": CourseIngestor,
    "curriculum": CurriculumIngestor,
    "department_intro": DepartmentIntroIngestor,
    "employment_status": EmploymentStatusIngestor,
}

@router.post("/embed")
def embed_documents(domain: str = Query(...)):
    Ingestor = domain_map.get(domain)
    if not Ingestor:
        raise HTTPException(status_code=400, detail=f"⚠잘못된 파라미터 요청입니다. : {domain}")

    ingestor = Ingestor()
    docs = ingestor.ingest(data_path=f"scripts/{domain}/data")
    add_documents(domain, docs)
    return {"message": f"✅ '{domain}' 도메인의 문서 {len(docs)}개 업로드 완료"}

@router.delete("/embed")
def delete_domain_documents(domain: str = Query(...)):
    if domain not in domain_map:
        raise HTTPException(status_code=400, detail=f"⚠잘못된 파라미터 요청입니다. : {domain}")
    delete_documents(domain)
    return {"message": f"🗑️ '{domain}' 도메인의 문서가 모두 삭제되었습니다"}