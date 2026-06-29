from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models import DocumentExtractResponse, ReviewRequest, ReviewResponse
from app.services.document_loader import extract_docx_text
from app.services.llm_reviewer import review_with_optional_llm

router = APIRouter(prefix="/api/reviews", tags=["reviews"])


@router.post("", response_model=ReviewResponse)
async def create_review(request: ReviewRequest) -> ReviewResponse:
    try:
        return review_with_optional_llm(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/extract-docx", response_model=DocumentExtractResponse)
async def extract_docx(file: UploadFile = File(...)) -> DocumentExtractResponse:
    if not file.filename or not file.filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        text = extract_docx_text(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not parse DOCX file.") from exc

    if len(text) < 80:
        raise HTTPException(status_code=400, detail="DOCX does not contain enough reviewable text.")

    return DocumentExtractResponse(filename=file.filename, text=text, character_count=len(text))
