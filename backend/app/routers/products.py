from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models import (
    ContextDocument,
    ContextDocumentCreate,
    Product,
    ProductCreate,
    ProductKnowledgeBase,
    ProductReviewRequest,
    ReviewRequest,
    ReviewResponse,
)
from app.services.document_loader import extract_docx_text
from app.services.knowledge_base import (
    add_context,
    add_context_file,
    build_review_context,
    create_product,
    get_knowledge_base,
    get_product,
    list_contexts,
    list_products,
)
from app.services.llm_reviewer import review_with_optional_llm

router = APIRouter(prefix="/api/products", tags=["products"])


@router.post("", response_model=Product)
async def create_product_endpoint(payload: ProductCreate) -> Product:
    return create_product(payload)


@router.get("", response_model=list[Product])
async def list_products_endpoint() -> list[Product]:
    return list_products()


@router.get("/{product_id}", response_model=ProductKnowledgeBase)
async def get_product_endpoint(product_id: str) -> ProductKnowledgeBase:
    try:
        return get_knowledge_base(product_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Product not found.") from exc


@router.post("/{product_id}/contexts", response_model=ContextDocument)
async def add_context_endpoint(product_id: str, payload: ContextDocumentCreate) -> ContextDocument:
    try:
        return add_context(product_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Product not found.") from exc


@router.get("/{product_id}/contexts", response_model=list[ContextDocument])
async def list_contexts_endpoint(product_id: str) -> list[ContextDocument]:
    try:
        return list_contexts(product_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Product not found.") from exc


@router.post("/{product_id}/contexts/upload", response_model=ContextDocument)
async def upload_context_endpoint(product_id: str, file: UploadFile = File(...)) -> ContextDocument:
    try:
        get_product(product_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Product not found.") from exc

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    filename = file.filename or "context"
    if not _is_supported_file(filename):
        raise HTTPException(status_code=400, detail="Only .docx, .txt, .md, .pdf, .pptx, .xlsx, .xls, .html, and .csv files are supported.")
    return add_context_file(product_id, filename, payload, source_type="upload")


@router.post("/{product_id}/reviews", response_model=ReviewResponse)
async def review_product_prd_endpoint(product_id: str, payload: ProductReviewRequest) -> ReviewResponse:
    try:
        context = build_review_context(product_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Product not found.") from exc

    request = ReviewRequest(
        feature_name=payload.feature_name,
        platform=payload.platform,
        prd_text=payload.prd_text,
        attachments=payload.attachments,
        organizational_context=context,
        token_budget=payload.token_budget,
        config=payload.config,
    )
    try:
        return review_with_optional_llm(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _extract_supported_file(filename: str, payload: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".docx"):
        text = extract_docx_text(payload)
    elif lower.endswith(".txt") or lower.endswith(".md"):
        text = payload.decode("utf-8", errors="replace").strip()
    else:
        raise HTTPException(status_code=400, detail="Only .docx, .txt, and .md files are supported.")
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="Uploaded context does not contain enough readable text.")
    return text


def _is_supported_file(filename: str) -> bool:
    return filename.lower().endswith((".docx", ".txt", ".md", ".markdown", ".pdf", ".pptx", ".xlsx", ".xls", ".html", ".htm", ".csv"))
