import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.config import get_llm_settings
from app.models import ContextDocument, ContextDocumentCreate, Product, ProductCreate, ProductKnowledgeBase
from app.services.document_loader import extract_docx_text


DATA_DIR = Path(os.getenv("PRD_REVIEW_DATA_DIR", Path(__file__).resolve().parents[2] / ".data"))
PRODUCTS_FILE = DATA_DIR / "products.json"
OPENKB_ROOT = DATA_DIR / "openkb"


def create_product(payload: ProductCreate) -> Product:
    product = Product(
        id=_slugify(payload.name),
        name=payload.name,
        description=payload.description,
        created_at=_now(),
    )
    products = _read_products()
    if any(item.id == product.id for item in products):
        product = product.model_copy(update={"id": f"{product.id}-{uuid4().hex[:6]}"})
    products.append(product)
    _write_json(PRODUCTS_FILE, [item.model_dump() for item in products])
    _ensure_openkb_workspace(product)
    return product


def list_products() -> list[Product]:
    products = _read_products()
    for product in products:
        _ensure_openkb_workspace(product)
    return products


def get_product(product_id: str) -> Product:
    for product in _read_products():
        if product.id == product_id:
            _ensure_openkb_workspace(product)
            return product
    raise KeyError(product_id)


def add_context(product_id: str, payload: ContextDocumentCreate) -> ContextDocument:
    product = get_product(product_id)
    kb_dir = _kb_dir(product.id)
    source_path = kb_dir / "raw" / f"{_slugify(payload.title)}-{uuid4().hex[:6]}.md"
    source_path.write_text(f"# {payload.title}\n\n{payload.text.strip()}\n", encoding="utf-8")
    return _ingest_file(product, source_path, payload.title, payload.source_type)


def add_context_file(product_id: str, filename: str, payload: bytes, source_type: str = "upload") -> ContextDocument:
    product = get_product(product_id)
    kb_dir = _kb_dir(product.id)
    raw_path = kb_dir / "raw" / f"{uuid4().hex[:6]}-{_safe_filename(filename)}"
    raw_path.write_bytes(payload)
    return _ingest_file(product, raw_path, filename, source_type)


def list_contexts(product_id: str) -> list[ContextDocument]:
    product = get_product(product_id)
    kb_dir = _kb_dir(product.id)
    registry = _read_registry(kb_dir)
    contexts: list[ContextDocument] = []
    for file_hash, meta in registry.items():
        name = meta.get("name") or meta.get("doc_name") or file_hash[:12]
        text = _read_compiled_text(kb_dir, meta)
        contexts.append(
            ContextDocument(
                id=file_hash,
                product_id=product.id,
                title=name,
                text=text,
                source_type=meta.get("type", "openkb"),
                character_count=len(text),
                created_at=meta.get("created_at", product.created_at),
            )
        )
    return contexts


def get_knowledge_base(product_id: str) -> ProductKnowledgeBase:
    product = get_product(product_id)
    contexts = list_contexts(product_id)
    return ProductKnowledgeBase(
        product=product,
        contexts=contexts,
        context_count=len(contexts),
        total_characters=sum(item.character_count for item in contexts),
    )


def build_review_context(product_id: str, max_characters: int = 16000) -> str:
    product = get_product(product_id)
    kb_dir = _kb_dir(product.id)
    parts = [
        f"Product: {product.name}",
        f"Product description: {product.description or 'Not provided.'}",
        f"OpenKB workspace: {kb_dir}",
    ]
    for path in _compiled_wiki_files(kb_dir):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            parts.append(f"OpenKB page: {path.relative_to(kb_dir)}\n{text}")
    combined = "\n\n---\n\n".join(parts)
    if len(combined) <= max_characters:
        return combined
    return combined[:max_characters] + "\n\n[OpenKB context compacted due to token budget.]"


def _ingest_file(product: Product, file_path: Path, title: str, source_type: str) -> ContextDocument:
    kb_dir = _kb_dir(product.id)
    if not _run_openkb_add(kb_dir, file_path):
        _fallback_compile(kb_dir, file_path, title)

    file_hash = _hash_file(file_path)
    registry = _read_registry(kb_dir)
    meta = _find_registry_meta(registry, file_hash, file_path)
    text = _read_compiled_text(kb_dir, meta or {"raw_path": str(file_path.relative_to(kb_dir)), "name": title})
    return ContextDocument(
        id=file_hash,
        product_id=product.id,
        title=(meta or {}).get("name", title),
        text=text,
        source_type=(meta or {}).get("type", source_type),
        character_count=len(text),
        created_at=(meta or {}).get("created_at", _now()),
    )


def _ensure_openkb_workspace(product: Product) -> None:
    kb_dir = _kb_dir(product.id)
    if _is_valid_openkb_workspace(kb_dir):
        return
    kb_dir.mkdir(parents=True, exist_ok=True)
    if not _run_openkb_init(kb_dir):
        _fallback_init(kb_dir)
    product_page = kb_dir / "wiki" / "entities" / f"{product.id}.md"
    product_page.parent.mkdir(parents=True, exist_ok=True)
    if not product_page.exists():
        product_page.write_text(
            f"# {product.name}\n\nType: product\n\n{product.description or 'No description provided.'}\n",
            encoding="utf-8",
        )


def _run_openkb_init(kb_dir: Path) -> bool:
    command = _openkb_command()
    if not command:
        return False
    settings = get_llm_settings()
    if not _has_openkb_llm_key(settings):
        return False
    model = os.getenv("OPENKB_MODEL") or settings.model or "gpt-4o-mini"
    env = _openkb_env(kb_dir)
    try:
        result = subprocess.run(
            [*command, "init", "--model", model, "--language", "en"],
            cwd=kb_dir,
            input="\n\n",
            text=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(os.getenv("OPENKB_INIT_TIMEOUT_SECONDS", "120")),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0 and (kb_dir / ".openkb").is_dir()


def _is_valid_openkb_workspace(kb_dir: Path) -> bool:
    return all(
        path.exists()
        for path in [
            kb_dir / ".openkb" / "config.yaml",
            kb_dir / ".openkb" / "hashes.json",
            kb_dir / "wiki" / "index.md",
        ]
    )


def _run_openkb_add(kb_dir: Path, file_path: Path) -> bool:
    command = _openkb_command()
    if not command:
        return False
    if not _has_openkb_llm_key(get_llm_settings()):
        return False
    result = subprocess.run(
        [*command, "--kb-dir", str(kb_dir), "add", str(file_path)],
        cwd=kb_dir,
        env=_openkb_env(kb_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=int(os.getenv("OPENKB_ADD_TIMEOUT_SECONDS", "45")),
        check=False,
    )
    return result.returncode == 0 and "failed" not in result.stdout.lower()


def _has_openkb_llm_key(settings) -> bool:
    return bool(os.getenv("LLM_API_KEY") or settings.api_key)


def _fallback_init(kb_dir: Path) -> None:
    for path in [
        kb_dir / "raw",
        kb_dir / "wiki" / "sources" / "images",
        kb_dir / "wiki" / "summaries",
        kb_dir / "wiki" / "concepts",
        kb_dir / "wiki" / "entities",
        kb_dir / "wiki" / "reports",
        kb_dir / ".openkb",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    (kb_dir / "wiki" / "AGENTS.md").write_text("# OpenKB Agent Instructions\n", encoding="utf-8")
    (kb_dir / "wiki" / "index.md").write_text("# Knowledge Base Index\n\n## Documents\n\n## Concepts\n\n## Entities\n\n## Explorations\n", encoding="utf-8")
    (kb_dir / "wiki" / "log.md").write_text("# Operations Log\n\n", encoding="utf-8")
    (kb_dir / ".openkb" / "config.yaml").write_text("model: gpt-4o-mini\nlanguage: en\npageindex_threshold: 20\n", encoding="utf-8")
    (kb_dir / ".openkb" / "hashes.json").write_text("{}\n", encoding="utf-8")


def _fallback_compile(kb_dir: Path, file_path: Path, title: str) -> None:
    doc_name = _slugify(Path(title).stem or file_path.stem)
    raw_path = kb_dir / "raw" / file_path.name
    if file_path.resolve() != raw_path.resolve():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, raw_path)

    text = _extract_text(raw_path)
    source_path = kb_dir / "wiki" / "sources" / f"{doc_name}.md"
    summary_path = kb_dir / "wiki" / "summaries" / f"{doc_name}.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(text, encoding="utf-8")
    summary_path.write_text(f"# {title}\n\n{text[:4000]}\n", encoding="utf-8")

    file_hash = _hash_file(raw_path)
    registry = _read_registry(kb_dir)
    registry[file_hash] = {
        "name": title,
        "doc_name": doc_name,
        "type": raw_path.suffix.lstrip(".") or "text",
        "path": str(raw_path.relative_to(kb_dir)),
        "raw_path": str(raw_path.relative_to(kb_dir)),
        "source_path": str(source_path.relative_to(kb_dir)),
        "summary_path": str(summary_path.relative_to(kb_dir)),
        "created_at": _now(),
        "compiler": "openkb-fallback",
    }
    _write_json(kb_dir / ".openkb" / "hashes.json", registry)
    _append_index(kb_dir, title, summary_path)


def _compiled_wiki_files(kb_dir: Path) -> list[Path]:
    files = [kb_dir / "wiki" / "index.md"]
    for folder in ["summaries", "concepts", "entities"]:
        files.extend(sorted((kb_dir / "wiki" / folder).glob("*.md")))
    return [path for path in files if path.exists()]


def _read_compiled_text(kb_dir: Path, meta: dict) -> str:
    candidates = [meta.get("summary_path"), meta.get("source_path"), meta.get("raw_path"), meta.get("path")]
    for candidate in candidates:
        if not candidate:
            continue
        path = kb_dir / candidate
        if path.exists() and path.is_file():
            return _extract_text(path)
    return ""


def _extract_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return extract_docx_text(path.read_bytes())
    if path.suffix.lower() in {".txt", ".md", ".markdown", ".csv", ".html", ".htm"}:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    return f"[Binary or rich document stored at {path.name}. Run OpenKB with LLM configured for full compilation.]"


def _openkb_command() -> list[str] | None:
    executable = shutil.which("openkb")
    if executable:
        return [executable]
    try:
        import openkb  # noqa: F401
    except Exception:
        return None
    return ["python3", "-m", "openkb"]


def _openkb_env(kb_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    settings = get_llm_settings()
    if settings.api_key:
        env.setdefault("LLM_API_KEY", settings.api_key)
    env.setdefault("OPENKB_DIR", str(kb_dir))
    return env


def _read_products() -> list[Product]:
    return [Product.model_validate(item) for item in _read_json(PRODUCTS_FILE)]


def _read_registry(kb_dir: Path) -> dict[str, dict]:
    path = kb_dir / ".openkb" / "hashes.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8") or "{}")


def _find_registry_meta(registry: dict[str, dict], file_hash: str, file_path: Path) -> dict | None:
    if file_hash in registry:
        return registry[file_hash]
    name = file_path.name
    for meta in registry.values():
        if meta.get("name") == name or str(meta.get("raw_path", "")).endswith(name):
            return meta
    return None


def _read_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_index(kb_dir: Path, title: str, summary_path: Path) -> None:
    index_path = kb_dir / "wiki" / "index.md"
    link = summary_path.relative_to(kb_dir / "wiki").with_suffix("").as_posix()
    current = index_path.read_text(encoding="utf-8") if index_path.exists() else "# Knowledge Base Index\n\n## Documents\n"
    entry = f"- [[{link}|{title}]]\n"
    if entry not in current:
        index_path.write_text(current.rstrip() + "\n" + entry + "\n", encoding="utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kb_dir(product_id: str) -> Path:
    return OPENKB_ROOT / product_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or uuid4().hex[:10]


def _safe_filename(value: str) -> str:
    name = Path(value).name
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or f"upload-{uuid4().hex[:6]}.txt"
