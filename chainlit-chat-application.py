"""
Chainlit 2.0 chat application for Vietnamese legal and education Q&A.

Provides:
- Token-by-token streaming answers via async bridge
- Real-time retrieval status updates
- Citation panel with article breadcrumbs
- Domain switching via APP_DOMAIN env var

Usage:
    chainlit run chainlit-chat-application.py
    APP_DOMAIN=education chainlit run chainlit-chat-application.py
"""

import json
import logging
import os
from importlib import import_module
from pathlib import Path

import chainlit as cl

from vn_legal_rag.online import create_legal_graphrag
from vn_legal_rag.utils import (
    load_config, load_kg, load_forest, load_summaries, build_forest_from_db,
)

_bridge_mod = import_module(
    ".async-retrieval-bridge-for-chainlit", "vn_legal_rag.online"
)
AsyncRetrievalBridge = _bridge_mod.AsyncRetrievalBridge

logger = logging.getLogger(__name__)

DOMAIN = os.environ.get("APP_DOMAIN", "legal")
CONFIG_MAP = {"legal": "config/default.yaml", "education": "config/education.yaml"}

STARTERS = {
    "legal": [
        "Điều kiện thành lập công ty cổ phần?",
        "Mức phạt vượt đèn đỏ?",
        "Quyền của cổ đông phổ thông?",
    ],
    "education": [
        "Điều kiện bảo vệ luận văn thạc sĩ?",
        "Yêu cầu ngoại ngữ đầu ra cao học?",
        "Thời gian đào tạo thạc sĩ tối đa?",
    ],
}


def _init_pipeline():
    """Load pipeline components once at module scope."""
    config_path = CONFIG_MAP.get(DOMAIN, "config/default.yaml")
    logger.info(f"Loading pipeline for domain '{DOMAIN}' from {config_path}")
    config = load_config(config_path)
    kg_config = config.get("kg", {})

    kg = load_kg(kg_config.get("path", "data/kg_enhanced/legal_kg.json"))

    forest_path = kg_config.get("forest", "data/document_forest.json")
    if Path(forest_path).exists():
        forest = load_forest(forest_path)
    else:
        db_path = config.get("database", {}).get("path", "data/legal_docs.db")
        logger.info(f"Forest file not found, building from DB: {db_path}")
        chapter_summaries = load_summaries(
            kg_config.get("chapter_summaries", "")
        )
        forest = build_forest_from_db(db_path, chapter_summaries=chapter_summaries)

    article_summaries = load_summaries(
        kg_config.get("article_summaries", "")
    )
    document_summaries = load_summaries(
        kg_config.get("document_summaries", "")
    )
    domain_groups_path = kg_config.get("domain_groups", "")
    domain_groups = None
    if domain_groups_path and Path(domain_groups_path).exists():
        with open(domain_groups_path, "r", encoding="utf-8") as f:
            domain_groups = json.load(f)

    llm_config = config.get("llm", {})
    graphrag = create_legal_graphrag(
        kg=kg,
        forest=forest,
        db_path=config.get("database", {}).get("path"),
        article_summaries=article_summaries,
        document_summaries=document_summaries,
        domain_groups=domain_groups,
        llm_provider=llm_config.get("provider", "anthropic"),
        llm_model=llm_config.get("model"),
        llm_base_url=llm_config.get("base_url"),
        llm_cache_db=llm_config.get("cache_db"),
        config=config,
    )

    logger.info(f"Pipeline initialized for domain '{DOMAIN}'")
    return AsyncRetrievalBridge(graphrag, domain=DOMAIN)


bridge = _init_pipeline()


@cl.on_chat_start
async def on_start():
    """Store shared bridge reference in session and send welcome."""
    cl.user_session.set("bridge", bridge)

    questions = STARTERS.get(DOMAIN, STARTERS["legal"])
    suggestions = "\n".join(f"- {q}" for q in questions)
    domain_label = "pháp luật" if DOMAIN == "legal" else "đào tạo sau đại học"
    await cl.Message(
        content=(
            f"Xin chào! Tôi có thể giúp bạn tra cứu quy định {domain_label}. "
            f"Thử hỏi:\n\n{suggestions}"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user query with streaming response."""
    user_bridge = cl.user_session.get("bridge")
    query = message.content

    status_msg = cl.Message(content="")
    await status_msg.send()

    answer_msg = cl.Message(content="")
    metadata_for_sources = None
    streaming_started = False

    async for event_type, data in user_bridge.stream_query(query):
        if event_type == "status":
            status_msg.content = f"⏳ {data}"
            await status_msg.update()

        elif event_type == "metadata":
            metadata_for_sources = data

        elif event_type == "token":
            if not streaming_started:
                streaming_started = True
                await status_msg.remove()
            await answer_msg.stream_token(data)

        elif event_type == "error":
            await status_msg.remove()
            answer_msg.content = f"Lỗi: {data}"

    if not streaming_started:
        await status_msg.remove()

    if metadata_for_sources:
        elements = _build_citation_elements(metadata_for_sources)
        if elements:
            source_names = [el.name for el in elements]
            answer_msg.content += "\n\n**Nguồn tham khảo:** " + ", ".join(source_names)
            answer_msg.elements = elements

    if streaming_started:
        await answer_msg.update()
    else:
        await answer_msg.send()


def _build_citation_elements(metadata):
    """Build citation panel elements with doc > article breadcrumbs."""
    elements = []
    seen = set()

    for ctx in metadata.get("contexts", []):
        source_id = ctx.get("metadata", {}).get("source_id", "")
        text = ctx.get("text", "")
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)

        breadcrumb = _source_id_to_breadcrumb(source_id)
        content = f"**{breadcrumb}**\n\n{text}" if breadcrumb else text
        elements.append(
            cl.Text(name=breadcrumb or source_id, content=content, display="side")
        )

    reasoning = metadata.get("reasoning_path", [])
    if reasoning:
        path_text = "\n".join(f"→ {step}" for step in reasoning)
        elements.append(
            cl.Text(
                name="Lộ trình truy xuất",
                content=f"**Lộ trình truy xuất**\n\n{path_text}",
                display="side",
            )
        )

    return elements


def _source_id_to_breadcrumb(source_id: str) -> str:
    """Convert source_id like '59-2020-QH14:d206' to 'Luật 59/2020/QH14 › Điều 206'."""
    if ":" not in source_id:
        return source_id
    try:
        doc_id, article_ref = source_id.rsplit(":", 1)
        article_num = article_ref[1:] if article_ref.startswith("d") else article_ref
        so_hieu = doc_id.replace("-", "/", 2)
        return f"{so_hieu} › Điều {article_num}"
    except Exception:
        return source_id
