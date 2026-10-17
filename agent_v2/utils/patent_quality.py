"""Deterministic quality checks and repairs for generated patent text.

The language model is still responsible for drafting the patent.  This module
owns the invariants that should not be left to a language model: claim
numbering, exact duplicate claims, repeated local sentences, and a small set
of unambiguous terminology aliases.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher


PATENT_FIELDS = (
    "标题",
    "摘要",
    "权利要求书",
    "技术领域",
    "背景技术",
    "发明内容",
    "具体实施方式",
)

CLAIMS_FIELD = "权利要求书"
_XML_TAG_RE = re.compile(r"</?[\w:-]+(?:\s[^>]*)?>")
_CLAIM_NUMBER_RE = re.compile(
    r"^\s*(?:(?:权利要求)\s*)?(\d+)\s*[\.．、\)）]\s*"
)
_UNNUMBERED_CLAIM_START_RE = re.compile(
    r"^\s*(?:一种|根据权利要求\s*\d+|如权利要求\s*\d+)"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])\s*")
_SPACE_RE = re.compile(r"\s+")

# These are safe to normalize because they do not change a technical
# feature.  Technical aliases with a possible semantic difference are only
# reported, not rewritten automatically.
_REFERENCE_ALIASES = ("本发明", "本申请", "本公开", "本专利")
_TECHNICAL_ALIAS_GROUPS = (
    ("数字串", "数字字符串"),
    ("数据库表", "数据库表格"),
)


@dataclass
class ClaimPart:
    body: str
    original_number: int | None = None


def _clean_generated_text(text: str) -> str:
    text = _XML_TAG_RE.sub("", text or "")
    text = re.sub(r"```(?:json|text|markdown)?", "", text, flags=re.IGNORECASE)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^\s*(?:权利要求书|Claims?)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _signature(text: str) -> str:
    """Return a conservative signature for exact duplicate detection."""
    text = re.sub(r"^\s*\d+\s*[\.．、\)）]\s*", "", text or "")
    text = _SPACE_RE.sub("", text)
    return re.sub(r"[，。；：、,.!?！？:;（）()【】\[\]\"“”‘’'《》<>]", "", text)


def _strip_claim_number(line: str) -> tuple[int | None, str]:
    match = _CLAIM_NUMBER_RE.match(line)
    if not match:
        return None, line.strip()
    return int(match.group(1)), line[match.end():].strip()


def _looks_like_claim_start(line: str) -> bool:
    return bool(_UNNUMBERED_CLAIM_START_RE.match(line))


def _flush_claim(parts: list[ClaimPart], lines: list[str], number: int | None) -> None:
    body = "\n".join(lines).strip()
    body = re.sub(r"\n{3,}", "\n\n", body)
    if body:
        parts.append(ClaimPart(body=body, original_number=number))


def parse_claims(text: str) -> list[ClaimPart]:
    """Parse numbered and unnumbered claim blocks.

    Generated text sometimes restarts numbering halfway through the claim
    section, and sometimes emits dependent claims without a number.  Parsing
    starts from line boundaries so a later normalization pass can recover both
    forms without relying on the model's numbering.
    """
    clean = _clean_generated_text(text)
    if not clean:
        return []

    parts: list[ClaimPart] = []
    current_lines: list[str] = []
    current_number: int | None = None
    previous_was_blank = True
    explicit_number_seen = False

    for raw_line in clean.splitlines():
        line = raw_line.strip()
        if not line:
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            previous_was_blank = True
            continue

        number, remainder = _strip_claim_number(line)
        if number is not None:
            _flush_claim(parts, current_lines, current_number)
            current_lines = [remainder] if remainder else []
            current_number = number
            explicit_number_seen = True
            previous_was_blank = False
            continue

        # A top-level dependent claim often appears as an unnumbered paragraph
        # after a numbered block.  Treat it as a new claim only at a natural
        # paragraph boundary, so "根据..." inside an ordinary sentence is kept.
        if (
            current_lines
            and _looks_like_claim_start(line)
            and (previous_was_blank or current_lines[-1].rstrip().endswith(("。", "；", ";")))
        ):
            _flush_claim(parts, current_lines, current_number)
            current_lines = [line]
            current_number = None
            previous_was_blank = False
            continue

        current_lines.append(line)
        previous_was_blank = False

    _flush_claim(parts, current_lines, current_number)

    # If the whole field had no recognizable claim boundary, it is still one
    # claim candidate.  Keeping it is preferable to silently deleting content.
    if not parts and clean:
        return [ClaimPart(body=clean)]
    return parts


def _near_duplicate(left: str, right: str) -> bool:
    left_sig = _signature(left)
    right_sig = _signature(right)
    if not left_sig or not right_sig:
        return False
    if left_sig == right_sig:
        return True
    if min(len(left_sig), len(right_sig)) < 100:
        return False
    ratio = SequenceMatcher(None, left_sig, right_sig).ratio()
    shorter, longer = sorted((left_sig, right_sig), key=len)
    return ratio >= 0.985 or (len(shorter) / len(longer) >= 0.97 and shorter in longer)


def normalize_claims(text: str) -> str:
    """Remove duplicate claim bodies and emit one continuous numbering."""
    unique: list[str] = []
    seen: set[str] = set()
    for part in parse_claims(text):
        body = _clean_generated_text(part.body)
        body = re.sub(r"^\s*\d+\s*[\.．、\)）]\s*", "", body).strip()
        if not body:
            continue
        signature = _signature(body)
        if not signature or signature in seen:
            continue
        if any(_near_duplicate(body, existing) for existing in unique):
            continue
        seen.add(signature)
        unique.append(body)

    normalized: list[str] = []
    for idx, body in enumerate(unique, 1):
        if idx == 1:
            body = re.sub(
                r"^\s*根据权利要求\s*1\s*所述的",
                "一种",
                body,
                count=1,
            )
        else:
            body = re.sub(
                rf"^\s*根据权利要求\s*{idx}\s*所述的",
                "根据权利要求1所述的",
                body,
                count=1,
            )
        normalized.append(f"{idx}. {body}")
    return "\n\n".join(normalized)


def _dedupe_sentences(text: str, min_length: int = 12) -> str:
    paragraphs = re.split(r"\n\s*\n", _clean_generated_text(text))
    output_paragraphs: list[str] = []
    seen_field: set[str] = set()
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(paragraph) if item.strip()]
        kept: list[str] = []
        for sentence in sentences:
            sig = _signature(sentence)
            if len(sig) >= min_length and sig in seen_field:
                continue
            if sig:
                seen_field.add(sig)
            kept.append(sentence)
        if kept:
            output_paragraphs.append("".join(kept))

    # A repeated paragraph is more damaging than a repeated short sentence.
    unique_paragraphs: list[str] = []
    seen_paragraphs: set[str] = set()
    for paragraph in output_paragraphs:
        sig = _signature(paragraph)
        if len(sig) >= 50 and sig in seen_paragraphs:
            continue
        seen_paragraphs.add(sig)
        unique_paragraphs.append(paragraph)
    return "\n\n".join(unique_paragraphs).strip()


def _replace_aliases(text: str, aliases: dict[str, str]) -> str:
    for source, target in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if source != target:
            text = text.replace(source, target)
    return text


def _reference_aliases(patent: dict[str, str]) -> dict[str, str]:
    # Self-reference should stay consistent across the document.
    return {item: "本发明" for item in _REFERENCE_ALIASES if item != "本发明"}


def _draft_term_aliases(draft: str) -> dict[str, str]:
    """Return only aliases whose meaning is unambiguous in this domain."""
    if not draft:
        return {}
    aliases: dict[str, str] = {}
    for group in _TECHNICAL_ALIAS_GROUPS:
        if any(term in draft for term in group):
            canonical = group[0]
            aliases.update({term: canonical for term in group[1:]})
    return aliases


def _sentence_occurrences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT_RE.split(_clean_generated_text(text))
        if sentence.strip()
    ]


def _cross_field_duplicates(patent: dict[str, str]) -> list[str]:
    occurrences: dict[str, list[str]] = defaultdict(list)
    examples: dict[str, str] = {}
    for field in PATENT_FIELDS:
        for sentence in _sentence_occurrences(patent.get(field, "") or ""):
            sig = _signature(sentence)
            if len(sig) < 35:
                continue
            occurrences[sig].append(field)
            examples.setdefault(sig, sentence)

    issues: list[str] = []
    for sig, fields in occurrences.items():
        unique_fields = list(dict.fromkeys(fields))
        if len(unique_fields) > 1:
            issues.append(
                f"跨字段重复句（{','.join(unique_fields)}）：{examples[sig][:80]}"
            )
    return issues


def _terminology_issues(patent: dict[str, str], draft: str = "") -> list[str]:
    issues: list[str] = []
    present_references = [
        term for term in _REFERENCE_ALIASES
        if any(term in patent.get(field, "") for field in PATENT_FIELDS)
    ]
    if len(present_references) > 1:
        issues.append(f"自称术语混用：{'、'.join(present_references)}")

    if draft and "知识图谱" in draft:
        graphish = [
            field for field in PATENT_FIELDS
            if re.search(
                r"(?:所述|该|目标|原始|初始|当前)图谱",
                patent.get(field, "") or "",
            )
        ]
        if graphish:
            issues.append(f"知识图谱术语未统一：{','.join(graphish)}")
    return issues


def _remove_cross_field_duplicates(patent: dict[str, str]) -> None:
    """Remove long exact copies from later explanatory sections.

    Claims are retained as the legal source of the limitation.  A sentence is
    removed from another field only when the later field still has enough
    substantive content left, so the repair stays conservative.
    """
    field_order = ("技术领域", "背景技术", "权利要求书", "发明内容", "具体实施方式")
    seen: set[str] = set()
    for field in field_order:
        value = patent.get(field, "") or ""
        paragraphs = re.split(r"\n\s*\n", _clean_generated_text(value))
        rebuilt_paragraphs: list[str] = []
        removed_any = False
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            kept_sentences: list[str] = []
            for sentence in _sentence_occurrences(paragraph):
                sig = _signature(sentence)
                if len(sig) < 35:
                    kept_sentences.append(sentence)
                    continue
                if field != "权利要求书" and sig in seen:
                    removed_any = True
                    continue
                kept_sentences.append(sentence)
                seen.add(sig)
            rebuilt = "".join(kept_sentences).strip()
            if rebuilt:
                rebuilt_paragraphs.append(rebuilt)

        if field == "权利要求书":
            continue
        candidate = "\n\n".join(rebuilt_paragraphs).strip()
        if removed_any and candidate and len(candidate) >= max(120, int(len(value) * 0.35)):
            patent[field] = candidate


def quality_issues(patent: dict[str, str], draft: str = "") -> list[str]:
    """Return quality findings after normalization-ready parsing."""
    issues: list[str] = []
    claims = patent.get(CLAIMS_FIELD, "") or ""
    parts = parse_claims(claims)
    numbers = [part.original_number for part in parts]
    if not parts:
        issues.append("权利要求书为空或无法解析")
    else:
        if any(number is None for number in numbers):
            issues.append("权利要求存在未编号条款")
        explicit_numbers = [number for number in numbers if number is not None]
        if explicit_numbers and explicit_numbers != list(range(1, len(explicit_numbers) + 1)):
            issues.append(f"权利要求编号不连续：{explicit_numbers[:20]}")
        signatures = [_signature(part.body) for part in parts]
        if len(signatures) != len(set(signatures)):
            issues.append("权利要求存在完全重复条款")
        for idx, part in enumerate(parts, 1):
            for reference in re.findall(r"根据权利要求\s*(\d+)", part.body):
                if int(reference) >= idx:
                    issues.append(f"权利要求{idx}引用了当前或后续条款{reference}")

    for field in PATENT_FIELDS:
        if field == CLAIMS_FIELD:
            continue
        text = patent.get(field, "") or ""
        sentence_sigs = [
            _signature(sentence)
            for sentence in _sentence_occurrences(text)
            if len(_signature(sentence)) >= 12
        ]
        if len(sentence_sigs) != len(set(sentence_sigs)):
            issues.append(f"{field}存在重复句或重复段")

    issues.extend(_cross_field_duplicates(patent))
    issues.extend(_terminology_issues(patent, draft))
    return list(dict.fromkeys(issues))


def repair_patent(patent: dict[str, str], draft: str = "") -> dict[str, str]:
    """Apply only deterministic, low-risk repairs to a patent dict."""
    repaired = dict(patent)
    aliases = _reference_aliases(repaired)
    aliases.update(_draft_term_aliases(draft))

    for field in PATENT_FIELDS:
        value = repaired.get(field, "")
        if not isinstance(value, str):
            continue
        value = _clean_generated_text(value)
        value = _replace_aliases(value, aliases)
        if draft and "知识图谱" in draft:
            value = re.sub(
                r"(所述|该|目标|原始|初始|当前)图谱",
                r"\1知识图谱",
                value,
            )
        if field == CLAIMS_FIELD:
            repaired[field] = normalize_claims(value)
        else:
            repaired[field] = _dedupe_sentences(value)
    _remove_cross_field_duplicates(repaired)
    return repaired
