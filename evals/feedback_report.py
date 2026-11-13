"""
evals/feedback_report.py - Summarises optional user feedback so it can drive prompt fixes.

Usage (from the project root):
    python evals/feedback_report.py                 # all stored feedback (TTL keeps ~30 days)
    python evals/feedback_report.py --days 7        # last 7 days only
    python evals/feedback_report.py --export disliked.jsonl

Shows:
- likes / dislikes and dislike rate per message kind (question, recap, answer, ...)
- dislike rate per prompt version, to compare before/after a prompt change
- most common reasons per kind
- latest comments
- whole-conversation ratings

--export writes every disliked message with the preceding messages as context (JSON lines),
as raw material for new eval personas or prompt fixes.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_db  # noqa: E402

CONTEXT_MESSAGES = 6


def _rate(down: int, total: int) -> str:
    return f"{(down / total * 100):.0f}%" if total else "-"


def summarize(docs: list[dict]) -> dict:
    """Pure aggregation over feedback documents (no DB access), so it can be unit-tested."""
    by_kind: dict[str, Counter] = defaultdict(Counter)
    by_version: dict[tuple[str, str], Counter] = defaultdict(Counter)
    reasons_by_kind: dict[str, Counter] = defaultdict(Counter)
    conversation = Counter()
    comments: list[dict] = []

    for d in docs:
        rating = d.get("rating")
        snap = d.get("snapshot") or {}
        if d.get("target") == "conversation":
            conversation[rating] += 1
            kind = "conversation"
        else:
            kind = snap.get("kind", "unknown")
            by_kind[kind][rating] += 1
            if snap.get("prompt_name"):
                by_version[(snap["prompt_name"], snap.get("prompt_version", "unknown"))][rating] += 1
        for r in d.get("reasons") or []:
            reasons_by_kind[f"{kind} ({rating})"][r] += 1
        if d.get("comment"):
            comments.append({
                "when": d.get("updated_at"),
                "kind": kind,
                "rating": rating,
                "comment": d["comment"],
            })

    comments.sort(key=lambda c: str(c["when"] or ""), reverse=True)
    return {
        "by_kind": {k: dict(v) for k, v in by_kind.items()},
        "by_version": {f"{name}@{ver}": dict(v) for (name, ver), v in by_version.items()},
        "reasons": {k: v.most_common(5) for k, v in reasons_by_kind.items()},
        "conversation": dict(conversation),
        "comments": comments[:20],
        "total": len(docs),
    }


def print_report(summary: dict) -> None:
    print(f"\n=== Feedback report ({summary['total']} entries) ===\n")

    print("By message kind:")
    print(f"  {'kind':<14}{'up':>6}{'down':>6}{'dislike':>9}")
    for kind, c in sorted(summary["by_kind"].items()):
        up, down = c.get("up", 0), c.get("down", 0)
        print(f"  {kind:<14}{up:>6}{down:>6}{_rate(down, up + down):>9}")

    print("\nBy prompt version (compare before/after prompt changes):")
    for key, c in sorted(summary["by_version"].items()):
        up, down = c.get("up", 0), c.get("down", 0)
        print(f"  {key:<34}{up:>4} up {down:>4} down   dislike {_rate(down, up + down)}")

    conv = summary["conversation"]
    if conv:
        up, down = conv.get("up", 0), conv.get("down", 0)
        print(f"\nWhole conversation: {up} up, {down} down (dislike {_rate(down, up + down)})")

    if summary["reasons"]:
        print("\nTop reasons:")
        for key, items in sorted(summary["reasons"].items()):
            print(f"  {key}: " + ", ".join(f"{r} ({n})" for r, n in items))

    if summary["comments"]:
        print("\nLatest comments:")
        for c in summary["comments"]:
            print(f"  [{c['kind']}, {c['rating']}] {c['comment']}")
    print()


def export_disliked(db, docs: list[dict], path: Path) -> int:
    """Writes disliked messages with preceding context as JSON lines. Returns the count written."""
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            if d.get("rating") != "down" or d.get("target") != "message":
                continue
            session = db.sessions.find_one({"_id": d["session_id"]}) or {}
            messages = session.get("messages") or []
            idx = next((i for i, m in enumerate(messages) if str(m.get("_id")) == d.get("message_id")), None)
            context = messages[max(0, idx - CONTEXT_MESSAGES):idx] if idx is not None else []
            f.write(json.dumps({
                "session_id": d["session_id"],
                "kind": (d.get("snapshot") or {}).get("kind"),
                "prompt": f"{(d.get('snapshot') or {}).get('prompt_name')}@{(d.get('snapshot') or {}).get('prompt_version')}",
                "reasons": d.get("reasons") or [],
                "comment": d.get("comment", ""),
                "context": [{"role": m.get("role"), "text": m.get("text", "")} for m in context],
                "disliked_message": (d.get("snapshot") or {}).get("text", ""),
            }, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise optional user feedback")
    parser.add_argument("--days", type=int, help="Only include feedback from the last N days")
    parser.add_argument("--export", type=str, help="Write disliked messages with context to this .jsonl file")
    args = parser.parse_args()

    db = get_db()
    query: dict = {}
    if args.days:
        query["updated_at"] = {"$gte": datetime.now(timezone.utc) - timedelta(days=args.days)}
    docs = list(db.feedback.find(query))

    print_report(summarize(docs))
    if args.export:
        n = export_disliked(db, docs, Path(args.export))
        print(f"Exported {n} disliked message(s) to {args.export}")


if __name__ == "__main__":
    main()
