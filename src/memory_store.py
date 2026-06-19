from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import path
from pathlib import Path
import re


def estimate_tokens(text: str) -> int:
    """Student TODO: implement a simple token estimator.

    Example idea:
    - Strip whitespace
    - Return 0 for empty text
    - Approximate tokens from character count, e.g. len(text) / 4
    """
    text = text.strip()
    if not text:
        return 0
    return len(text) // 4


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        safe_id = user_id.replace(" ", "_")
        return self.root_dir / f"{safe_id}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return ""   
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.write_text(content, encoding="utf-8")
        return path
    
    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text in content:
            content = content.replace(search_text, replacement, 1)
            self.write_text(user_id, content)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size
    
    def upsert_fact(
        self,
        user_id: str,
        key: str,
        value: str,
        confidence: float = 1.0,
        source: str = "message"
    ) -> None:
        content = self.read_text(user_id)

        # format line có metadata
        new_line = f"{key}: {value} (conf={confidence:.2f}, src={source})"

        lines = content.splitlines()
        updated = False

        for i, line in enumerate(lines):
            if line.startswith(f"{key}:"):
                lines[i] = new_line
                updated = True
                break

        if not updated:
            lines.append(new_line)

        new_content = "\n".join(lines)
        self.write_text(user_id, new_content)


def extract_profile_updates(message: str) -> dict[str, str]:
    facts = {}

    name_match = re.search(r"tên tôi là (\w+)", message.lower())
    if name_match:
        facts["name"] = name_match.group(1)

    job_match = re.search(r"tôi là (.+)", message.lower())
    if job_match:
        facts["job"] = job_match.group(1)

    if "thích" in message.lower():
        facts["preference"] = message

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    if not messages:
        return ""

    # Lấy tối đa max_items message cũ nhất
    selected = messages[:max_items]

    lines = []
    for m in selected:
        role = m.get("role", "")
        content = m.get("content", "").strip()

        if not content:
            continue

        # rút gọn nội dung để tránh quá dài
        short = content[:100]

        lines.append(f"{role}: {short}")

    return " | ".join(lines)


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        # 1. create thread state if missing
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

        thread = self.state[thread_id]

        # 2. append the new message
        thread["messages"].append({
            "role": role,
            "content": content,
        })

        # tính tổng tokens hiện tại
        total_tokens = sum(
            estimate_tokens(m["content"]) for m in thread["messages"]
        )

        # 3. trigger compaction if needed
        if total_tokens > self.threshold_tokens:
            # tách phần cũ và phần giữ lại
            old_messages = thread["messages"][:-self.keep_messages]
            recent_messages = thread["messages"][-self.keep_messages:]

            # đưa phần cũ vào summary
            summary_text = summarize_messages(old_messages)

            if thread["summary"]:
                thread["summary"] += "\n" + summary_text
            else:
                thread["summary"] = summary_text

            # giữ lại messages mới
            thread["messages"] = recent_messages

            # tăng số lần compact
            thread["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        # return state của thread, nếu chưa có thì trả default
        if thread_id not in self.state:
            return {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        if thread_id not in self.state:
            return 0
        return self.state[thread_id]["compactions"]
