from __future__ import annotations

from ast import pattern
from dataclasses import dataclass
import profile
import re
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str

@dataclass
class Fact:
    value: str
    confidence: float
    source: str = "message"


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent.
        self.langchain_agent = None
        if not self.force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: route between offline mode and live mode."""
        if self.langchain_agent:
            try:
                response = self.langchain_agent.invoke(message)
                return {"response": response.content}
            except Exception:
                return self._reply_offline(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def extract_profile_updates_with_confidence(self, message: str) -> dict[str, Fact]:
        facts = {}

        msg = message.strip()
        lower_msg = msg.lower()

        # Không lưu các câu hỏi recall
        recall_phrases = [
            "nhắc lại",
            "mình tên gì",
            "tôi tên gì",
            "ở đâu",
            "là gì",
            "recall",
        ]

        if any(p in lower_msg for p in recall_phrases):
            return facts

        # ===== NAME =====
        name_patterns = [
            r"tên tôi là\s+(.+?)(?:\.|,|$)",
            r"mình tên là\s+(.+?)(?:\.|,|$)",
            r"tôi tên là\s+(.+?)(?:\.|,|$)",
        ]

        for pattern in name_patterns:
            match = re.search(pattern, msg, re.IGNORECASE)
            if match:
                facts["name"] = Fact(
                    value=match.group(1).strip(),
                    confidence=0.95,
                )
                break

        # ===== JOB =====
        job_patterns = [
            r"đang làm\s+(.+?)(?:\.|,|$)",
            r"nghề nghiệp.*?là\s+(.+?)(?:\.|,|$)",
            r"tôi là\s+(.+?)(?:\.|,|$)",
        ]

        for pattern in job_patterns:
            match = re.search(pattern, msg, re.IGNORECASE)
            if match:
                value = match.group(1).strip()

                # tránh bắt nhầm "tôi là Dũng"
                if len(value.split()) > 1:
                    facts["job"] = Fact(
                        value=value,
                        confidence=0.85,
                    )
                    break

        # ===== CITY / LOCATION =====
        city_patterns = [
            r"hiện ở\s+([A-Za-zÀ-ỹ\s]+?)(?:\s+để|\.|,|$)",
            r"đang ở\s+([A-Za-zÀ-ỹ\s]+?)(?:\s+để|\.|,|$)",
            r"làm việc ở\s+([A-Za-zÀ-ỹ\s]+?)(?:\s+để|\.|,|$)",
            r"sống ở\s+([A-Za-zÀ-ỹ\s]+?)(?:\s+để|\.|,|$)",
        ]

        for pattern in city_patterns:
            match = re.search(pattern, msg, re.IGNORECASE)
            if match:
                facts["city"] = Fact(
                    value=match.group(1).strip(),
                    confidence=0.90,
                )
                break

        # ===== FAVORITE DRINK =====
        drink_patterns = [
            r"đồ uống yêu thích.*?là\s+(.+?)(?:\.|,|$)",
            r"thích uống\s+(.+?)(?:\.|,|$)",
        ]

        for pattern in drink_patterns:
            m = re.search(pattern, msg, re.IGNORECASE)
            if m:
                facts["favorite_drink"] = Fact(
                    value=m.group(1).strip(),
                    confidence=0.95,
                )
                break

        # ===== PET =====
        pet_patterns = [
            r"tôi nuôi\s+(?:một\s+)?con\s+([A-Za-zÀ-ỹ\s]+?)(?:\.|,|$)",
            r"mình nuôi\s+(?:một\s+)?con\s+([A-Za-zÀ-ỹ\s]+?)(?:\.|,|$)",
            r"nuôi một bé\s+([A-Za-zÀ-ỹ]+)",
            r"nuôi một con\s+([A-Za-zÀ-ỹ]+)",
        ]

        for pattern in pet_patterns:
            m = re.search(pattern, msg, re.IGNORECASE)
            if m:
                pet = m.group(1).strip()

                facts["pet"] = Fact(
                    value=pet,
                    confidence=0.95,
                )
                break

        #==== FAVORITE FOOD =====
        food_match = re.search(
            r"món ăn yêu thích là\s+(.+?)(?:\.|$)",
            lower_msg
        )

        if food_match:
            facts["favorite_food"] = Fact(
                value=food_match.group(1).strip(),
                confidence=0.95,
            )

        # ===== STYLE =====
        if (
            "3 bullet" in msg.lower()
            or "ba bullet" in msg.lower()
            or "trả lời ngắn gọn" in msg.lower()
        ):
            facts["style"] = Fact(
                value="3 bullet ngắn, có ví dụ thực chiến, nhấn trade-off",
                confidence=0.95,
            )

        return facts
    
    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        facts = self.extract_profile_updates_with_confidence(message)

        for key, fact in facts.items():
            if fact.confidence >= self.config.confidence_threshold:
                self.profile_store.upsert_fact(
                    user_id=user_id,
                    key=key,
                    value=fact.value,
                    confidence=fact.confidence,
                    source=fact.source
                )

        # 3. Append user message vào compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 4. Estimate prompt tokens
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        # 5. Generate response (phải dùng memory)
        response = self._offline_response(user_id, thread_id, message)

        # 6. Append assistant message
        self.compact_memory.append(thread_id, "assistant", response)

        # update token usage
        tokens = estimate_tokens(message) + estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + tokens

        return {"response": response}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Student TODO: estimate the context carried into one turn.

        Hint:
        - Include `User.md`
        - Include compact summary text
        - Include recent kept messages
        """
        profile = self.profile_store.read_text(user_id)

        ctx = self.compact_memory.context(thread_id)

        summary = ctx.get("summary", "")
        messages = ctx.get("messages", [])

        tokens = estimate_tokens(profile) + estimate_tokens(summary)

        for m in messages:
            tokens += estimate_tokens(m["content"])

        return tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Student TODO: return a deterministic answer using persisted memory.

        Make sure the advanced agent can answer questions like:
        - "Mình tên gì?"
        - "Hiện tại mình làm nghề gì?"
        - "Nhắc lại style trả lời mình thích"
        - questions in the long stress dataset
        """
        profile = self.profile_store.read_text(user_id)

        msg = message.lower()

        def get_fact(key: str):
            for line in profile.splitlines():
                if line.startswith(f"{key}:"):
                    value = line.split(":", 1)[1].strip()
                    value = value.split(" (")[0].strip()
                    return value
            return None

        # recall name
        if "tên" in msg:
            for line in profile.splitlines():
                if "name:" in line:
                    value = line.split(':',1)[1].strip()
                    value = value.split(' (')[0]   
                    return f"Bạn tên là {value}"

        # recall job
        if "nghề" in msg or "làm gì" in msg:
            for line in profile.splitlines():
                if "job:" in line:
                    value = line.split(':',1)[1].strip()
                    value = value.split(' (')[0]   
                    return f"Bạn làm {value}"

        # recall preference
        if "thích" in msg:
            for line in profile.splitlines():
                if "preference:" in line:
                    value = line.split(':',1)[1].strip()
                    value = value.split(' (')[0]   
                    return f"Bạn thích: {value}"
                
        if "đồ uống" in msg:
            drink = get_fact("favorite_drink")
            if drink:
                return f"Đồ uống yêu thích của bạn là {drink}"

        if "nuôi con gì" in msg or "thú cưng" in msg:
            for line in profile.splitlines():
                if line.startswith("pet:"):
                    value = line.split(":", 1)[1].strip()
                    value = value.split(" (")[0]
                    return f"Bạn nuôi {value}"
            
        if "món ăn" in msg:
            for line in profile.splitlines():
                if line.startswith("favorite_food:"):
                    value = line.split(":", 1)[1].strip()
                    value = value.split(" (")[0]
                    return f"Món ăn yêu thích của bạn là {value}"

        # fallback dùng context
        ctx = self.compact_memory.context(thread_id)
        summary = ctx.get("summary", "")

        if summary:
            return f"(Advanced) Dựa trên lịch sử: {summary[:100]}..."

        return f"(Advanced) Bạn vừa nói: {message}"

    def _maybe_build_langchain_agent(self):
        """Student TODO: wire a live agent with tools and compact middleware.

        High-level design:
        - `build_chat_model(self.config.model)` for the selected provider
        - `InMemorySaver` for short-term thread state
        - tool to read `User.md`
        - tool to write/edit `User.md`
        - dynamic prompt that injects profile memory
        - summarization middleware for long threads
        """
        try:
            llm = build_chat_model(self.config.model)
            self.langchain_agent = llm
        except Exception:
            self.langchain_agent = None
