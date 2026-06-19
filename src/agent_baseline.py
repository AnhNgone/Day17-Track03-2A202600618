from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent when dependencies exist.
        self.langchain_agent = None
        if not self.force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: return the agent response and token accounting.

        Pseudocode:
        - If a live agent exists, call the live path.
        - Otherwise use a deterministic offline path.
        """
        if self.langchain_agent:
            try:
                response = self.langchain_agent.invoke(message)
                text = response.content

                tokens = estimate_tokens(message) + estimate_tokens(text)

                state = self.sessions.setdefault(thread_id, SessionState())
                state.messages.append({"role": "user", "content": message})
                state.messages.append({"role": "assistant", "content": text})

                state.token_usage += tokens
                state.prompt_tokens_processed += sum(
                    estimate_tokens(m["content"]) for m in state.messages
                )

                return {"response": text}
            except Exception:
                # fallback offline nếu lỗi
                return self._reply_offline(thread_id, message)

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        state = self.sessions.get(thread_id)
        if not state:
            return 0
        return state.token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        state = self.sessions.get(thread_id)
        if not state:
            return 0
        return state.prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement a simple offline behavior.

        Suggested behavior:
        - Store the new user message in the session
        - Generate a short deterministic reply
        - Update token counts
        - Never remember facts across different thread ids
        """
        state = self.sessions.setdefault(thread_id, SessionState())

        # 1. lưu user message
        state.messages.append({"role": "user", "content": message})

        # 2. tạo response đơn giản (deterministic)
        reply = f"(Baseline) Bạn vừa nói: {message}"

        # 3. lưu assistant message
        state.messages.append({"role": "assistant", "content": reply})

        # 4. tính token usage
        tokens = estimate_tokens(message) + estimate_tokens(reply)
        state.token_usage += tokens

        # 5. prompt tokens processed (baseline = full history mỗi lần)
        state.prompt_tokens_processed += sum(
            estimate_tokens(m["content"]) for m in state.messages
        )

        return {"response": reply}

    def _maybe_build_langchain_agent(self):
        """Student TODO: optionally wire `create_agent` + `InMemorySaver` here.

        Use `build_chat_model(self.config.model)` so the baseline can run with any supported provider.
        """
        try:
            llm = build_chat_model(self.config.model)
            self.langchain_agent = llm
        except Exception:
            self.langchain_agent = None
