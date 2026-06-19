"""test_agents.py — pytest test suite for BaselineAgent and AdvancedAgent.

Test coverage
-------------
- test_user_markdown_read_write_edit   : User.md create / update / edit
- test_compact_trigger                 : long thread fires compaction
- test_cross_session_recall            : Advanced remembers, Baseline forgets
- test_compact_reduces_prompt_load     : compact cuts prompt tokens vs baseline
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import UserProfileStore, CompactMemoryManager, estimate_tokens
from model_provider import ProviderConfig


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def make_config(tmp_path: Path, compact_threshold: int = 150) -> LabConfig:
    """Build an isolated LabConfig pointing at *tmp_path*.

    The compact_threshold is deliberately small so compaction fires quickly
    in tests without needing very long messages.
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model = ProviderConfig(provider="openai", model_name="gpt-4o-mini")

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=2,     # keep only 2 messages after compaction
        confidence_threshold=0.5,
        model=model,
        judge_model=model,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_long_message(n: int = 80) -> str:
    """Return a message long enough to contribute meaningfully to token counts."""
    return " ".join([f"word{i}" for i in range(n)])


# ---------------------------------------------------------------------------
# Test 1 — User.md read / write / edit
# ---------------------------------------------------------------------------

def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """UserProfileStore must create, read, update, and overwrite User.md facts."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user"

    # 1. Initially empty
    assert store.read_text(user_id) == ""
    assert store.file_size(user_id) == 0

    # 2. Write a fact
    store.upsert_fact(user_id, "name", "DũngCT", confidence=0.95)
    text = store.read_text(user_id)
    assert "name:" in text
    assert "DũngCT" in text
    assert store.file_size(user_id) > 0

    # 3. Add another fact
    store.upsert_fact(user_id, "job", "backend engineer", confidence=0.8)
    text = store.read_text(user_id)
    assert "job:" in text
    assert "backend engineer" in text

    # 4. Update (upsert) the existing name fact — should NOT duplicate the key
    store.upsert_fact(user_id, "name", "DũngCT Updated", confidence=0.99)
    text = store.read_text(user_id)
    lines_with_name = [l for l in text.splitlines() if l.startswith("name:")]
    assert len(lines_with_name) == 1, "upsert must not duplicate the key"
    assert "DũngCT Updated" in lines_with_name[0]

    # 5. write_text overwrites everything
    store.write_text(user_id, "name: Overwritten\n")
    assert store.read_text(user_id).strip() == "name: Overwritten"


# ---------------------------------------------------------------------------
# Test 2 — compact memory trigger
# ---------------------------------------------------------------------------

def test_compact_trigger(tmp_path: Path) -> None:
    """Sending many long messages must trigger at least one compaction."""
    config = make_config(tmp_path, compact_threshold=60)
    agent = AdvancedAgent(config=config, force_offline=True)

    user_id = "compact_user"
    thread_id = "compact_thread"

    # Before any messages — zero compactions
    assert agent.compaction_count(thread_id) == 0

    # Send enough long messages to exceed the threshold
    for i in range(12):
        msg = f"Đây là tin nhắn số {i}: " + " ".join([f"token{j}" for j in range(20)])
        agent.reply(user_id, thread_id, msg)

    # At least one compaction should have fired
    assert agent.compaction_count(thread_id) >= 1, (
        f"Expected compaction but count={agent.compaction_count(thread_id)}"
    )


# ---------------------------------------------------------------------------
# Test 3 — cross-session recall
# ---------------------------------------------------------------------------

def test_cross_session_recall(tmp_path: Path) -> None:
    """Advanced agent must recall facts from a previous thread; Baseline must not."""
    config = make_config(tmp_path)

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    user_id = "recall_user"
    learn_thread = "thread-learn"
    new_thread = "thread-new"

    fact_message = "Tên tôi là DũngCT"

    # --- teach both agents the user's name ---
    baseline.reply(user_id, learn_thread, fact_message)
    advanced.reply(user_id, learn_thread, fact_message)

    # --- ask in a brand-new thread ---
    question = "Mình tên gì?"

    baseline_response = baseline.reply(user_id, new_thread, question)["response"]
    advanced_response = advanced.reply(user_id, new_thread, question)["response"]

    # Advanced must recall the name
    assert "dũngct" in advanced_response.lower(), (
        f"Advanced should recall name. Got: {advanced_response!r}"
    )

    # Baseline must NOT recall (no persistent memory, fresh thread = blank slate)
    # Its response will be the generic echo, not a recall answer.
    assert "dũngct" not in baseline_response.lower(), (
        f"Baseline should NOT recall name across threads. Got: {baseline_response!r}"
    )


# ---------------------------------------------------------------------------
# Test 4 — compact memory reduces prompt token load
# ---------------------------------------------------------------------------

def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Over a long thread, Advanced's prompt token growth must be less than Baseline's.

    Rationale: Baseline re-feeds the *entire* history every turn (O(n^2) tokens).
    Advanced compacts old messages into a short summary, so its prompt token
    growth is sublinear.
    """
    # Very small threshold so compaction fires quickly
    config = make_config(tmp_path, compact_threshold=40)

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    user_id = "load_user"
    thread_b = "thread-baseline-load"
    thread_a = "thread-advanced-load"

    # Send enough turns that compaction fires for Advanced
    for i in range(20):
        msg = f"Turn {i}: " + " ".join([f"word{j}" for j in range(15)])
        baseline.reply(user_id, thread_b, msg)
        advanced.reply(user_id, thread_a, msg)

    baseline_prompt = baseline.prompt_token_usage(thread_b)
    advanced_prompt = advanced.prompt_token_usage(thread_a)

    # Advanced must use strictly fewer prompt tokens than Baseline
    assert advanced_prompt < baseline_prompt, (
        f"Expected Advanced prompt tokens ({advanced_prompt}) "
        f"< Baseline ({baseline_prompt}), "
        f"but compact memory did not reduce load."
    )

    # At least one compaction must have happened to validate the mechanism
    assert advanced.compaction_count(thread_a) >= 1, (
        "No compaction fired — increase number of turns or reduce threshold."
    )


# ---------------------------------------------------------------------------
# Test 5 — baseline truly forgets across NEW thread_ids
# ---------------------------------------------------------------------------

def test_baseline_no_cross_thread_memory(tmp_path: Path) -> None:
    """Baseline must treat each thread_id as an isolated session."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)

    user_id = "forgetful_user"
    baseline.reply(user_id, "t1", "Tên tôi là Alpha")

    # Token usage in t1 must be > 0
    assert baseline.token_usage("t1") > 0

    # t2 starts fresh — zero tokens processed
    assert baseline.token_usage("t2") == 0
    assert baseline.prompt_token_usage("t2") == 0


# ---------------------------------------------------------------------------
# Test 6 — Advanced writes to profile even without a live LLM
# ---------------------------------------------------------------------------

def test_advanced_profile_written_offline(tmp_path: Path) -> None:
    """AdvancedAgent must persist profile facts in offline mode."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)

    user_id = "profile_write_user"
    thread_id = "t1"

    # This message should trigger name extraction and profile write
    agent.reply(user_id, thread_id, "Tên tôi là TestUser")

    size = agent.memory_file_size(user_id)
    assert size > 0, "User.md should have been written with the extracted name fact."


# ---------------------------------------------------------------------------
# Test 7 — estimate_tokens sanity check
# ---------------------------------------------------------------------------

def test_estimate_tokens_basic() -> None:
    """estimate_tokens must return a positive integer for non-empty text."""
    assert estimate_tokens("hello world") > 0
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 100) > estimate_tokens("a" * 10)


# ---------------------------------------------------------------------------
# Run manually (python test_agents.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    results: dict[str, str] = {}
    tests = [
        ("User.md read/write/edit",        test_user_markdown_read_write_edit),
        ("Compact trigger",                 test_compact_trigger),
        ("Cross-session recall",            test_cross_session_recall),
        ("Compact reduces prompt load",     test_compact_reduces_prompt_load_on_long_thread),
        ("Baseline no cross-thread memory", test_baseline_no_cross_thread_memory),
        ("Advanced profile written offline",test_advanced_profile_written_offline),
        ("estimate_tokens basic",           test_estimate_tokens_basic),
    ]

    print("\n" + "=" * 60)
    print("Running test_agents.py manually")
    print("=" * 60)
    passed = 0
    failed = 0
    for name, fn in tests:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            try:
                # Tests that need tmp_path receive it; others take no args.
                import inspect
                sig = inspect.signature(fn)
                if sig.parameters:
                    fn(tmp)
                else:
                    fn()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as exc:
                print(f"  FAIL  {name}")
                print(f"        {type(exc).__name__}: {exc}")
                failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed else 0)