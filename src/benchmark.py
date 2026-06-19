"""benchmark.py — evaluate BaselineAgent vs AdvancedAgent.

Benchmark sections
------------------
6.1  Standard Benchmark   – data/conversations.json
6.2  Long-Context Stress  – data/advanced_long_context.json

Mandatory columns
-----------------
- Agent tokens only        : tokens used exclusively by this turn (message + reply)
- Prompt tokens processed  : cumulative context fed to the model each turn
- Cross-session recall     : 0 / 0.5 / 1 per recall question, averaged
- Response quality         : heuristic score 0–1 for non-recall turns
- Memory growth (bytes)    : size of User.md (0 for baseline)
- Compactions              : number of compact-memory compaction events
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read a JSON file that contains a list of conversation dicts."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("conversations", [data])
    return data


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 based on how many expected facts appear in *answer*.

    Rules:
    - All expected facts present  -> 1.0
    - At least half present       -> 0.5
    - Fewer than half present     -> 0.0
    """
    if not expected:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for fact in expected if fact.lower() in answer_lower)
    ratio = hits / len(expected)
    if ratio >= 1.0:
        return 1.0
    if ratio >= 0.5:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality heuristic for offline / deterministic responses.

    Scoring components (each 0-1, averaged):
    1. Relevance   : at least one expected keyword found
    2. Length      : not too short (>=10 chars) and not too long (<=500 chars)
    3. Coherence   : answer is not a raw echo of the question
    4. Fact density: proportion of expected facts present
    """
    if not answer:
        return 0.0

    answer_lower = answer.lower()

    # 1. relevance
    relevance = 1.0 if any(k.lower() in answer_lower for k in expected) else 0.3

    # 2. length score
    length = len(answer)
    if length < 10:
        length_score = 0.2
    elif length > 500:
        length_score = 0.7
    else:
        length_score = 1.0

    # 3. coherence: penalise raw echo responses
    if answer.startswith("(Baseline)") or answer.startswith("(Advanced) Bạn vừa nói"):
        coherence = 0.4
    else:
        coherence = 1.0

    # 4. fact density
    hits = sum(1 for k in expected if k.lower() in answer_lower)
    fact_density = hits / len(expected) if expected else 1.0

    return (relevance + length_score + coherence + fact_density) / 4.0


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Run *agent* over all *conversations* and return aggregated metrics.

    Steps per conversation:
    1. Feed all turns in the primary thread (train thread).
    2. Track agent_tokens_only and prompt_tokens_processed.
    3. Open a **fresh** thread to ask recall questions (cross-session test).
    4. Score recall and heuristic quality.
    5. Collect memory growth and compaction counts.
    """
    total_agent_tokens: int = 0
    total_prompt_tokens: int = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    total_memory_bytes: int = 0
    total_compactions: int = 0

    has_memory = hasattr(agent, "memory_file_size")

    for conv in conversations:
        user_id: str = conv["user_id"]
        turns: list[str] = conv.get("turns", [])
        recall_questions: list[dict[str, Any]] = conv.get("recall_questions", [])

        # Unique thread for training turns
        train_thread = f"train-{conv['id']}-{uuid.uuid4().hex[:6]}"

        # ----------------------------------------------------------------
        # Phase 1 – feed training turns
        # ----------------------------------------------------------------
        for turn in turns:
            agent.reply(user_id, train_thread, turn)

        # Accumulate token metrics from training thread
        total_agent_tokens += agent.token_usage(train_thread)
        total_prompt_tokens += agent.prompt_token_usage(train_thread)
        total_compactions += agent.compaction_count(train_thread)

        # ----------------------------------------------------------------
        # Phase 2 – cross-session recall in a fresh thread
        # ----------------------------------------------------------------
        recall_thread = f"recall-{conv['id']}-{uuid.uuid4().hex[:6]}"

        for rq in recall_questions:
            question: str = rq["question"]
            expected: list[str] = rq.get("expected_contains", [])

            result = agent.reply(user_id, recall_thread, question)
            answer: str = result.get("response", "")

            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

        # ----------------------------------------------------------------
        # Phase 3 – memory file size (advanced only)
        # ----------------------------------------------------------------
        if has_memory:
            total_memory_bytes += agent.memory_file_size(user_id)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=total_memory_bytes,
        compactions=total_compactions,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render benchmark results as a Markdown table."""
    header = (
        "| Agent                | Agent Tokens Only | Prompt Tokens Processed"
        " | Cross-Session Recall | Response Quality | Memory Growth (bytes) | Compactions |"
    )
    separator = (
        "|----------------------|------------------:|------------------------:"
        "|---------------------:|-----------------:|----------------------:|------------:|"
    )
    lines = [header, separator]
    for r in rows:
        lines.append(
            f"| {r.agent_name:<20} | {r.agent_tokens_only:>17} "
            f"| {r.prompt_tokens_processed:>23} "
            f"| {r.recall_score:>20.3f} "
            f"| {r.response_quality:>16.3f} "
            f"| {r.memory_growth_bytes:>21} "
            f"| {r.compactions:>11} |"
        )
    return "\n".join(lines)


def _section(title: str, rows: list[BenchmarkRow]) -> str:
    return f"\n### {title}\n\n{format_rows(rows)}\n"


def _analysis(standard: list[BenchmarkRow], stress: list[BenchmarkRow]) -> str:
    lines: list[str] = ["\n### Analysis\n"]

    def find(rows: list[BenchmarkRow], name: str):
        for r in rows:
            if r.agent_name.lower() == name.lower():
                return r
        return None

    for label, rows in [("Standard", standard), ("Stress", stress)]:
        baseline = find(rows, "Baseline")
        advanced = find(rows, "Advanced")
        if not baseline or not advanced:
            continue

        recall_delta = advanced.recall_score - baseline.recall_score
        prompt_delta = baseline.prompt_tokens_processed - advanced.prompt_tokens_processed
        lines.append(f"**{label} benchmark**")
        lines.append(f"- Recall improvement (Advanced - Baseline): {recall_delta:+.3f}")
        lines.append(f"- Prompt token savings via compact memory:  {prompt_delta:+d} tokens")
        lines.append(f"- Advanced compactions fired: {advanced.compactions}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run both benchmark suites and print comparison tables."""
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent if src_dir.name == "src" else src_dir

    config = load_config(project_root)

    data_dir = project_root / "data"
    standard_path = data_dir / "conversations.json"
    stress_path = data_dir / "advanced_long_context.json"

    standard_convs = load_conversations(standard_path)
    stress_convs = load_conversations(stress_path)

    print(f"\nLoaded {len(standard_convs)} standard conversation(s).")
    print(f"Loaded {len(stress_convs)} stress conversation(s).\n")

    # ----------------------------------------------------------------
    # 6.1  Standard Benchmark
    # ----------------------------------------------------------------
    print("=" * 70)
    print("## 6.1 Standard Benchmark")
    print("=" * 70)

    baseline_std = BaselineAgent(config=config, force_offline=True)
    advanced_std = AdvancedAgent(config=config, force_offline=True)

    std_rows = [
        run_agent_benchmark("Baseline", baseline_std, standard_convs, config),
        run_agent_benchmark("Advanced", advanced_std, standard_convs, config),
    ]
    print(_section("Standard — Baseline vs Advanced", std_rows))

    # ----------------------------------------------------------------
    # 6.2  Long-Context Stress Benchmark
    # ----------------------------------------------------------------
    print("=" * 70)
    print("## 6.2 Long-Context Stress Benchmark")
    print("=" * 70)

    # Tighten compact threshold so compaction is visible on the stress dataset
    stress_config = LabConfig(
        base_dir=config.base_dir,
        data_dir=config.data_dir,
        state_dir=config.state_dir,
        compact_threshold_tokens=300,
        compact_keep_messages=4,
        confidence_threshold=config.confidence_threshold,
        model=config.model,
        judge_model=config.judge_model,
    )

    baseline_stress = BaselineAgent(config=stress_config, force_offline=True)
    advanced_stress = AdvancedAgent(config=stress_config, force_offline=True)

    stress_rows = [
        run_agent_benchmark("Baseline", baseline_stress, stress_convs, stress_config),
        run_agent_benchmark("Advanced", advanced_stress, stress_convs, stress_config),
    ]
    print(_section("Stress — Baseline vs Advanced", stress_rows))

    print(_analysis(std_rows, stress_rows))


if __name__ == "__main__":
    main()