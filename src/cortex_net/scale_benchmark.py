"""Scale Benchmark — measures cortex-net vs baselines on large history retrieval.

Loads synthetic 6-month history into MemoryBank, runs ground truth queries,
compares retrieval quality and token efficiency across methods.

Baselines:
- cosine_topk: standard RAG with sentence-transformer cosine similarity
- bm25: keyword-based retrieval (TF-IDF approximation)
- random: random sample (control)
- recency: most recent turns (sliding window)
- cortex_net: Memory Gate + Situation Encoder (our system)

Metrics per query:
- Precision@k, Recall@k, MRR (mean reciprocal rank)
- Context size (chars, estimated tokens)
- Retrieval latency (ms)

Usage:
    bench = ScaleBenchmark(k=5)
    bench.load_history(months=6)
    results = bench.run()
    results.print_summary()
    results.save("benchmark_results.json")
"""

from __future__ import annotations

import json
import time
import random
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from cortex_net.memory_gate import MemoryGate
from cortex_net.situation_encoder import SituationEncoder, extract_metadata_features
from cortex_net.synthetic_history import HistoryGenerator, SyntheticHistory, Turn, GroundTruthQuery


# ── Metrics ───────────────────────────────────────────────────────────────────

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k that are relevant."""
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in relevant_ids) / len(top)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant items found in top-k."""
    if not relevant_ids:
        return 0.0
    top = retrieved_ids[:k]
    return sum(1 for r in top if r in relevant_ids) / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean reciprocal rank — 1/position of first relevant result."""
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return len(text) // 4


# ── Result structures ─────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Result of one query against one method."""
    query: str
    query_category: str
    method: str
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr_score: float = 0.0
    context_chars: int = 0
    context_tokens: int = 0
    retrieval_ms: float = 0.0
    retrieved_ids: list[str] = field(default_factory=list)
    relevant_ids: list[str] = field(default_factory=list)


@dataclass
class MethodSummary:
    """Aggregated results for one retrieval method."""
    method: str
    avg_precision_at_3: float = 0.0
    avg_precision_at_5: float = 0.0
    avg_recall_at_3: float = 0.0
    avg_recall_at_5: float = 0.0
    avg_mrr: float = 0.0
    avg_context_chars: int = 0
    avg_context_tokens: int = 0
    avg_retrieval_ms: float = 0.0
    total_queries: int = 0
    # Per-category breakdown
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    query_results: list[QueryResult]
    summaries: dict[str, MethodSummary]
    history_stats: dict[str, Any]
    timestamp: float = 0.0

    def print_summary(self) -> None:
        """Print a formatted summary table."""
        print("\n" + "=" * 90)
        print("SCALE BENCHMARK RESULTS")
        print(f"History: {self.history_stats['total_turns']} turns, "
              f"{self.history_stats['total_sessions']} sessions, "
              f"{self.history_stats['months']} months")
        print(f"Queries: {self.history_stats['total_queries']}")
        print("=" * 90)

        # Main table
        header = f"{'Method':<15} {'P@3':>6} {'P@5':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6} {'Tokens':>8} {'Chars':>8} {'ms':>8}"
        print(f"\n{header}")
        print("-" * 90)

        for name in ["cortex_net", "cosine_topk", "bm25", "recency", "random"]:
            s = self.summaries.get(name)
            if not s:
                continue
            row = (f"{s.method:<15} {s.avg_precision_at_3:>6.3f} {s.avg_precision_at_5:>6.3f} "
                   f"{s.avg_recall_at_3:>6.3f} {s.avg_recall_at_5:>6.3f} {s.avg_mrr:>6.3f} "
                   f"{s.avg_context_tokens:>8} {s.avg_context_chars:>8} {s.avg_retrieval_ms:>8.1f}")
            print(row)

        # Token efficiency
        cn = self.summaries.get("cortex_net")
        cos = self.summaries.get("cosine_topk")
        if cn and cos and cos.avg_context_tokens > 0:
            ratio = cos.avg_context_tokens / max(cn.avg_context_tokens, 1)
            p3_diff = cn.avg_precision_at_3 - cos.avg_precision_at_3
            print(f"\n📊 cortex-net uses {ratio:.1f}x fewer tokens than cosine top-k")
            print(f"📊 cortex-net P@3 delta: {p3_diff:+.3f} vs cosine")

        # Per-category breakdown
        print(f"\n{'Category':<20}", end="")
        for name in ["cortex_net", "cosine_topk"]:
            print(f" {name + ' P@5':>15}", end="")
        print()
        print("-" * 60)

        categories = set()
        for s in self.summaries.values():
            categories.update(s.by_category.keys())

        for cat in sorted(categories):
            print(f"{cat:<20}", end="")
            for name in ["cortex_net", "cosine_topk"]:
                s = self.summaries.get(name)
                if s and cat in s.by_category:
                    print(f" {s.by_category[cat].get('precision_at_5', 0):>15.3f}", end="")
                else:
                    print(f" {'—':>15}", end="")
            print()

        print("=" * 90)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "query_results": [asdict(qr) for qr in self.query_results],
            "summaries": {k: asdict(v) for k, v in self.summaries.items()},
            "history_stats": self.history_stats,
            "timestamp": self.timestamp,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ── Retrieval methods ─────────────────────────────────────────────────────────

class RetrievalMethod:
    """Base class for retrieval methods."""

    name: str = "base"

    def retrieve(self, query_emb: torch.Tensor, k: int) -> list[tuple[str, float]]:
        """Return list of (turn_id, score) pairs."""
        raise NotImplementedError


class CosineTopK(RetrievalMethod):
    """Standard cosine similarity retrieval."""

    name = "cosine_topk"

    def __init__(self, turn_ids: list[str], embeddings: torch.Tensor):
        self.turn_ids = turn_ids
        self.embeddings = embeddings  # (N, dim), L2-normalized

    def retrieve(self, query_emb: torch.Tensor, k: int) -> list[tuple[str, float]]:
        # query_emb: (dim,)
        scores = self.embeddings @ query_emb  # (N,)
        topk = torch.topk(scores, min(k, len(self.turn_ids)))
        results = []
        for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
            results.append((self.turn_ids[idx], score))
        return results


class BM25Retrieval(RetrievalMethod):
    """Simple TF-IDF-like keyword retrieval."""

    name = "bm25"

    def __init__(self, turn_ids: list[str], texts: list[str]):
        self.turn_ids = turn_ids
        self.texts = [t.lower().split() for t in texts]
        # Build IDF
        self.vocab: dict[str, float] = {}
        n = len(self.texts)
        from collections import Counter
        df = Counter()
        for doc in self.texts:
            for word in set(doc):
                df[word] += 1
        import math
        for word, count in df.items():
            self.vocab[word] = math.log((n - count + 0.5) / (count + 0.5) + 1)

    def retrieve(self, query_emb: torch.Tensor, k: int, query_text: str = "") -> list[tuple[str, float]]:
        query_words = query_text.lower().split()
        scores = []
        for i, doc in enumerate(self.texts):
            score = 0.0
            doc_len = len(doc)
            for qw in query_words:
                if qw in self.vocab:
                    tf = doc.count(qw)
                    # BM25 formula (simplified)
                    score += self.vocab[qw] * (tf * 2.0) / (tf + 1.0 + 0.75 * (doc_len / 50))
            scores.append((self.turn_ids[i], score))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


class RecencyRetrieval(RetrievalMethod):
    """Most recent turns (sliding window baseline)."""

    name = "recency"

    def __init__(self, turn_ids: list[str]):
        self.turn_ids = turn_ids

    def retrieve(self, query_emb: torch.Tensor, k: int) -> list[tuple[str, float]]:
        # Return last k turns
        recent = self.turn_ids[-k:]
        return [(tid, 1.0 - i * 0.01) for i, tid in enumerate(reversed(recent))]


class RandomRetrieval(RetrievalMethod):
    """Random sample (control baseline)."""

    name = "random"

    def __init__(self, turn_ids: list[str], seed: int = 42):
        self.turn_ids = turn_ids
        self.rng = random.Random(seed)

    def retrieve(self, query_emb: torch.Tensor, k: int) -> list[tuple[str, float]]:
        sample = self.rng.sample(self.turn_ids, min(k, len(self.turn_ids)))
        return [(tid, 1.0 / (i + 1)) for i, tid in enumerate(sample)]


class CortexNetRetrieval(RetrievalMethod):
    """cortex-net: Situation Encoder + Memory Gate."""

    name = "cortex_net"

    def __init__(
        self,
        turn_ids: list[str],
        embeddings: torch.Tensor,
        encoder: SituationEncoder,
        gate: MemoryGate,
        device: str = "cpu",
    ):
        self.turn_ids = turn_ids
        self.embeddings = embeddings
        self.encoder = encoder
        self.gate = gate
        self.device = device

    def retrieve(
        self, query_emb: torch.Tensor, k: int,
        hist_emb: torch.Tensor | None = None,
        metadata: dict | None = None,
    ) -> list[tuple[str, float]]:
        # Build situation embedding
        if hist_emb is None:
            hist_emb = torch.zeros_like(query_emb)
        if metadata is None:
            metadata = {}

        meta_tensor = extract_metadata_features(metadata, "", []).to(self.device)

        with torch.no_grad():
            situation = self.encoder(query_emb, hist_emb, meta_tensor)
            # Score all memories through the gate
            scores = self.gate.score_memories(situation, self.embeddings)  # (N,)

        topk = torch.topk(scores, min(k, len(self.turn_ids)))
        results = []
        for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
            results.append((self.turn_ids[idx], score))
        return results


# ── Benchmark runner ──────────────────────────────────────────────────────────

class ScaleBenchmark:
    """Runs the scale benchmark comparing retrieval methods."""

    def __init__(self, k: int = 5, seed: int = 42, device: str = "cpu"):
        self.k = k
        self.seed = seed
        self.device = device
        self.history: SyntheticHistory | None = None
        self.text_encoder: SentenceTransformer | None = None

    def load_history(self, months: int = 6, turns_per_day: tuple[int, int] = (8, 25)) -> SyntheticHistory:
        """Generate or load synthetic history."""
        gen = HistoryGenerator(seed=self.seed)
        self.history = gen.generate(months=months, turns_per_day=turns_per_day)
        print(f"Generated history: {len(self.history.turns)} turns, "
              f"{self.history.metadata['total_sessions']} sessions, "
              f"{len(self.history.queries)} queries")
        return self.history

    def load_history_from_file(self, path: Path | str) -> SyntheticHistory:
        """Load history from saved JSON."""
        self.history = SyntheticHistory.load(Path(path))
        return self.history

    def run(self, state_dir: str | None = None) -> BenchmarkResults:
        """Run the full benchmark."""
        assert self.history is not None, "Call load_history() first"

        print("Encoding turns...")
        t0 = time.time()
        self.text_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

        # Encode all turns (ground truth references both user and assistant)
        all_turns = self.history.turns
        turn_texts = [t.content for t in all_turns]
        turn_ids = [t.id for t in all_turns]

        embeddings = self.text_encoder.encode(
            turn_texts, convert_to_tensor=True, show_progress_bar=True
        ).to(self.device)
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        print(f"Encoded {len(all_turns)} turns in {time.time() - t0:.1f}s")

        # Initialize retrieval methods
        methods: dict[str, RetrievalMethod] = {}

        # 1. Cosine top-k
        methods["cosine_topk"] = CosineTopK(turn_ids, embeddings)

        # 2. BM25
        methods["bm25"] = BM25Retrieval(turn_ids, turn_texts)

        # 3. Recency
        methods["recency"] = RecencyRetrieval(turn_ids)

        # 4. Random
        methods["random"] = RandomRetrieval(turn_ids, seed=self.seed)

        # 5. cortex-net
        dim = embeddings.shape[1]
        encoder = SituationEncoder(text_dim=dim, output_dim=dim, hidden_dim=512, dropout=0.1).to(self.device)
        gate = MemoryGate(situation_dim=dim, memory_dim=dim).to(self.device)

        # Load trained weights if available
        if state_dir:
            sd = Path(state_dir)
            enc_path = sd / "situation_encoder.pt"
            gate_path = sd / "memory_gate.pt"
            if enc_path.exists():
                try:
                    encoder.load_state_dict(torch.load(enc_path, weights_only=True))
                    print(f"Loaded trained Situation Encoder from {enc_path}")
                except Exception:
                    print("Could not load encoder weights, using untrained")
            if gate_path.exists():
                try:
                    gate.load_state_dict(torch.load(gate_path, weights_only=True))
                    print(f"Loaded trained Memory Gate from {gate_path}")
                except Exception:
                    print("Could not load gate weights, using untrained")

        methods["cortex_net"] = CortexNetRetrieval(turn_ids, embeddings, encoder, gate, self.device)

        # Run queries
        print(f"\nRunning {len(self.history.queries)} queries against {len(methods)} methods...")
        all_results: list[QueryResult] = []

        for qi, query in enumerate(self.history.queries):
            query_emb = self.text_encoder.encode(query.query, convert_to_tensor=True).to(self.device)
            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=0)
            relevant = set(query.relevant_turn_ids)

            for method_name, method in methods.items():
                t_start = time.time()

                if method_name == "bm25":
                    retrieved = method.retrieve(query_emb, self.k, query_text=query.query)
                elif method_name == "cortex_net":
                    retrieved = method.retrieve(query_emb, self.k)
                else:
                    retrieved = method.retrieve(query_emb, self.k)

                elapsed_ms = (time.time() - t_start) * 1000
                retrieved_ids = [r[0] for r in retrieved]

                # Calculate context size (chars + estimated tokens)
                turn_map = {t.id: t for t in all_turns}
                context_text = "\n".join(
                    turn_map[tid].content for tid in retrieved_ids if tid in turn_map
                )

                qr = QueryResult(
                    query=query.query,
                    query_category=query.category,
                    method=method_name,
                    precision_at_3=precision_at_k(retrieved_ids, relevant, 3),
                    precision_at_5=precision_at_k(retrieved_ids, relevant, 5),
                    recall_at_3=recall_at_k(retrieved_ids, relevant, 3),
                    recall_at_5=recall_at_k(retrieved_ids, relevant, 5),
                    mrr_score=mrr(retrieved_ids, relevant),
                    context_chars=len(context_text),
                    context_tokens=estimate_tokens(context_text),
                    retrieval_ms=elapsed_ms,
                    retrieved_ids=retrieved_ids,
                    relevant_ids=list(relevant),
                )
                all_results.append(qr)

            if (qi + 1) % 10 == 0:
                print(f"  {qi + 1}/{len(self.history.queries)} queries done")

        # Aggregate results
        summaries = self._aggregate(all_results)

        return BenchmarkResults(
            query_results=all_results,
            summaries=summaries,
            history_stats={
                "total_turns": len(self.history.turns),
                "indexed_turns": len(all_turns),
                "total_sessions": self.history.metadata["total_sessions"],
                "months": self.history.metadata["months"],
                "total_queries": len(self.history.queries),
                "k": self.k,
            },
            timestamp=time.time(),
        )

    def _aggregate(self, results: list[QueryResult]) -> dict[str, MethodSummary]:
        """Aggregate per-query results into method summaries."""
        by_method: dict[str, list[QueryResult]] = defaultdict(list)
        for r in results:
            by_method[r.method].append(r)

        summaries = {}
        for method_name, qrs in by_method.items():
            n = len(qrs)
            summary = MethodSummary(
                method=method_name,
                avg_precision_at_3=sum(q.precision_at_3 for q in qrs) / n,
                avg_precision_at_5=sum(q.precision_at_5 for q in qrs) / n,
                avg_recall_at_3=sum(q.recall_at_3 for q in qrs) / n,
                avg_recall_at_5=sum(q.recall_at_5 for q in qrs) / n,
                avg_mrr=sum(q.mrr_score for q in qrs) / n,
                avg_context_chars=int(sum(q.context_chars for q in qrs) / n),
                avg_context_tokens=int(sum(q.context_tokens for q in qrs) / n),
                avg_retrieval_ms=sum(q.retrieval_ms for q in qrs) / n,
                total_queries=n,
            )

            # Per-category
            by_cat: dict[str, list[QueryResult]] = defaultdict(list)
            for q in qrs:
                by_cat[q.query_category].append(q)

            for cat, cat_qrs in by_cat.items():
                cn = len(cat_qrs)
                summary.by_category[cat] = {
                    "precision_at_3": sum(q.precision_at_3 for q in cat_qrs) / cn,
                    "precision_at_5": sum(q.precision_at_5 for q in cat_qrs) / cn,
                    "recall_at_3": sum(q.recall_at_3 for q in cat_qrs) / cn,
                    "recall_at_5": sum(q.recall_at_5 for q in cat_qrs) / cn,
                    "mrr": sum(q.mrr_score for q in cat_qrs) / cn,
                    "count": cn,
                }

            summaries[method_name] = summary

        return summaries


# ── Training on synthetic data ─────────────────────────────────────────────────

class SyntheticTrainer:
    """Train cortex-net components on synthetic history ground truth."""

    def __init__(
        self,
        encoder: SituationEncoder,
        gate: MemoryGate,
        text_encoder: SentenceTransformer,
        device: str = "cpu",
    ):
        self.encoder = encoder.to(device)
        self.gate = gate.to(device)
        self.text_encoder = text_encoder
        self.device = device

    def train(
        self,
        history: "SyntheticHistory",
        epochs: int = 30,
        lr: float = 1e-3,
        neg_ratio: int = 10,
        seed: int = 42,
    ) -> list[float]:
        """Train on ground truth queries. Returns loss history."""
        rng = random.Random(seed)

        # Pre-embed all user turns
        user_turns = [t for t in history.turns if t.role == "user"]
        turn_texts = [t.content for t in user_turns]
        turn_ids = [t.id for t in user_turns]
        turn_id_set = set(turn_ids)

        embeddings = self.text_encoder.encode(
            turn_texts, convert_to_tensor=True, show_progress_bar=False
        ).to(self.device)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        id_to_idx = {tid: i for i, tid in enumerate(turn_ids)}

        # Build training pairs from ground truth
        pairs: list[tuple[torch.Tensor, list[int], list[int]]] = []
        for query in history.queries:
            q_emb = self.text_encoder.encode(
                query.query, convert_to_tensor=True
            ).to(self.device)
            q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=0)

            pos_indices = [id_to_idx[tid] for tid in query.relevant_turn_ids if tid in id_to_idx]
            if not pos_indices:
                continue

            neg_pool = [i for i in range(len(turn_ids)) if i not in set(pos_indices)]
            pairs.append((q_emb, pos_indices, neg_pool))

        if not pairs:
            print("No training pairs found!")
            return []

        print(f"Training on {len(pairs)} queries, {sum(len(p[1]) for p in pairs)} positive pairs")

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.gate.parameters()),
            lr=lr,
        )

        loss_history = []
        for epoch in range(epochs):
            total_loss = 0.0
            rng.shuffle(pairs)

            for q_emb, pos_indices, neg_pool in pairs:
                optimizer.zero_grad()
                # Situation embedding
                hist_emb = torch.zeros_like(q_emb)
                meta = extract_metadata_features({}, "", []).to(self.device)
                situation = self.encoder(q_emb, hist_emb, meta)

                # All positives + sampled negatives in one loss
                pos_embs = embeddings[pos_indices]  # (P, D)
                negs = rng.sample(neg_pool, min(neg_ratio * len(pos_indices), len(neg_pool)))
                neg_embs = embeddings[negs]  # (N, D)

                loss = self.gate.contrastive_loss(situation, pos_embs, neg_embs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.gate.parameters()), 1.0
                )
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(sum(len(p[1]) for p in pairs), 1)
            loss_history.append(avg_loss)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

        return loss_history


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    """Run the benchmark from command line."""
    import argparse
    parser = argparse.ArgumentParser(description="cortex-net Scale Benchmark")
    parser.add_argument("--months", type=int, default=6, help="Months of history")
    parser.add_argument("--k", type=int, default=5, help="Top-k retrieval")
    parser.add_argument("--state-dir", type=str, default=None, help="Path to trained weights")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", action="store_true", help="Train on synthetic ground truth before benchmarking")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    args = parser.parse_args()

    bench = ScaleBenchmark(k=args.k, seed=args.seed)
    bench.load_history(months=args.months)

    if args.train:
        # Train cortex-net on ground truth, then benchmark
        from sentence_transformers import SentenceTransformer
        text_enc = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        dim = 384
        encoder = SituationEncoder(text_dim=dim, output_dim=dim, hidden_dim=512, dropout=0.1)
        gate = MemoryGate(situation_dim=dim, memory_dim=dim)

        trainer = SyntheticTrainer(encoder, gate, text_enc)
        print("\n--- Training cortex-net on synthetic ground truth ---")
        losses = trainer.train(bench.history, epochs=args.epochs)
        print(f"Final loss: {losses[-1]:.4f}\n")

        # Save trained weights to temp location and use them
        import tempfile
        tmpdir = tempfile.mkdtemp()
        torch.save(encoder.state_dict(), f"{tmpdir}/situation_encoder.pt")
        torch.save(gate.state_dict(), f"{tmpdir}/memory_gate.pt")
        results = bench.run(state_dir=tmpdir)
    else:
        results = bench.run(state_dir=args.state_dir)

    results.print_summary()
    results.save(args.output)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
