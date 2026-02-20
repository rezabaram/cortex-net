"""Synthetic History Generator — generates realistic multi-month agent interaction data.

Creates diverse conversation histories that simulate a developer agent used over 6 months.
Used for benchmarking cortex-net's context assembly at scale (10-20K turns).

Topics span: coding, debugging, architecture, ops, preferences, planning, reviews.
Temporal structure: recurring themes, topic drift, knowledge accumulation.

Usage:
    gen = HistoryGenerator(seed=42)
    history = gen.generate(months=6, turns_per_day=(5, 30))
    # history.turns: list of Turn objects with timestamps, topics, content
    # history.ground_truth: queries with expected relevant turns
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


# ── Topic definitions ─────────────────────────────────────────────────────────

@dataclass
class Topic:
    """A conversation topic with templates and metadata."""
    id: str
    category: str  # coding, debugging, architecture, ops, preferences, planning
    keywords: list[str]
    user_templates: list[str]
    assistant_templates: list[str]
    # Topics this naturally leads to
    follow_ups: list[str] = field(default_factory=list)


TOPICS = [
    # ── Coding ──
    Topic(
        id="api_design",
        category="coding",
        keywords=["api", "endpoint", "rest", "graphql", "schema"],
        user_templates=[
            "I need to design a REST API for {service}. What's the best approach?",
            "Should we use GraphQL or REST for the {service} API?",
            "How should we version the {service} API?",
            "The {service} endpoint is getting too complex, how do we refactor?",
        ],
        assistant_templates=[
            "For {service}, I'd recommend REST with versioned paths. Here's why: {reason}",
            "GraphQL makes sense for {service} because {reason}. But watch out for N+1 queries.",
            "Use URL path versioning (/v2/{service}) — it's explicit and easy to deprecate.",
        ],
        follow_ups=["testing", "deployment", "api_auth"],
    ),
    Topic(
        id="data_modeling",
        category="coding",
        keywords=["database", "schema", "model", "migration", "sql", "postgres"],
        user_templates=[
            "How should we model {entity} in the database?",
            "We need to add {field} to the {entity} table. Best migration strategy?",
            "The {entity} table is getting slow with {count} rows. Indexing strategy?",
            "Should {entity} be a separate table or embedded in {parent}?",
        ],
        assistant_templates=[
            "For {entity}, use a normalized schema with {approach}. Add indexes on {field}.",
            "Migration strategy: add column as nullable, backfill, then add NOT NULL constraint.",
            "At {count} rows, you need a composite index on ({field}, created_at). Consider partitioning.",
        ],
        follow_ups=["performance", "deployment"],
    ),
    Topic(
        id="frontend",
        category="coding",
        keywords=["react", "component", "ui", "css", "state", "hook"],
        user_templates=[
            "The {component} component is re-rendering too much. How to fix?",
            "Best way to manage state for {feature}?",
            "Should we use {lib_a} or {lib_b} for {feature}?",
            "The {component} UI doesn't match the design. CSS issue.",
        ],
        assistant_templates=[
            "Use React.memo for {component} and move the state up to {parent}.",
            "For {feature}, use {lib_a} — it handles {reason} better than {lib_b}.",
            "The CSS issue is specificity. Use CSS modules or styled-components for {component}.",
        ],
        follow_ups=["testing", "performance"],
    ),

    # ── Debugging ──
    Topic(
        id="bug_investigation",
        category="debugging",
        keywords=["bug", "error", "crash", "fix", "reproduce", "stacktrace"],
        user_templates=[
            "Getting this error in {service}: {error}. Any ideas?",
            "Users are reporting {symptom} in {service}. Can't reproduce locally.",
            "{service} crashed in production with: {error}",
            "This test keeps failing intermittently: {test_name}",
        ],
        assistant_templates=[
            "That {error} usually means {cause}. Check {location} first.",
            "For intermittent failures, it's likely a race condition in {component}. Add logging at {location}.",
            "The crash is a null pointer in {component}. The fix is to add a guard at line {line}.",
        ],
        follow_ups=["testing", "monitoring"],
    ),
    Topic(
        id="performance",
        category="debugging",
        keywords=["slow", "latency", "profile", "optimize", "memory", "cpu"],
        user_templates=[
            "{service} response time went from {fast}ms to {slow}ms. What changed?",
            "Memory usage of {service} keeps growing. Possible leak?",
            "How do I profile {service} to find the bottleneck?",
            "The {query} query takes {slow}ms. Can we optimize it?",
        ],
        assistant_templates=[
            "Profile with {tool}. The bottleneck is likely in {component} — it's doing {bad_pattern}.",
            "Memory leak is probably in {component}. Use {tool} to track allocations.",
            "That query needs an index on {field}. Also, it's doing a full table scan on {table}.",
        ],
        follow_ups=["monitoring", "deployment"],
    ),

    # ── Architecture ──
    Topic(
        id="system_design",
        category="architecture",
        keywords=["architecture", "microservice", "monolith", "event", "queue"],
        user_templates=[
            "Should we split {service} into microservices?",
            "How do we handle {pattern} between {service_a} and {service_b}?",
            "We need to add {feature}. Where does it fit in the architecture?",
            "The monolith is getting hard to maintain. Migration strategy?",
        ],
        assistant_templates=[
            "Don't split {service} yet — extract {component} as a library first. Microservices add ops cost.",
            "Use event sourcing for {pattern}. {service_a} publishes events, {service_b} subscribes.",
            "Start with the Strangler Fig pattern: route new traffic to the new service, keep old running.",
        ],
        follow_ups=["deployment", "monitoring", "data_modeling"],
    ),
    Topic(
        id="api_auth",
        category="architecture",
        keywords=["auth", "oauth", "jwt", "token", "permission", "rbac"],
        user_templates=[
            "How should we handle authentication for {service}?",
            "JWT tokens are getting too large. Alternative?",
            "Need to add {permission} to the RBAC system.",
            "OAuth flow for {provider} — what's the right approach?",
        ],
        assistant_templates=[
            "Use short-lived JWTs (15min) + refresh tokens for {service}. Store sessions server-side.",
            "For {provider} OAuth, use authorization code flow with PKCE. Never implicit flow.",
            "Add {permission} as a scope, not a role. Roles are collections of scopes.",
        ],
        follow_ups=["api_design", "deployment"],
    ),

    # ── Ops ──
    Topic(
        id="deployment",
        category="ops",
        keywords=["deploy", "docker", "kubernetes", "ci", "cd", "rollback"],
        user_templates=[
            "How should we deploy {service} to production?",
            "The deploy of {service} failed. How to rollback?",
            "CI pipeline for {service} takes {minutes} minutes. How to speed up?",
            "Should we use {tool_a} or {tool_b} for container orchestration?",
        ],
        assistant_templates=[
            "Blue-green deployment for {service}. Keep the old version running until health checks pass.",
            "Rollback: `kubectl rollout undo deployment/{service}`. Check the events for why it failed.",
            "Parallelize the test stage and cache Docker layers. Should cut CI to {fast_minutes}min.",
        ],
        follow_ups=["monitoring"],
    ),
    Topic(
        id="monitoring",
        category="ops",
        keywords=["logs", "metrics", "alert", "dashboard", "observability", "trace"],
        user_templates=[
            "What metrics should we track for {service}?",
            "Getting too many false alerts for {service}. How to tune?",
            "Need a dashboard for {feature}. What should it show?",
            "How do we set up distributed tracing for {service}?",
        ],
        assistant_templates=[
            "Track: request rate, error rate, latency p50/p95/p99, saturation. USE method for {service}.",
            "For false alerts: increase window to 5min, require 3 consecutive failures, add auto-resolve.",
            "Use OpenTelemetry for tracing. Instrument at service boundaries and DB calls.",
        ],
        follow_ups=["performance", "deployment"],
    ),

    # ── Testing ──
    Topic(
        id="testing",
        category="coding",
        keywords=["test", "mock", "fixture", "coverage", "integration", "unit"],
        user_templates=[
            "How should we test {component}?",
            "Integration tests for {service} are flaky. How to stabilize?",
            "What's the right test strategy for {feature}?",
            "Coverage is at {percent}%. Is that enough?",
        ],
        assistant_templates=[
            "Unit test the logic, integration test the boundaries. For {component}, mock {dependency}.",
            "Flaky tests: use test containers for DB, retry transient failures, isolate test data.",
            "{percent}% is fine if it covers the critical paths. Don't chase 100% — test behavior, not lines.",
        ],
        follow_ups=["deployment"],
    ),

    # ── Preferences (personal, accumulated over time) ──
    Topic(
        id="code_style",
        category="preferences",
        keywords=["style", "naming", "convention", "format", "lint"],
        user_templates=[
            "I prefer {style_a} over {style_b} for {language}.",
            "Let's use {convention} for naming in {project}.",
            "Always use {pattern} for error handling, not {anti_pattern}.",
            "Don't use {thing} in this project — I've had bad experiences with it.",
        ],
        assistant_templates=[
            "Noted — {style_a} for {language}. I'll follow that in all suggestions.",
            "Got it, {convention} naming. Makes sense for consistency.",
            "Agreed, {pattern} is more robust. I'll avoid {anti_pattern}.",
        ],
        follow_ups=[],
    ),
    Topic(
        id="tool_preferences",
        category="preferences",
        keywords=["tool", "editor", "terminal", "workflow"],
        user_templates=[
            "I use {tool} for {task}. Don't suggest {other_tool}.",
            "My workflow is: {step1} → {step2} → {step3}.",
            "I switched from {old_tool} to {new_tool} because {reason}.",
        ],
        assistant_templates=[
            "Got it — {tool} for {task}. Will keep suggestions compatible.",
            "That workflow makes sense. {step2} could be automated with {suggestion}.",
        ],
        follow_ups=[],
    ),

    # ── Planning ──
    Topic(
        id="sprint_planning",
        category="planning",
        keywords=["sprint", "priority", "backlog", "estimate", "deadline"],
        user_templates=[
            "What should we prioritize this sprint?",
            "We have {days} days. Can we finish {feature_a} and {feature_b}?",
            "The deadline for {project} is {date}. Are we on track?",
            "{feature} is taking longer than expected. Should we cut scope?",
        ],
        assistant_templates=[
            "Priority: {feature_a} first (blocks others), then {feature_b}. Skip {feature_c} this sprint.",
            "In {days} days, {feature_a} is doable but {feature_b} needs scope cut. Suggest: {suggestion}.",
            "We're behind by ~{days} days. Cut {feature} to MVP and ship the rest on time.",
        ],
        follow_ups=["system_design", "api_design"],
    ),
    Topic(
        id="code_review",
        category="planning",
        keywords=["review", "pr", "pull request", "merge", "feedback"],
        user_templates=[
            "Can you review this approach for {feature}?",
            "PR is ready for {feature}. Main concern is {concern}.",
            "Got feedback on the {feature} PR: {feedback}. How to address?",
        ],
        assistant_templates=[
            "The approach is solid. One concern: {issue}. Consider {alternative}.",
            "Address the feedback by {action}. The reviewer's point about {concern} is valid.",
            "LGTM with one nit: {nit}. Ship it.",
        ],
        follow_ups=["testing", "deployment"],
    ),
]

TOPIC_MAP = {t.id: t for t in TOPICS}

# ── Filler values for templates ───────────────────────────────────────────────

SERVICES = ["user-service", "auth-service", "payment-service", "notification-service",
            "search-service", "analytics-service", "inventory-service", "order-service"]
ENTITIES = ["User", "Order", "Product", "Payment", "Session", "Event", "Subscription"]
COMPONENTS = ["AuthMiddleware", "CacheLayer", "EventBus", "RateLimiter", "SearchIndex",
              "NotificationQueue", "PaymentProcessor", "UserProfileCache"]
ERRORS = ["NullPointerException", "ConnectionTimeout", "OutOfMemoryError",
          "DeadlockDetected", "ValidationError", "RateLimitExceeded"]
TOOLS = ["DataDog", "Grafana", "Prometheus", "Jaeger", "Sentry", "PagerDuty"]
LANGUAGES = ["Python", "TypeScript", "Go", "Rust"]
PROJECTS = ["atlas", "cortex", "nexus", "forge", "pulse"]


def _fill_template(template: str, rng: random.Random) -> str:
    """Fill template placeholders with random values."""
    replacements = {
        "{service}": rng.choice(SERVICES),
        "{service_a}": rng.choice(SERVICES),
        "{service_b}": rng.choice(SERVICES),
        "{entity}": rng.choice(ENTITIES),
        "{parent}": rng.choice(ENTITIES),
        "{component}": rng.choice(COMPONENTS),
        "{error}": rng.choice(ERRORS),
        "{tool}": rng.choice(TOOLS),
        "{tool_a}": rng.choice(TOOLS),
        "{tool_b}": rng.choice(TOOLS),
        "{old_tool}": rng.choice(TOOLS),
        "{new_tool}": rng.choice(TOOLS),
        "{language}": rng.choice(LANGUAGES),
        "{project}": rng.choice(PROJECTS),
        "{feature}": f"{rng.choice(['user', 'admin', 'search', 'notification', 'payment'])} {rng.choice(['dashboard', 'API', 'module', 'flow', 'page'])}",
        "{feature_a}": f"{rng.choice(['auth', 'billing', 'search'])} {rng.choice(['refactor', 'v2', 'migration'])}",
        "{feature_b}": f"{rng.choice(['notification', 'analytics', 'export'])} {rng.choice(['feature', 'integration', 'endpoint'])}",
        "{feature_c}": f"{rng.choice(['dark mode', 'i18n', 'audit log'])}",
        "{field}": rng.choice(["user_id", "created_at", "status", "email", "tenant_id"]),
        "{count}": str(rng.choice([100_000, 500_000, 1_000_000, 10_000_000])),
        "{percent}": str(rng.randint(55, 92)),
        "{days}": str(rng.randint(3, 14)),
        "{date}": f"March {rng.randint(1, 28)}",
        "{minutes}": str(rng.randint(15, 45)),
        "{fast_minutes}": str(rng.randint(3, 8)),
        "{fast}": str(rng.randint(50, 200)),
        "{slow}": str(rng.randint(500, 5000)),
        "{line}": str(rng.randint(42, 500)),
        "{test_name}": f"test_{rng.choice(['auth', 'payment', 'search', 'cache'])}_{rng.choice(['timeout', 'race', 'flaky', 'order'])}",
        "{reason}": rng.choice(["better performance", "simpler API", "more maintainable", "team familiarity"]),
        "{cause}": rng.choice(["missing null check", "race condition", "stale cache", "config mismatch"]),
        "{location}": rng.choice(["the middleware", "the DB connection pool", "the event handler", "the cache layer"]),
        "{bad_pattern}": rng.choice(["N+1 queries", "synchronous blocking", "unbounded memory allocation"]),
        "{pattern}": rng.choice(["Result types", "try/except with specific errors", "guard clauses"]),
        "{anti_pattern}": rng.choice(["bare except", "silent failures", "global error handlers"]),
        "{approach}": rng.choice(["foreign keys", "soft deletes", "event sourcing", "CQRS"]),
        "{style_a}": rng.choice(["snake_case", "type hints everywhere", "dataclasses", "explicit returns"]),
        "{style_b}": rng.choice(["camelCase", "dynamic typing", "plain dicts", "implicit returns"]),
        "{convention}": rng.choice(["PascalCase for classes", "verb_noun for functions", "ALL_CAPS for constants"]),
        "{thing}": rng.choice(["ORM magic", "global mutable state", "monkey patching", "metaclasses"]),
        "{step1}": rng.choice(["write tests first", "branch from main", "check CI"]),
        "{step2}": rng.choice(["implement", "code review", "staging deploy"]),
        "{step3}": rng.choice(["merge to main", "production deploy", "monitor for 30min"]),
        "{task}": rng.choice(["debugging", "profiling", "deployment", "code review"]),
        "{other_tool}": rng.choice(["vim", "emacs", "sublime", "atom"]),
        "{suggestion}": rng.choice(["a shell script", "a Makefile target", "a GitHub Action"]),
        "{lib_a}": rng.choice(["zustand", "jotai", "redux"]),
        "{lib_b}": rng.choice(["mobx", "recoil", "context API"]),
        "{concern}": rng.choice(["error handling", "performance", "backwards compatibility"]),
        "{feedback}": rng.choice(["needs more tests", "simplify the interface", "handle edge cases"]),
        "{issue}": rng.choice(["tight coupling to the DB layer", "no retry logic", "missing validation"]),
        "{alternative}": rng.choice(["dependency injection", "exponential backoff", "schema validation"]),
        "{action}": rng.choice(["extracting a helper function", "adding a test case", "simplifying the interface"]),
        "{dependency}": rng.choice(["the database", "the API client", "the cache", "the event bus"]),
        "{nit}": rng.choice(["rename the variable", "add a docstring", "use a constant instead of magic number"]),
        "{permission}": rng.choice(["admin:write", "billing:read", "reports:export"]),
        "{provider}": rng.choice(["Google", "GitHub", "Okta", "Auth0"]),
        "{symptom}": rng.choice(["slow page loads", "missing data", "random logouts", "duplicate entries"]),
        "{query}": rng.choice(["user lookup", "order history", "search", "analytics rollup"]),
        "{table}": rng.choice(["users", "orders", "events", "sessions"]),
    }
    result = template
    for key, val in replacements.items():
        result = result.replace(key, val)
    return result


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Turn:
    """A single conversation turn."""
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    topic_id: str
    category: str
    session_id: str  # groups turns into conversations
    turn_in_session: int


@dataclass
class GroundTruthQuery:
    """A query with expected relevant turns for benchmarking."""
    query: str
    category: str
    relevant_turn_ids: list[str]
    description: str  # what kind of recall this tests


@dataclass
class SyntheticHistory:
    """Complete generated history with ground truth."""
    turns: list[Turn]
    queries: list[GroundTruthQuery]
    metadata: dict[str, Any]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "turns": [asdict(t) for t in self.turns],
            "queries": [asdict(q) for q in self.queries],
            "metadata": self.metadata,
        }
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path) -> SyntheticHistory:
        with open(path) as f:
            data = json.load(f)
        return cls(
            turns=[Turn(**t) for t in data["turns"]],
            queries=[GroundTruthQuery(**q) for q in data["queries"]],
            metadata=data["metadata"],
        )


# ── Generator ─────────────────────────────────────────────────────────────────

class HistoryGenerator:
    """Generates realistic multi-month agent interaction history."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.topics = TOPICS
        self._turn_counter = 0

    def _make_turn_id(self) -> str:
        self._turn_counter += 1
        return f"t{self._turn_counter:05d}"

    def _pick_topic(self, prev_topic: Topic | None = None) -> Topic:
        """Pick next topic, biased toward follow-ups of the previous topic."""
        if prev_topic and prev_topic.follow_ups and self.rng.random() < 0.3:
            follow_id = self.rng.choice(prev_topic.follow_ups)
            if follow_id in TOPIC_MAP:
                return TOPIC_MAP[follow_id]
        return self.rng.choice(self.topics)

    def _generate_session(
        self, session_id: str, start_time: float, num_exchanges: int
    ) -> list[Turn]:
        """Generate a single conversation session (sequence of exchanges)."""
        turns = []
        current_time = start_time
        topic = self._pick_topic()

        for i in range(num_exchanges):
            # Occasionally switch topics mid-conversation
            if i > 0 and self.rng.random() < 0.2:
                topic = self._pick_topic(topic)

            # User turn
            user_text = _fill_template(self.rng.choice(topic.user_templates), self.rng)
            turns.append(Turn(
                id=self._make_turn_id(),
                role="user",
                content=user_text,
                timestamp=current_time,
                topic_id=topic.id,
                category=topic.category,
                session_id=session_id,
                turn_in_session=i * 2,
            ))
            current_time += self.rng.uniform(1, 10)  # user thinks 1-10s

            # Assistant turn
            asst_text = _fill_template(self.rng.choice(topic.assistant_templates), self.rng)
            turns.append(Turn(
                id=self._make_turn_id(),
                role="assistant",
                content=asst_text,
                timestamp=current_time,
                topic_id=topic.id,
                category=topic.category,
                session_id=session_id,
                turn_in_session=i * 2 + 1,
            ))
            current_time += self.rng.uniform(0.5, 3)  # LLM responds 0.5-3s

        return turns

    def generate(
        self,
        months: int = 6,
        turns_per_day: tuple[int, int] = (5, 30),
    ) -> SyntheticHistory:
        """Generate a full synthetic history.

        Args:
            months: Number of months of history to generate.
            turns_per_day: (min, max) conversation turns per day.

        Returns:
            SyntheticHistory with turns and ground truth queries.
        """
        all_turns: list[Turn] = []
        start_date = datetime(2025, 6, 1)  # start 6 months ago
        session_counter = 0

        # Track which turns belong to which topics (for ground truth)
        topic_turns: dict[str, list[str]] = {t.id: [] for t in self.topics}
        # Track specific "memorable" events for cross-session recall queries
        memorable_events: list[dict] = []

        for day_offset in range(months * 30):
            date = start_date + timedelta(days=day_offset)

            # Skip some weekends (reduced activity)
            if date.weekday() >= 5 and self.rng.random() < 0.4:
                continue

            # Number of turns today
            day_turns = self.rng.randint(*turns_per_day)
            exchanges_today = day_turns // 2

            # 1-4 sessions per day
            num_sessions = self.rng.randint(1, min(4, max(1, exchanges_today // 3)))
            exchanges_per_session = max(1, exchanges_today // num_sessions)

            for s in range(num_sessions):
                session_counter += 1
                session_id = f"s{session_counter:04d}"

                # Session starts at a random work hour
                hour = self.rng.randint(8, 20)
                minute = self.rng.randint(0, 59)
                session_start = datetime(date.year, date.month, date.day, hour, minute)
                start_ts = session_start.timestamp()

                # Vary session length
                n_exchanges = max(1, exchanges_per_session + self.rng.randint(-2, 3))
                turns = self._generate_session(session_id, start_ts, n_exchanges)
                all_turns.extend(turns)

                # Track topic associations
                for turn in turns:
                    topic_turns[turn.topic_id].append(turn.id)

                # Mark some exchanges as "memorable" for ground truth
                if self.rng.random() < 0.1 and len(turns) >= 2:
                    memorable_events.append({
                        "turn_ids": [turns[0].id, turns[1].id],
                        "topic_id": turns[0].topic_id,
                        "date": date.isoformat(),
                        "content_preview": turns[0].content[:80],
                    })

        # Generate ground truth queries
        queries = self._generate_queries(all_turns, topic_turns, memorable_events)

        metadata = {
            "months": months,
            "total_turns": len(all_turns),
            "total_sessions": session_counter,
            "topics": len(self.topics),
            "queries": len(queries),
            "generated_at": time.time(),
        }

        return SyntheticHistory(turns=all_turns, queries=queries, metadata=metadata)

    def _generate_queries(
        self,
        all_turns: list[Turn],
        topic_turns: dict[str, list[str]],
        memorable_events: list[dict],
    ) -> list[GroundTruthQuery]:
        """Generate benchmark queries with content-aligned ground truth.

        Key principle: the query must semantically match the relevant turns.
        We extract distinctive phrases from actual turn content to build
        queries that cosine similarity CAN find — then cortex-net's job
        is to find them better (higher precision, fewer tokens).
        """
        queries = []
        turn_map = {t.id: t for t in all_turns}

        # Type 1: Session-contextual recall — find turns from the same session
        # The anchor turn is semantically close to the query, but OTHER relevant
        # turns are from the same session (conversationally related but different words).
        # This is where cortex-net can beat cosine: session context > keyword match.
        sessions: dict[str, list[str]] = {}
        for t in all_turns:
            sessions.setdefault(t.session_id, []).append(t.id)

        for topic in self.topics:
            tids = topic_turns.get(topic.id, [])
            if len(tids) < 4:
                continue

            # Find sessions that contain this topic's turns
            topic_sessions: dict[str, list[str]] = {}
            for tid in tids:
                t = turn_map[tid]
                topic_sessions.setdefault(t.session_id, []).append(tid)

            for sid, session_tids in topic_sessions.items():
                if len(session_tids) < 3:
                    continue
                # Anchor = first user turn in this session for this topic
                anchor_tid = session_tids[0]
                anchor = turn_map.get(anchor_tid)
                if not anchor or anchor.role != "user":
                    continue

                # Relevant = ALL turns from this session about this topic
                # (these share session context but may use different words)
                relevant = session_tids[:5]

                # Query derived from anchor content
                content = anchor.content
                phrases = content.replace("?", ".").replace("!", ".").split(".")
                phrase = phrases[0].strip() if phrases else content[:80]
                if len(phrase) < 15:
                    phrase = content[:80]

                query_text = phrase if "?" in phrase else phrase + "?"

                queries.append(GroundTruthQuery(
                    query=query_text,
                    category="topic_recall",
                    relevant_turn_ids=relevant,
                    description=f"Find session turns about: {phrase[:60]}",
                ))
                if len([q for q in queries if q.category == "topic_recall"]) >= 40:
                    break
            if len([q for q in queries if q.category == "topic_recall"]) >= 40:
                break

        # Type 2: Cross-session recall — use actual turn content as query
        for event in memorable_events[:15]:
            anchor_tid = event["turn_ids"][0]
            anchor = turn_map.get(anchor_tid)
            if not anchor:
                continue
            # Query = paraphrase of the turn content
            content = anchor.content
            if "?" in content:
                query_text = content  # already a question
            else:
                words = content.split()[:12]
                query_text = " ".join(words) + "?"
            queries.append(GroundTruthQuery(
                query=query_text,
                category="cross_session",
                relevant_turn_ids=event["turn_ids"],
                description=f"Find specific discussion from {event['date']}",
            ))

        # Type 3: Preference recall — use actual preference content
        pref_turns = topic_turns.get("code_style", []) + topic_turns.get("tool_preferences", [])
        if len(pref_turns) >= 3:
            relevant = self.rng.sample(pref_turns, min(5, len(pref_turns)))
            anchor = turn_map.get(relevant[0])
            if anchor:
                phrase = anchor.content.split(".")[0].strip()[:80]
                queries.append(GroundTruthQuery(
                    query=f"{phrase}?",
                    category="preference_recall",
                    relevant_turn_ids=relevant,
                    description="Recall coding preferences",
                ))

        # Type 4: Failure pattern — use actual error content
        bug_turns = topic_turns.get("bug_investigation", [])
        if len(bug_turns) >= 4:
            for error in ERRORS[:5]:
                # Find turns that actually mention this error
                relevant = [tid for tid in bug_turns
                            if error.lower() in turn_map.get(tid, Turn("","","",0,"","","",0)).content.lower()]
                if len(relevant) < 2:
                    continue
                queries.append(GroundTruthQuery(
                    query=f"How did we fix the {error}?",
                    category="failure_pattern",
                    relevant_turn_ids=relevant[:5],
                    description=f"Recall past debugging of {error}",
                ))

        # Type 5: Decision recall — use actual architecture content
        arch_turns = topic_turns.get("system_design", []) + topic_turns.get("api_auth", [])
        if len(arch_turns) >= 3:
            for _ in range(min(3, len(arch_turns) // 3)):
                relevant = self.rng.sample(arch_turns, min(4, len(arch_turns)))
                anchor = turn_map.get(relevant[0])
                if anchor:
                    phrase = anchor.content.split(".")[0].strip()[:80]
                    queries.append(GroundTruthQuery(
                        query=f"{phrase}?",
                        category="decision_recall",
                        relevant_turn_ids=relevant,
                        description="Recall architecture decision",
                    ))

        return queries
