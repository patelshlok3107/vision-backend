"""
Coding benchmark runner for VISION Learning.

Uses Ollama itself as the evaluator (LLM-as-judge).
Prompts are deterministic; scores are consistent across runs.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Benchmark problem set (25 problems across 4 categories)
# ---------------------------------------------------------------------------

BENCHMARK_PROBLEMS = [
    # --- Correctness (10) ---
    {
        "id": "c01", "category": "correctness",
        "prompt": "Write a Python function that returns the nth Fibonacci number using dynamic programming.",
        "criteria": "Must use memoization or iterative DP. Must handle n=0 and n=1 edge cases. O(n) time.",
    },
    {
        "id": "c02", "category": "correctness",
        "prompt": "Write a JavaScript function to deeply clone a nested object without using JSON.parse/stringify.",
        "criteria": "Must handle arrays, nested objects, null, and primitives. Must not use JSON methods.",
    },
    {
        "id": "c03", "category": "correctness",
        "prompt": "Write a Python function to binary search a sorted list and return the index or -1.",
        "criteria": "Correct binary search implementation. Returns index if found, -1 if not. Handles empty list.",
    },
    {
        "id": "c04", "category": "correctness",
        "prompt": "Write a TypeScript function that flattens a nested array of unknown depth.",
        "criteria": "Works for arbitrary depth. Uses TypeScript types properly. Handles empty arrays.",
    },
    {
        "id": "c05", "category": "correctness",
        "prompt": "Write SQL to find the second-highest salary from an employees table.",
        "criteria": "Correct result for all edge cases including ties. Works in PostgreSQL.",
    },
    {
        "id": "c06", "category": "correctness",
        "prompt": "Implement a stack using two queues in Python.",
        "criteria": "push/pop operations are correct. Handles empty stack scenario.",
    },
    {
        "id": "c07", "category": "correctness",
        "prompt": "Write a regex pattern to validate an email address.",
        "criteria": "Handles standard email formats, rejects invalid ones. Well-commented.",
    },
    {
        "id": "c08", "category": "correctness",
        "prompt": "Write a Python context manager that times code execution.",
        "criteria": "Uses __enter__/__exit__ or contextlib.contextmanager. Prints elapsed time.",
    },
    {
        "id": "c09", "category": "correctness",
        "prompt": "Write a debounce function in JavaScript/TypeScript.",
        "criteria": "Correctly delays execution. Resets timer on repeated calls. Returns cleanup.",
    },
    {
        "id": "c10", "category": "correctness",
        "prompt": "Write a Python function to detect if a string is a palindrome, ignoring case and non-alphanumeric characters.",
        "criteria": "Case insensitive. Ignores spaces and punctuation. Returns True/False.",
    },
    # --- Security (8) ---
    {
        "id": "s01", "category": "security",
        "prompt": "Write a Django view that handles user login securely.",
        "criteria": "Uses Django's authenticate(). No plaintext passwords. CSRF protection. Rate limiting mention.",
    },
    {
        "id": "s02", "category": "security",
        "prompt": "Write a Python function to hash a password for storage.",
        "criteria": "Uses bcrypt or argon2 or werkzeug. Never stores plaintext. Includes salt.",
    },
    {
        "id": "s03", "category": "security",
        "prompt": "Write an Express.js middleware to prevent SQL injection in a search endpoint.",
        "criteria": "Uses parameterised queries. Never string-concatenates user input into SQL.",
    },
    {
        "id": "s04", "category": "security",
        "prompt": "What are the top 5 OWASP security vulnerabilities and how do you prevent each?",
        "criteria": "Correct identification of OWASP Top 10 items. Concrete prevention strategies.",
    },
    {
        "id": "s05", "category": "security",
        "prompt": "Write a secure file upload handler in Python/Django.",
        "criteria": "Validates file type by content not extension. Sets size limit. Sanitises filename. No path traversal.",
    },
    {
        "id": "s06", "category": "security",
        "prompt": "How do you prevent XSS attacks in a React application?",
        "criteria": "Explains React's default escaping. Warns about dangerouslySetInnerHTML. CSP headers. Input validation.",
    },
    {
        "id": "s07", "category": "security",
        "prompt": "Write a JWT authentication middleware for Express.js.",
        "criteria": "Verifies token signature. Checks expiration. Does not expose sensitive info in errors.",
    },
    {
        "id": "s08", "category": "security",
        "prompt": "Write a safe SQL query builder in Python that prevents SQL injection.",
        "criteria": "Uses parameterised queries. Never formats user input into SQL strings.",
    },
    # --- Code Quality (4) ---
    {
        "id": "q01", "category": "quality",
        "prompt": "Refactor this code to follow SOLID principles: [monolithic class handling DB, email, auth, and business logic]",
        "criteria": "Identifies SRP, OCP violations. Proposes separated classes. Maintains functionality.",
    },
    {
        "id": "q02", "category": "quality",
        "prompt": "What is the difference between synchronous and asynchronous programming? Give Python examples.",
        "criteria": "Clear explanation. Correct async/await example. Mentions event loop, I/O bound tasks.",
    },
    {
        "id": "q03", "category": "quality",
        "prompt": "Explain the difference between SQL and NoSQL databases with use cases.",
        "criteria": "Accurate comparison. Real-world use cases. Honest about trade-offs.",
    },
    {
        "id": "q04", "category": "quality",
        "prompt": "Write a React component that fetches and displays paginated data.",
        "criteria": "Handles loading/error states. Implements pagination controls. Clean component structure.",
    },
    # --- Architecture (3) ---
    {
        "id": "a01", "category": "architecture",
        "prompt": "Design a REST API for a blog platform with users, posts, and comments.",
        "criteria": "Proper REST conventions. Authentication strategy. CRUD endpoints. Pagination design.",
    },
    {
        "id": "a02", "category": "architecture",
        "prompt": "How would you implement rate limiting in a Django REST API?",
        "criteria": "Mentions DRF throttling or Redis-based rate limiting. Token bucket or sliding window explanation.",
    },
    {
        "id": "a03", "category": "architecture",
        "prompt": "Explain how you would structure a large React application with multiple teams.",
        "criteria": "Feature-based folder structure. State management strategy. Shared component library. Code splitting.",
    },
]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def _evaluate_answer(prompt: str, answer: str, criteria: str) -> int:
    """
    Use Ollama as LLM-as-judge to score an answer 0-100.
    Returns 0 on failure.
    """
    from ai.services.ollama_client import client, OllamaError

    judge_prompt = (
        f"You are a strict technical code reviewer. "
        f"Score the following answer on a scale of 0-100 based on the criteria.\n\n"
        f"Question: {prompt}\n\n"
        f"Evaluation Criteria: {criteria}\n\n"
        f"Answer:\n{answer[:2000]}\n\n"
        f"Respond with only a single integer between 0 and 100. No explanation."
    )
    try:
        result = client.chat(
            [{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            stream=False,
        )
        score_str = (result or "0").strip().split()[0]
        return max(0, min(100, int(score_str)))
    except (OllamaError, ValueError, Exception) as e:
        logger.warning("[BENCHMARK] Eval failed: %s", e)
        return 0


def run_coding_benchmark_sync(quick: bool = False) -> dict:
    """
    Run the benchmark suite. quick=True uses a 5-problem subset for daily checks.
    Returns dict: {category: avg_score, overall: avg, problems: [detail…]}
    """
    from ai.services.ollama_client import client, OllamaError

    problems = BENCHMARK_PROBLEMS if not quick else BENCHMARK_PROBLEMS[:5]
    category_scores: dict[str, list[int]] = {}
    problem_results = []

    for prob in problems:
        logger.debug("[BENCHMARK] Running %s", prob["id"])
        try:
            answer = client.chat(
                [{"role": "user", "content": prob["prompt"]}],
                temperature=0.1,
                stream=False,
            )
        except OllamaError as e:
            logger.warning("[BENCHMARK] Chat failed for %s: %s", prob["id"], e)
            answer = ""

        score = _evaluate_answer(prob["prompt"], answer or "", prob["criteria"])
        category_scores.setdefault(prob["category"], []).append(score)
        problem_results.append({
            "id": prob["id"],
            "category": prob["category"],
            "score": score,
        })
        logger.debug("[BENCHMARK] %s → %d/100", prob["id"], score)

    # Aggregate per category
    cat_avgs = {cat: round(sum(scores) / len(scores), 1) for cat, scores in category_scores.items()}

    all_scores = [r["score"] for r in problem_results]
    overall = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0

    return {
        "overall": overall,
        "categories": cat_avgs,
        "problems_run": len(problem_results),
        "quick_mode": quick,
    }
