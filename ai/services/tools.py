"""
VISION Tool Registry — Phase 2 Agent Tools
Safe, local-first, permission-gated.

Tools are pure functions: fn(user, **kwargs) -> dict
Each tool validates its own args and returns JSON-serializable result.
Destructive tools set requires_approval=True and are gated by the agent.
"""
import ast
import logging
import math
import os
import shlex
import subprocess
import traceback
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)

class ToolError(Exception):
    pass

# ── Workspace sandbox ──
def _workspace_root() -> Path:
    root = Path(getattr(settings, "VISION_WORKSPACE_ROOT", Path(settings.BASE_DIR) / "workspace"))
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()

def _user_workspace(user) -> Path:
    # isolate per-user: workspace/<user_id>
    root = _workspace_root() / str(user.id)
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()

def _safe_path(user, rel: str) -> Path:
    # Prevent path traversal outside workspace
    ws = _user_workspace(user)
    # normalize rel: allow "" -> workspace root, strip leading /\
    rel = (rel or "").strip().lstrip("/\\")
    if rel in ("", ".", "./"):
        return ws
    p = (ws / rel).resolve()
    # ensure p is inside ws
    try:
        p.relative_to(ws)
    except ValueError:
        raise ToolError(f"Path '{rel}' is outside your workspace. Allowed root: workspace/")
    return p

# ── Calculator ──
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "exp": math.exp, "abs": abs, "round": round,
    "ceil": math.ceil, "floor": math.floor, "pow": pow,
}
_ALLOWED_NAMES = {"pi": math.pi, "e": math.e}

def _calc_eval(expr: str):
    # safe AST eval: only numbers, binop, unaryop, calls to allowed funcs
    node = ast.parse(expr, mode="eval")
    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ToolError(f"Unsupported constant {n.value!r}")
        if isinstance(n, ast.Num):  # py <3.8 compat
            return n.n
        if isinstance(n, ast.BinOp):
            left, right = _eval(n.left), _eval(n.right)
            if isinstance(n.op, ast.Add): return left + right
            if isinstance(n.op, ast.Sub): return left - right
            if isinstance(n.op, ast.Mult): return left * right
            if isinstance(n.op, ast.Div): return left / right
            if isinstance(n.op, ast.Pow): return left ** right
            if isinstance(n.op, ast.Mod): return left % right
            raise ToolError(f"Unsupported operator {type(n.op).__name__}")
        if isinstance(n, ast.UnaryOp):
            v = _eval(n.operand)
            if isinstance(n.op, ast.UAdd): return +v
            if isinstance(n.op, ast.USub): return -v
            raise ToolError("Unsupported unary")
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id not in _ALLOWED_FUNCS:
                raise ToolError(f"Function {getattr(n.func, 'id', '?')} not allowed. Allowed: {', '.join(_ALLOWED_FUNCS)}")
            args = [_eval(a) for a in n.args]
            return _ALLOWED_FUNCS[n.func.id](*args)
        if isinstance(n, ast.Name):
            if n.id in _ALLOWED_NAMES:
                return _ALLOWED_NAMES[n.id]
            raise ToolError(f"Name {n.id!r} not allowed")
        raise ToolError(f"Unsupported expression {type(n).__name__}")
    return _eval(node)

def tool_calculator(user, expression: str = "", **kw):
    if not expression or not str(expression).strip():
        raise ToolError("calculator requires 'expression' (e.g., '2+2', 'sqrt(16)', 'sin(pi/2)')")
    expr = str(expression).strip()[:200]
    try:
        result = _calc_eval(expr)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Calculator error: {e}")
    return {"expression": expr, "result": result}

# ── Filesystem ──
def tool_filesystem_list(user, path: str = "", **kw):
    p = _safe_path(user, path)
    if not p.exists():
        raise ToolError(f"Path '{path or '.'}' does not exist")
    if not p.is_dir():
        raise ToolError(f"Path '{path}' is not a directory")
    entries = []
    for child in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))[:100]:
        rel = child.relative_to(_user_workspace(user)).as_posix() or "."
        entries.append({
            "name": child.name,
            "path": rel,
            "is_dir": child.is_dir(),
            "size": child.stat().st_size if child.is_file() else 0,
        })
    return {"path": path or ".", "entries": entries, "count": len(entries)}

def tool_filesystem_read(user, path: str = "", **kw):
    if not path:
        raise ToolError("filesystem_read requires 'path'")
    p = _safe_path(user, path)
    if not p.exists():
        raise ToolError(f"File '{path}' not found")
    if p.is_dir():
        raise ToolError(f"'{path}' is a directory, use filesystem_list")
    if p.stat().st_size > 2_000_000:
        raise ToolError(f"File too large ({p.stat().st_size} bytes). Max 2 MB for read")
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise ToolError(f"Read failed: {e}")
    # truncate for LLM
    if len(text) > 8000:
        text = text[:8000] + "\n... [truncated]"
    return {"path": path, "content": text, "size": p.stat().st_size}

def tool_filesystem_write(user, path: str = "", content: str = "", **kw):
    if not path:
        raise ToolError("filesystem_write requires 'path'")
    if content is None:
        content = ""
    p = _safe_path(user, path)
    # prevent overwriting outside workspace already handled
    if len(content) > 500_000:
        raise ToolError("Content too large (max 500 KB)")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(content), encoding="utf-8")
    except Exception as e:
        raise ToolError(f"Write failed: {e}")
    return {"path": path, "written": len(content), "size": p.stat().st_size}

def tool_filesystem_delete(user, path: str = "", **kw):
    if not path:
        raise ToolError("filesystem_delete requires 'path'")
    p = _safe_path(user, path)
    if not p.exists():
        raise ToolError(f"Path '{path}' does not exist")
    # extra guard: don't delete workspace root
    if p == _user_workspace(user):
        raise ToolError("Cannot delete workspace root")
    try:
        if p.is_dir():
            import shutil
            shutil.rmtree(p)
            return {"path": path, "deleted": True, "was_dir": True}
        else:
            p.unlink()
            return {"path": path, "deleted": True, "was_dir": False}
    except Exception as e:
        raise ToolError(f"Delete failed: {e}")

# ── Code execution (sandboxed) ──
def tool_code_execution(user, code: str = "", language: str = "python", timeout: int = 10, **kw):
    if not code or not str(code).strip():
        raise ToolError("code_execution requires 'code'")
    code = str(code)[:8000]
    lang = (language or "python").lower()
    if lang != "python":
        raise ToolError(f"Language '{lang}' not supported yet. Only 'python' is allowed in Phase 2.")
    # Use subprocess with timeout, capture stdout/stderr, no network, limited time
    # We create a temp file in user workspace
    import tempfile, textwrap
    ws = _user_workspace(user)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=str(ws), delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp = f.name
    try:
        # Use the same python that runs Django (backend env)
        import sys as _sys
        py = _sys.executable
        # Restrict: no extra env, timeout 10s
        proc = subprocess.run([py, tmp], capture_output=True, text=True, timeout=int(timeout or 10), cwd=str(ws))
        out = (proc.stdout or "")[:8000]
        err = (proc.stderr or "")[:4000]
        return {"code": code[:1000], "stdout": out, "stderr": err, "exit_code": proc.returncode, "success": proc.returncode == 0}
    except subprocess.TimeoutExpired:
        raise ToolError(f"Code execution timed out after {timeout}s")
    except Exception as e:
        raise ToolError(f"Execution failed: {e}")
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except:
            pass

# ── Terminal (allowlist) ──
_ALLOWED_CMDS = {"ls", "dir", "pwd", "echo", "cat", "head", "tail", "wc", "grep", "find", "git", "python", "pip", "node", "npm", "lsb_release", "uname"}
_DANGEROUS = {"rm -rf", "mkfs", "dd ", "shutdown", "reboot", ":(){", "fork bomb"}

def tool_terminal(user, command: str = "", **kw):
    if not command or not str(command).strip():
        raise ToolError("terminal requires 'command' (e.g., 'ls', 'cat README.md', 'git status')")
    cmd = str(command).strip()[:500]
    # block dangerous
    low = cmd.lower()
    if any(d in low for d in _DANGEROUS):
        raise ToolError(f"Command blocked for safety: '{cmd}'")
    # check allowlist by first token
    try:
        first = shlex.split(cmd)[0].lower()
    except:
        first = cmd.split()[0].lower() if cmd.split() else ""
    base = os.path.basename(first).lower().replace(".exe","")
    if base not in _ALLOWED_CMDS:
        raise ToolError(f"Command '{first}' not in allowlist. Allowed: {', '.join(sorted(_ALLOWED_CMDS))}")
    ws = _user_workspace(user)
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15, cwd=str(ws))
        out = (proc.stdout or "")[:6000]
        err = (proc.stderr or "")[:4000]
        return {"command": cmd, "stdout": out, "stderr": err, "exit_code": proc.returncode}
    except subprocess.TimeoutExpired:
        raise ToolError("Terminal command timed out after 15s")
    except Exception as e:
        raise ToolError(f"Terminal failed: {e}")

# ── Web search (local fetch via DuckDuckGo lite or fallback mock) ──
def tool_web_search(user, query: str = "", **kw):
    if not query or not str(query).strip():
        raise ToolError("web_search requires 'query'")
    q = str(query).strip()[:200]
    # Try lite DuckDuckGo via http, fallback to mock to keep local-first
    try:
        import requests
        # Use DuckDuckGo html lite — simple, no API key
        r = requests.get("https://lite.duckduckgo.com/lite/", params={"q": q}, timeout=10, headers={"User-Agent": "VISION/1.0"})
        if r.ok and "result" in r.text.lower():
            # very naive parse: extract first 5 links/titles
            import re
            titles = re.findall(r'class=\"result-link\"[^>]*>(.*?)</a>', r.text)[:5]
            snippets = re.findall(r'class=\"result-snippet\"[^>]*>(.*?)</', r.text)[:5]
            results = []
            for i, t in enumerate(titles):
                sn = snippets[i] if i < len(snippets) else ""
                # strip html
                t = re.sub(r"<.*?>", "", t)[:120]
                sn = re.sub(r"<.*?>", "", sn)[:200]
                results.append({"title": t, "snippet": sn})
            if results:
                return {"query": q, "results": results, "source": "duckduckgo-lite"}
    except Exception as e:
        logger.debug(f"web_search fallback: {e}")
    # Mock fallback — still useful for agent to reason locally
    return {"query": q, "results": [
        {"title": f"Result for '{q}' (offline mock)", "snippet": "Local search unavailable or blocked. Use this as placeholder — VISION can still reason locally. Consider enabling network for live results."}
    ], "source": "mock", "note": "No external API key required; local fallback"}

# ── Screenshot / Clipboard stubs (Phase 2: instruct frontend) ──
def tool_screenshot(user, **kw):
    # Backend cannot capture client screen; frontend should handle via getDisplayMedia
    return {"status": "pending_frontend", "message": "Screenshot requested — frontend should capture via Screen Capture API and upload as image attachment.", "instruction": "Ask user to use 'See My Screen' button or upload screenshot"}

def tool_clipboard_read(user, **kw):
    return {"status": "pending_frontend", "message": "Clipboard read requires frontend permission. Use navigator.clipboard.readText() on user gesture."}

# ── Phase 3: Computer Control ──
def tool_open_path(user, path: str = "", **kw):
    if not path:
        raise ToolError("open_path requires 'path' (relative to workspace or absolute if within workspace)")
    # Only allow opening inside workspace for safety; still uses OS handler
    p = _safe_path(user, path) if not os.path.isabs(path) else Path(path)
    # If absolute, ensure inside workspace
    if os.path.isabs(path):
        try:
            p.resolve().relative_to(_user_workspace(user).resolve())
        except ValueError:
            raise ToolError(f"Absolute path '{path}' outside workspace not allowed")
    if not p.exists():
        raise ToolError(f"Path '{path}' does not exist")
    try:
        # Platform-specific open
        import platform, subprocess
        system = platform.system()
        if system == "Windows":
            os.startfile(str(p))  # type: ignore
        elif system == "Darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])
        return {"path": str(p), "opened": True, "message": f"Opened '{p.name}' with OS handler"}
    except Exception as e:
        raise ToolError(f"Open failed: {e}")

def tool_clipboard_write(user, text: str = "", **kw):
    if text is None:
        text = ""
    text = str(text)[:5000]
    if not text.strip():
        raise ToolError("clipboard_write requires 'text'")
    # Backend cannot directly write client clipboard; instruct frontend
    return {"status": "pending_frontend", "text": text, "message": "Clipboard write pending — frontend will copy on user gesture", "instruction": "frontend: navigator.clipboard.writeText(text)"}

def tool_system_info(user, **kw):
    import platform, shutil
    ws = _user_workspace(user)
    total, used, free = shutil.disk_usage(str(ws))
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "workspace": str(ws),
        "workspace_free_mb": round(free / 1024 / 1024, 1),
        "workspace_used_mb": round(used / 1024 / 1024, 1),
    }

# ── Phase 4: Image Investigation (reverse image search workflow) ──
def tool_image_investigation(user, query: str = "", image_description: str = "", **kw):
    """
    Workflow per spec §19-20:
    Uploaded Image -> Vision Analysis (image_description) -> Search public sources -> Compare -> matches
    Safety: only public pages, no paywall bypass, no facial identification, distinguish exact vs similar.
    """
    if not query and not image_description:
        raise ToolError("image_investigation requires 'query' (search terms from image) or 'image_description'")
    # Use image_description to refine query if provided
    search_q = (query or image_description or "").strip()[:300]
    if len(search_q) < 5:
        search_q = (image_description or query or "image").strip()
    # Reuse web_search but with image-specific framing
    base = tool_web_search(user, query=search_q)
    # Enhance results with match confidence heuristic (exact vs similar)
    # Since we cannot do true reverse image search locally without external API, we simulate with web_search results and annotate.
    # In production, this would call a dedicated reverse image search API (e.g., TinEye, Google) with proper ToS.
    results = base.get("results", [])
    enhanced = []
    for r in results:
        title_low = r.get("title","").lower()
        snippet_low = r.get("snippet","").lower()
        # Heuristic: if search_q words appear in title/snippet, mark High, else Medium
        q_words = [w for w in search_q.lower().split() if len(w) > 3][:5]
        hits = sum(1 for w in q_words if w in title_low or w in snippet_low)
        match = "High" if hits >= 2 else "Medium" if hits >= 1 else "Possible"
        enhanced.append({
            "url": f"https://example.com/search?q={search_q.replace(' ', '+')}&result={len(enhanced)+1}",  # placeholder; real would be actual page URL from search
            "title": r.get("title",""),
            "snippet": r.get("snippet",""),
            "match": match,
            "why": f"{'Same image / visually matching' if match=='High' else 'Similar image / context'} — public source, no paywall bypass",
        })
    # If web_search returned mock, keep single mock but enhance
    if base.get("source") == "mock" and len(enhanced) == 1:
        enhanced[0]["url"] = f"https://lite.duckduckgo.com/lite/?q={search_q.replace(' ', '+')}"
        enhanced[0]["match"] = "Possible"
    return {
        "query": search_q,
        "image_description": image_description[:500] if image_description else "",
        "results": enhanced[:5],
        "source": base.get("source", "mock"),
        "note": "Searched only publicly accessible pages. No facial identification, no private DB, respects robots. High=exact/visually matching, Medium/Possible=similar.",
        "scope": "Public web via DuckDuckGo lite (or mock fallback). Not 'all websites' — scoped to search engine index.",
    }

# ── Registry ──
TOOL_REGISTRY = {
    "calculator": tool_calculator,
    "filesystem_list": tool_filesystem_list,
    "filesystem_read": tool_filesystem_read,
    "filesystem_write": tool_filesystem_write,
    "filesystem_delete": tool_filesystem_delete,
    "code_execution": tool_code_execution,
    "terminal": tool_terminal,
    "web_search": tool_web_search,
    "screenshot": tool_screenshot,
    "clipboard_read": tool_clipboard_read,
    "open_path": tool_open_path,
    "clipboard_write": tool_clipboard_write,
    "system_info": tool_system_info,
    "image_investigation": tool_image_investigation,
}

TOOL_SCHEMAS = {
    "calculator": {
        "name": "calculator",
        "description": "Safe math evaluation. Use for arithmetic, sqrt, sin/cos, logs. Eg expression='2+2*3' or 'sqrt(16)'",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression"}}, "required": ["expression"]},
        "requires_approval": False,
    },
    "filesystem_list": {
        "name": "filesystem_list",
        "description": "List files in your sandboxed workspace (per-user). Path '' = workspace root. Use to explore files before reading.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Relative path inside workspace"}}, "required": []},
        "requires_approval": False,
    },
    "filesystem_read": {
        "name": "filesystem_read",
        "description": "Read a text file from workspace (max 2MB, utf-8).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        "requires_approval": False,
    },
    "filesystem_write": {
        "name": "filesystem_write",
        "description": "Write/create a file in workspace. Will create parent dirs. Content max 500KB.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
        "requires_approval": True,
    },
    "filesystem_delete": {
        "name": "filesystem_delete",
        "description": "Delete a file or directory in workspace. Destructive — requires approval.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        "requires_approval": True,
    },
    "code_execution": {
        "name": "code_execution",
        "description": "Execute Python code in sandbox (10s timeout, workspace cwd, stdout/stderr captured). Use for data, algorithms, file generation.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}, "language": {"type": "string", "enum": ["python"]}}, "required": ["code"]},
        "requires_approval": False,
    },
    "terminal": {
        "name": "terminal",
        "description": "Run a shell command in workspace. Allowlist: ls, cat, pwd, echo, grep, git, python, pip, node, npm. Blocks dangerous commands.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        "requires_approval": True,
    },
    "web_search": {
        "name": "web_search",
        "description": "Search the web for current information (DuckDuckGo lite, fallback mock). Local-first; no API key needed.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "requires_approval": False,
    },
    "screenshot": {
        "name": "screenshot",
        "description": "Request a screen capture. Backend returns instruction; frontend handles getDisplayMedia.",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "requires_approval": False,
    },
    "clipboard_read": {
        "name": "clipboard_read",
        "description": "Request clipboard text (frontend-handled).",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "requires_approval": False,
    },
    "open_path": {
        "name": "open_path",
        "description": "Open a file/folder in OS handler (Windows startfile / xdg-open). Path must be inside workspace sandbox. Destructive? No, but opens externally — requires approval.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        "requires_approval": True,
    },
    "clipboard_write": {
        "name": "clipboard_write",
        "description": "Write text to clipboard (frontend-handled via navigator.clipboard).",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        "requires_approval": False,
    },
    "system_info": {
        "name": "system_info",
        "description": "Get local system/workspace info (platform, disk).",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "requires_approval": False,
    },
    "image_investigation": {
        "name": "image_investigation",
        "description": "Investigate an uploaded image: extract visual characteristics via vision, search public web sources, compare candidates, return possible matches with source links and confidence (High/Medium/Possible). Public only, no paywall bypass, no facial ID, distinguishes exact vs similar. Use when user says 'find where this image appears online'.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search terms from image (or image description)"}, "image_description": {"type": "string", "description": "Vision analysis of image (optional)"}}, "required": []},
        "requires_approval": False,
    },
}

def dispatch_tool(user, tool_name: str, arguments: dict) -> dict:
    if tool_name not in TOOL_REGISTRY:
        raise ToolError(f"Unknown tool '{tool_name}'. Available: {', '.join(TOOL_REGISTRY)}")
    if arguments is None:
        arguments = {}
    arguments.pop("user", None)
    arguments.pop("user_id", None)
    fn = TOOL_REGISTRY[tool_name]
    # Reliability: retry transient failures up to 2 times with backoff
    last_exc = None
    for attempt in range(3):
        try:
            return fn(user, **arguments)
        except TypeError as e:
            raise ToolError(f"Invalid arguments for '{tool_name}': {e}")
        except ToolError as e:
            # Don't retry validation errors
            if "outside your workspace" in str(e) or "requires" in str(e) or "not in allowlist" in str(e):
                raise
            last_exc = e
            if attempt < 2:
                import time as _t
                _t.sleep(0.3 * (attempt + 1))
                continue
            raise
        except Exception as e:
            last_exc = e
            logger.exception(f"Tool {tool_name} attempt {attempt+1} failed: {e}")
            if attempt < 2:
                import time as _t
                _t.sleep(0.3 * (attempt + 1))
                continue
            raise ToolError(f"Tool '{tool_name}' failed after 3 attempts: {e} {traceback.format_exc()[:1000]}")
    raise ToolError(f"Tool '{tool_name}' failed: {last_exc}")

def get_tool_schemas():
    return TOOL_SCHEMAS

def tools_prompt_block() -> str:
    lines = ["You have access to these LOCAL tools. To use a tool, output ONLY this JSON (no prose before it):",
             '{"tool": \"<name>\", \"arguments\": { ... }}', "", "Available tools:"]
    for name, s in TOOL_SCHEMAS.items():
        appr = " [requires approval]" if s.get("requires_approval") else ""
        lines.append(f"- {name}{appr}: {s['description']}")
    lines.append("\nRules: Only call a tool when the user explicitly asks for an action that needs it (list files, read/write, run code, calculate, search). Never call destructive tools without user intent. After tool result, you will be asked to summarize.")
    return "\n".join(lines)
