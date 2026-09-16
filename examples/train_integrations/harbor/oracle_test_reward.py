"""
Oracle-checked "discriminating test" reward.

The regex-based `wrote_tests`/`ran_tests` bonuses in reward_shaping.py reward the
*presence* of a test command, which is gameable: the model writes ever more test
files while task success stays flat. This module rewards tests that are actually
*meaningful* -- i.e. that would distinguish a correct solution from a wrong one.

We use verified oracle solutions (extract_oracle_solutions.py output) as ground
truth. For a task's trajectory we:
  1. extract the test INPUTS the model created (stdin it fed to its solution),
  2. run the ORACLE on each input  -> the correct output,
  3. run a set of MUTANTS (known-wrong programs) on each input,
  4. an input *discriminates* if >=1 mutant's output differs from the oracle's,
  5. (optional) if the model asserted an explicit expected output, it is *valid*
     when it matches the oracle.

Reward = capped fraction of discriminating (and, if present, valid) test inputs.
This teaches the generic, transferable skill of writing tests that catch bugs
(edge-case selection), not merely emitting a test file.

Mutants default to AST mutations of the oracle, filtered to ones that actually
*run and are wrong* on the task's shipped I/O (so they are genuine, killable
negatives). Real wrong-solutions mined from teacher traces are a higher-quality
drop-in source (pass them in via `mutants`).

SAFETY: executes untrusted code (oracle, mutants, and implicitly the model's
chosen inputs) with a hard per-run timeout in a temp cwd. Not sandboxed -- run
inside the same container/sandbox the verifier already uses.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


# Reward runs in a thread-pool executor and spawns subprocesses; the harbor bridge WORKER
# lives on the same node. Unbounded reward subprocesses starve the bridge -> it 409s on log
# downloads -> trials fail -> generation collapses. This semaphore caps how many reward
# subprocesses run at once across all reward threads. Tune via ORACLE_TEST_MAX_CONCURRENCY.
_SUBPROC_SEM = threading.Semaphore(int(os.environ.get("ORACLE_TEST_MAX_CONCURRENCY", "8")))


# ------------------------------------------------------------------ execution --
def run_program(code: str, stdin: str, timeout: float) -> tuple[bool, str]:
    """Run python `code` with `stdin`. Returns (ok, stdout). ok=False on error/timeout.
    Subprocess spawns are gated by a global semaphore to protect the co-located bridge."""
    with _SUBPROC_SEM, tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "prog.py")
        with open(path, "w") as f:
            f.write(code)
        try:
            r = subprocess.run(
                [sys.executable, path], input=stdin, capture_output=True,
                text=True, timeout=timeout, cwd=td,
            )
        except subprocess.TimeoutExpired:
            return False, ""
        except Exception:
            return False, ""
        if r.returncode != 0:
            return False, ""
        return True, r.stdout


def _norm(s: str) -> str:
    lines = [ln.rstrip() for ln in s.replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


# -------------------------------------------------- extract model test inputs --
# model writes inputs as `cat > foo.txt << 'EOF' ... EOF` then `python sol.py < foo.txt`,
# or pipes `printf/echo ... | python sol.py`. We collect the stdin payloads.
_HEREDOC = re.compile(
    r"cat\s*>\s*(?P<path>\S+)\s*<<\s*['\"]?(?P<m>[A-Za-z_][A-Za-z0-9_]*)['\"]?\s*\n(?P<body>.*?)\n(?P=m)\b",
    re.DOTALL,
)
_REDIR_IN = re.compile(r"python[0-9.]*\s+\S+\.py\s*<\s*(?P<path>\S+)")

# The agent's keystroke commands live inside terminus JSON in each assistant turn as
# JSON strings (nested in `message`), so their newlines are ESCAPED. We must parse the
# JSON to recover keystrokes with the CORRECT escaping -- unescaping the raw blob would
# corrupt the nested JSON inside tests.json (double-escaping). Use brace-matching to pull
# balanced {...} objects out of mixed text (<think> ... {json} ...).
def _iter_balanced_objects(text: str):
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start:i + 1]


_CAST_INPUT = re.compile(r',\s*"i",\s*("(?:[^"\\]|\\.)*")')


def command_text_from_cast(cast_path: str) -> str:
    """Reliable source: the terminal recording. Returns the model's keystroke inputs as
    text with correct escaping (heredoc bodies intact). '' if the cast is missing."""
    import json as _json
    try:
        c = open(cast_path).read()
    except OSError:
        return ""
    ks = []
    for tok in _CAST_INPUT.findall(c):
        try:
            ks.append(_json.loads(tok))
        except Exception:
            ks.append(tok)
    return "\n".join(ks)


def command_text_from_steps(steps: Sequence[Mapping[str, Any]]) -> str:
    """Fallback source: parse terminus JSON commands out of trajectory steps. Less reliable
    than the cast (large turns may not parse), so used only when the cast is unavailable."""
    import json as _json
    parts: list[str] = []
    for st in steps:
        msg = st.get("message") or st.get("content") or ""
        if not isinstance(msg, str):
            continue
        found = False
        for cand in _iter_balanced_objects(msg):
            if '"commands"' not in cand:
                continue
            try:
                j = _json.loads(cand)
            except Exception:
                continue
            cmds = j.get("commands")
            if isinstance(cmds, list):
                for c in cmds:
                    if isinstance(c, dict) and isinstance(c.get("keystrokes"), str):
                        parts.append(c["keystrokes"])
                        found = True
        if not found:
            parts.append(msg)
    return "\n".join(parts)


def extract_model_test_inputs(command_text: str) -> list[str]:
    """Best-effort: return the distinct stdin payloads the model tested its solution on.
    Excludes the solution file itself; keeps files that are fed to the solution via `<`,
    plus any *.txt / test* heredocs (models often make input files then redirect).
    `command_text` from command_text_from_cast()/_from_steps()."""
    blob = command_text

    files: dict[str, str] = {}
    for m in _HEREDOC.finditer(blob):
        p = m.group("path")
        if p.endswith(".py"):
            continue  # that's code, not an input
        files[os.path.basename(p)] = m.group("body")

    redirected = {os.path.basename(m.group("path")) for m in _REDIR_IN.finditer(blob)}

    inputs: list[str] = []
    seen = set()
    # prefer files that were actually fed to the solution via `<`
    ordered = [f for f in files if f in redirected] + [f for f in files if f not in redirected]
    for f in ordered:
        body = files[f]
        key = _norm(body)
        if key and key not in seen:
            seen.add(key)
            inputs.append(body if body.endswith("\n") else body + "\n")
    return inputs


# ------------------------------------------- declared tests (explicit format) --
# STRONGER path: instruct the model (via the harness prompt, see PROMPT_SNIPPET)
# to write its tests to a single file as JSON  [{"input": "...", "expected": "..."}, ...].
# Then we can grade the model's *expected outputs* against the oracle (validity),
# not just input discrimination -- this rewards actually predicting the right answer.
PROMPT_SNIPPET = (
    "Before marking the task complete, write your own test cases to /app/tests.json "
    "as a JSON list of objects, each with \"input\" (the exact stdin) and \"expected\" "
    "(the exact stdout you expect a correct solution to produce). Choose inputs that "
    "cover edge cases and would catch a wrong solution."
)
_DECLARED_DEFAULT = "tests.json"


def extract_declared_tests(command_text: str,
                           filename: str = _DECLARED_DEFAULT) -> list[dict]:
    """Parse the model's declared test file (heredoc writing <filename>) into a list of
    {"input":..., "expected":...}. Returns [] if absent/unparseable. Takes the LAST write.
    `command_text` from command_text_from_cast()/_from_steps()."""
    blob = command_text
    bodies = [m.group("body") for m in _HEREDOC.finditer(blob)
              if os.path.basename(m.group("path")) == filename]
    if not bodies:
        return []
    import json as _json
    body = bodies[-1]
    try:
        data = _json.loads(body)
    except Exception:
        # tolerate JSONL
        data = []
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data.append(_json.loads(line))
            except Exception:
                return []
    out = []
    if isinstance(data, list):
        for d in data:
            if isinstance(d, dict) and "input" in d:
                inp = str(d["input"])
                if not inp.endswith("\n"):
                    inp += "\n"
                out.append({"input": inp,
                            "expected": None if d.get("expected") is None else str(d["expected"])})
    return out


# ----------------------------------------------------------------- mutants -----
class _IntBump(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return ast.copy_location(ast.Constant(value=node.value + 1), node)
        return node


_CMP_SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
             ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_BIN_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv}


class _CmpSwap(ast.NodeTransformer):
    def visit_Compare(self, node):
        self.generic_visit(node)
        node.ops = [_CMP_SWAP.get(type(o), type(o))() for o in node.ops]
        return node


class _BinSwap(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        t = type(node.op)
        if t in _BIN_SWAP:
            node.op = _BIN_SWAP[t]()
        return node


def make_mutants(oracle_code: str, shipped_io: Sequence[tuple[str, str]],
                 timeout: float, max_mutants: int = 6) -> list[str]:
    """Return AST-mutated copies of the oracle that (a) parse+run and (b) are WRONG on
    at least one shipped I/O pair (so they are genuine killable negatives)."""
    try:
        tree = ast.parse(oracle_code)
    except SyntaxError:
        return []
    candidates = []
    for T in (_IntBump, _CmpSwap, _BinSwap):
        try:
            mut = ast.unparse(ast.fix_missing_locations(T().visit(ast.parse(oracle_code))))
            candidates.append(mut)
        except Exception:
            continue
    kept = []
    for mut in candidates:
        if mut.strip() == oracle_code.strip():
            continue
        wrong = False
        for inp, exp in shipped_io[:4]:
            ok, out = run_program(mut, inp, timeout)
            if not ok or _norm(out) != _norm(exp):
                wrong = True
                break
        if wrong:
            kept.append(mut)
        if len(kept) >= max_mutants:
            break
    return kept


# ------------------------------------------------------------------- scoring ---
def extract_solution_versions(command_text: str, filename: str = "solution.py") -> list[str]:
    """All `cat > .../solution.py << EOF ... EOF` bodies, in the order written. Multiple
    versions = the model iterated on its solution."""
    return [m.group("body") for m in _HEREDOC.finditer(command_text)
            if os.path.basename(m.group("path")) == filename]


def caught_and_fixed_score(oracle_code: str, solution_versions: Sequence[str],
                           tests: Sequence[Mapping[str, Any]], timeout: float = 4.0,
                           target: int = 2, max_versions: int = 2, max_tests: int = 3):
    """The test-USE signal: count valid test inputs where an EARLIER solution version fails
    (output != oracle) and a LATER version passes -> the model used its test to catch & fix a
    real bug. Returns (reward[0,1], n_fixed). 0 if <2 solution versions."""
    versions = list(solution_versions)[-max_versions:]
    if len(versions) < 2 or not tests:
        return 0.0, 0
    n_fixed = 0
    for t in list(tests)[:max_tests]:
        inp = t["input"]
        ok, o_out = run_program(oracle_code, inp, timeout)
        if not ok:
            continue
        passed = []
        for v in versions:
            vok, v_out = run_program(v, inp, timeout)
            passed.append(vok and _norm(v_out) == _norm(o_out))
        # a fail followed later by a pass on the same input = caught & fixed
        if any((not passed[i]) and any(passed[j] for j in range(i + 1, len(passed)))
               for i in range(len(passed) - 1)):
            n_fixed += 1
    return min(1.0, n_fixed / max(1, target)), n_fixed


@dataclass
class TestRewardResult:
    n_test_inputs: int = 0
    n_ran_on_oracle: int = 0        # inputs the oracle handled (valid inputs)
    n_discriminating: int = 0       # inputs that separate oracle from >=1 mutant
    n_mutants: int = 0
    mutants_killed: int = 0         # distinct mutants caught by >=1 input
    n_with_expected: int = 0        # declared tests that carried an expected output
    n_valid: int = 0               # declared tests whose expected == oracle output
    n_solution_versions: int = 0    # how many times the model (re)wrote solution.py
    n_fixed: int = 0               # valid tests where an earlier sol failed & a later passed
    test_use_reward: float = 0.0    # caught-and-fixed reward [0,1]
    reward: float = 0.0
    detail: dict = field(default_factory=dict)


def score_tests(
    oracle_code: str,
    tests: Sequence[Mapping[str, Any]],
    mutants: Sequence[str],
    timeout: float = 6.0,
    target: int = 3,
    w_valid: float = 0.5,
    w_disc: float = 0.5,
) -> TestRewardResult:
    """Score declared tests [{"input", "expected"(optional)}].

    validity     = fraction of tests whose declared `expected` matches oracle(input)
                   -> rewards predicting the correct answer (real understanding).
    discrimination = fraction of inputs that separate oracle from >=1 mutant
                   -> rewards choosing bug-catching inputs.

    reward = w_valid * validity + w_disc * discrimination, each capped at `target`.
    If no test carries an expected output, validity is skipped and the weight shifts
    entirely to discrimination (falls back to the input-only regime)."""
    res = TestRewardResult(n_test_inputs=len(tests), n_mutants=len(mutants))
    if not tests:
        return res
    killed = set()
    for t in tests:
        inp = t["input"]
        ok, o_out = run_program(oracle_code, inp, timeout)
        if not ok:
            continue
        res.n_ran_on_oracle += 1
        exp = t.get("expected")
        if exp is not None:
            res.n_with_expected += 1
            if _norm(exp) == _norm(o_out):
                res.n_valid += 1
        discs = False
        for mi, mut in enumerate(mutants):
            mok, m_out = run_program(mut, inp, timeout)
            if (not mok) or _norm(m_out) != _norm(o_out):
                discs = True
                killed.add(mi)
        if discs:
            res.n_discriminating += 1
    res.mutants_killed = len(killed)

    disc_frac = min(1.0, res.n_discriminating / max(1, target)) if mutants else 0.0
    if res.n_with_expected > 0:
        valid_frac = min(1.0, res.n_valid / max(1, target))
        if mutants:
            res.reward = w_valid * valid_frac + w_disc * disc_frac
        else:
            res.reward = valid_frac          # no mutants -> validity carries it
    else:
        res.reward = disc_frac               # input-only -> discrimination only
    return res


def score_discriminating_tests(
    oracle_code: str,
    test_inputs: Sequence[str],
    mutants: Sequence[str],
    timeout: float = 6.0,
    target_discriminating: int = 3,
) -> TestRewardResult:
    """Core, deterministic scorer. reward in [0,1]:
    - 0 if no mutants (cannot judge) or no valid test inputs;
    - else fraction of discriminating inputs, capped at target, blended with the
      fraction of distinct mutants killed (coverage)."""
    res = TestRewardResult(n_test_inputs=len(test_inputs), n_mutants=len(mutants))
    if not test_inputs or not mutants:
        return res
    killed = set()
    for inp in test_inputs:
        ok, o_out = run_program(oracle_code, inp, timeout)
        if not ok:
            continue  # oracle can't run it -> not a usable test input
        res.n_ran_on_oracle += 1
        discs = False
        for mi, mut in enumerate(mutants):
            mok, m_out = run_program(mut, inp, timeout)
            # a crash on a mutant also counts as "caught" (input exposes different behavior)
            if (not mok) or _norm(m_out) != _norm(o_out):
                discs = True
                killed.add(mi)
        if discs:
            res.n_discriminating += 1
    res.mutants_killed = len(killed)
    disc_frac = min(1.0, res.n_discriminating / max(1, target_discriminating))
    cov_frac = res.mutants_killed / max(1, res.n_mutants)
    res.reward = 0.5 * disc_frac + 0.5 * cov_frac
    return res


def _read_io(tdj: str) -> list[tuple[str, str]]:
    import json as _json
    try:
        j = _json.load(open(tdj))
        return list(zip(j.get("inputs", []), j.get("outputs", [])))
    except Exception:
        return []


def load_shipped_io_for_trajectory(trajectory_path, task_data_dir: str = "") -> tuple[str, list[tuple[str, str]]]:
    """From .../<task>__<hash>/agent/trajectory.json derive (task_id, [(input,output),...]).
    Prefer the co-located trial copy; fall back to <task_data_dir>/<task>/tests/test_data.json
    (the trial dir is a staging copy that may NOT retain tests/test_data.json). ("", []) if
    unresolved."""
    p = str(trajectory_path)
    trial_dir = os.path.dirname(os.path.dirname(p))  # .../<task>__<hash>
    task = os.path.basename(trial_dir).split("__")[0]
    io = _read_io(os.path.join(trial_dir, "tests", "test_data.json"))
    if not io and task_data_dir:
        io = _read_io(os.path.join(task_data_dir, task, "tests", "test_data.json"))
    return task, io


# per-task mutant cache: the 8 samples of one prompt (and repeats across epochs)
# reuse the same oracle+mutants; keyed by (task, oracle-file mtime).
_MUTANT_CACHE: dict = {}


def _get_mutants(task, oracle_path, oracle_code, shipped_io, timeout, max_mutants=6):
    try:
        key = (task, os.path.getmtime(oracle_path))
    except OSError:
        key = (task, 0)
    if key not in _MUTANT_CACHE:
        _MUTANT_CACHE[key] = make_mutants(oracle_code, shipped_io, timeout, max_mutants)
    return _MUTANT_CACHE[key]


def discriminating_test_bonus(
    steps: Sequence[Mapping[str, Any]],
    task: str,
    oracle_dir: str,
    shipped_io: Sequence[tuple[str, str]],
    timeout: float = 6.0,
    bonus_scale: float = 0.05,
    target_discriminating: int = 3,
    mutants: Sequence[str] | None = None,
    trajectory_path: str = "",
    test_use_scale: float = 0.0,
    test_use_target: int = 2,
) -> tuple[float, TestRewardResult]:
    """Entry point for reward_shaping. Returns (bonus, result).
    bonus = bonus_scale * write_reward + test_use_scale * caught_and_fixed_reward.
    Requires an oracle at <oracle_dir>/<task>.py. If none, returns (0, empty).

    Command text is read from the terminal recording.cast (reliable) when trajectory_path
    is given, else parsed from `steps` (lossy for large turns)."""
    op = os.path.join(oracle_dir, task + ".py")
    if not os.path.exists(op):
        return 0.0, TestRewardResult()
    oracle_code = open(op).read()
    if mutants is None:
        mutants = _get_mutants(task, op, oracle_code, shipped_io, timeout)
    # build command text: prefer the cast (co-located with the trajectory)
    ctext = ""
    if trajectory_path:
        cast = os.path.join(os.path.dirname(str(trajectory_path)), "recording.cast")
        ctext = command_text_from_cast(cast)
    if not ctext:
        ctext = command_text_from_steps(steps)
    # prefer the explicit declared-test file (has expected outputs -> validity signal);
    # fall back to inferred input-only tests from heredocs/redirects.
    declared = extract_declared_tests(ctext)
    if declared:
        res = score_tests(oracle_code, declared, mutants, timeout, target_discriminating)
        tests_for_use = declared
    else:
        inputs = extract_model_test_inputs(ctext)
        tests_for_use = [{"input": i} for i in inputs]
        res = score_tests(oracle_code, tests_for_use, mutants, timeout, target_discriminating)
    # test-USE: did the model catch & fix a bug its tests exposed?
    if test_use_scale > 0.0:
        versions = extract_solution_versions(ctext)
        res.n_solution_versions = len(versions)
        tu, nfix = caught_and_fixed_score(oracle_code, versions, tests_for_use,
                                          timeout, test_use_target)
        res.test_use_reward = tu
        res.n_fixed = nfix
    bonus = bonus_scale * res.reward + test_use_scale * res.test_use_reward
    return bonus, res


# ----------------------------------------------------------------- self-test ---
if __name__ == "__main__":
    # toy problem: read n, print n*2
    oracle = "n=int(input())\nprint(n*2)\n"
    shipped = [("5\n", "10\n"), ("0\n", "0\n")]
    muts = make_mutants(oracle, shipped, timeout=5)
    print("mutants kept:", len(muts))
    for m in muts:
        print("  ---\n  " + m.replace("\n", "\n  "))
    # model wrote two test inputs by heredoc + redirect
    steps = [
        {"source": "agent", "message":
         "cat > /app/solution.py << 'EOF'\nn=int(input())\nprint(n*2)\nEOF\n"
         "cat > t1.txt << 'EOF'\n5\nEOF\n"
         "cat > t2.txt << 'EOF'\n7\nEOF\n"
         "python3 /app/solution.py < t1.txt\npython3 /app/solution.py < t2.txt\n"},
    ]
    inputs = extract_model_test_inputs(command_text_from_steps(steps))
    print("extracted test inputs (input-only):", inputs)
    res = score_discriminating_tests(oracle, inputs, muts, timeout=5, target_discriminating=3)
    print("input-only result:", res)
    assert res.n_test_inputs == 2 and res.n_ran_on_oracle == 2 and res.reward > 0, res

    # --- declared-test path (with expected outputs) ---
    import json as _json
    good = _json.dumps([{"input": "5", "expected": "10"}, {"input": "7", "expected": "14"},
                        {"input": "0", "expected": "0"}])
    bad = _json.dumps([{"input": "5", "expected": "999"}])  # wrong expected -> invalid
    steps_good = [{"source": "agent", "message": f"cat > /app/tests.json << 'EOF'\n{good}\nEOF\n"}]
    steps_bad = [{"source": "agent", "message": f"cat > /app/tests.json << 'EOF'\n{bad}\nEOF\n"}]
    dg = extract_declared_tests(command_text_from_steps(steps_good))
    print("declared tests parsed:", dg)
    assert len(dg) == 3 and dg[0]["expected"] == "10"
    rg = score_tests(oracle, dg, muts, timeout=5, target=3)
    rb = score_tests(oracle, extract_declared_tests(command_text_from_steps(steps_bad)), muts, timeout=5, target=3)
    print("declared GOOD:", rg)
    print("declared BAD :", rb)
    assert rg.n_valid == 3 and rg.reward > rb.reward, (rg, rb)
    assert rb.n_valid == 0, rb  # wrong expected earns no validity
    # negative controls
    assert score_tests(oracle, [], muts).reward == 0.0
    assert score_tests(oracle, [{"input": "5"}], []).reward == 0.0  # no mutants, no expected

    # --- test-USE (caught & fixed) ---
    wrong = "n=int(input())\nprint(n*3)\n"   # v1: buggy
    right = "n=int(input())\nprint(n*2)\n"   # v2: fixed
    tests_cf = [{"input": "5", "expected": "10"}, {"input": "7", "expected": "14"}]
    tu, nfix = caught_and_fixed_score(oracle, [wrong, right], tests_cf, timeout=5, target=2)
    print("caught&fixed: reward=%.3f n_fixed=%d"%(tu, nfix))
    assert nfix == 2 and tu == 1.0, (tu, nfix)
    # no fix if only one version, or if later version doesn't fix
    assert caught_and_fixed_score(oracle, [wrong], tests_cf, timeout=5)[1] == 0
    assert caught_and_fixed_score(oracle, [right, wrong], tests_cf, timeout=5)[1] == 0  # regressed, not fixed
    # extraction of solution versions
    ct = ("cat > /app/solution.py << 'EOF'\n%s\nEOF\n"
          "cat > /app/solution.py << 'EOF'\n%s\nEOF\n") % (wrong.strip(), right.strip())
    assert len(extract_solution_versions(ct)) == 2
    print("\nSELF-TEST PASSED")
