# Submitting To The ICCAD 2026 Contest

This document is now based on the **actual** `iccad2026contest/` framework
(README, `iccad2026_evaluate.py`, `optimizer_template.py`,
`training_example.py`, v9 PDF). Earlier versions of this guide were
informed guesses; this one is verified.

---

## TL;DR — the contract

The contest does **not** take a tensor file. It takes a **Python module**
with a class named (at minimum) `MyOptimizer` that subclasses
`FloorplanOptimizer` from `iccad2026_evaluate`. The framework calls
`optimizer.solve()` once per test case (100 cases) and times each call
for the `RuntimeFactor` term in the cost.

Required `solve()` signature:

```python
def solve(
    self,
    block_count:        int,
    area_targets:       torch.Tensor,    # [n_blocks]  (-1 padded)
    b2b_connectivity:   torch.Tensor,    # [edges, 3]  (block_i, block_j, weight)
    p2b_connectivity:   torch.Tensor,    # [edges, 3]  (terminal_i, block_j, weight)
    pins_pos:           torch.Tensor,    # [n_pins, 2] terminal (x, y)
    constraints:        torch.Tensor,    # [n_blocks, 5]
    target_positions:   Optional[torch.Tensor] = None,  # [n_blocks, 4] = (x, y, w, h)
) -> List[Tuple[float, float, float, float]]:    # one (x, y, w, h) per block
    ...
```

Return type: `List[(x, y, w, h)]`, length exactly `block_count`. Note
the order is `(x, y, w, h)` — *not* the `(w, h, x, y)` of `fp_sol`.

`my_optimizer.py` in this repo wraps our C++ binary as a
`FloorplanOptimizer` subclass and is the file you submit.

---

## Constraint encoding (verified from `iccad2026_evaluate.py`)

`constraints[i]` is a length-5 row:

| Column | Meaning | Values |
|---:|---|---|
| 0 | fixed flag        | 0 or 1 (any non-zero = fixed) |
| 1 | preplaced flag    | 0 or 1 |
| 2 | MIB group id      | 0 = none; ≥1 = group id |
| 3 | cluster (grouping) group id | 0 = none; ≥1 = group id |
| 4 | boundary bitmask  | see below |

**Boundary is a bitmask** (not a sequential enum):

| Bit | Meaning |
|---|---|
| 1   | left edge   |
| 2   | right edge  |
| 4   | top edge    |
| 8   | bottom edge |

Corners are **sums** of two single-edge bits:

| Code | Corner |
|---|---|
| 5  (1+4) | top-left |
| 9  (1+8) | bottom-left |
| 6  (2+4) | top-right |
| 10 (2+8) | bottom-right |

Our C++ uses a sequential enum (`E_LEFT=0 ... C_TR=7`).
`my_optimizer.py::_convert_boundary` translates between the two
encodings; this is unit-tested.

---

## `target_positions` semantics

The framework constructs this tensor before calling `solve()`:

- All rows default to `(-1, -1, -1, -1)`.
- For **fixed-shape** blocks, columns 2 and 3 (`w, h`) are set to the
  required dimensions; columns 0 and 1 (`x, y`) stay `-1` (free).
- For **preplaced** blocks, all four `(x, y, w, h)` are set.
- For **soft** blocks, all four stay `-1`.

`my_optimizer.py` reads these out and passes them through to our text
format as the input geometry for fixed/preplaced blocks. The packer
in `src/packer.cpp` already locks fixed/preplaced dimensions and
preplaced positions, so the hard constraint is enforced.

---

## Hard vs soft constraints (v9, post-Apr-19 update)

**Hard** — violation ⇒ `Cost = 10`:
- No block overlaps
- Soft-block area within 1% of `area_targets[i]`
- Fixed-shape blocks: `(w, h)` exactly match input
- Preplaced blocks: `(x, y, w, h)` all exactly match input
  (tolerance `1e-4`)

**Soft** — penalised via `exp(2 · V_relative)`:
- Boundary, grouping, MIB

Note this is the v9 change. Older specs treated fixed/preplaced as
soft; the evaluator was updated April 19 to treat them as hard.

---

## Deployment / packaging (C++ binary + Python wrapper)

Our solver is a compiled C++ binary; the contest framework is Python.
The two layers communicate through `subprocess`:

```
iccad2026_evaluate.py  ──importlib──▶  my_optimizer.py  ──subprocess──▶  ./floorplanner
   (the framework)                       (~300 lines)                     (compiled C++)
```

This pattern is explicitly endorsed by the v9 spec ("we invite the
application of any algorithmic paradigm"). The framework only inspects
the `solve()` return value — it doesn't care whether `solve()` is
implemented in pure Python, calls a C extension, or shells out to an
external binary.

### Two-file deployment

The contest folder needs exactly these two artefacts together:

| File | Where it goes | What it does |
|---|---|---|
| `my_optimizer.py` | `FloorSet/iccad2026contest/`         | Implements `MyOptimizer.solve()`; converts tensors → text, runs the C++ binary, parses the output |
| `floorplanner`    | next to `my_optimizer.py` *or* `$FLOORPLANNER_BIN` | The actual SA solver |

`my_optimizer.py` looks for the binary in three places, in order:
1. `$FLOORPLANNER_BIN` if set
2. `floorplanner` next to `my_optimizer.py`
3. `./floorplanner` in the current working directory

### Build options

| Make target | Use when | Output |
|---|---|---|
| `make`        | Local development on your own machine | `floorplanner` (dynamically linked) |
| `make static` | The contest server may have a different glibc / libstdc++ than your build box | `floorplanner` (statically linked, ~3-5 MB, no runtime deps) |
| `make submit` | Packaging the final submission | `submit/floorplanner_submission.zip` containing `my_optimizer.py` + `floorplanner` |

For final submission, **prefer `make static`** — it removes glibc
version mismatch as a failure mode. Verify with:

```bash
file floorplanner | grep "statically linked"
ldd floorplanner    # should print "not a dynamic executable"
```

### Submission packaging

```bash
make static          # build a portable binary
make submit          # create submit/floorplanner_submission.zip
```

The zip layout is:
```
floorplanner_submission/
├── my_optimizer.py
└── floorplanner            (statically linked, executable)
```

Drop the unzipped contents into `FloorSet/iccad2026contest/` on the
contest machine, then validate as in steps 4–6 below.

### What the contest server needs

- A POSIX-y Linux kernel (the `make static` binary works on
  ≥ Linux 3.2)
- Python 3 with the contest's `requirements.txt` already installed
- Permission to fork/exec subprocesses (needed for the
  `subprocess.run(["./floorplanner", ...])` call)

That's it — no compiler, no libraries, no Python C extensions to
install. If the server permits Python optimisers (the framework
is *itself* Python), it almost certainly permits subprocess calls
to a static binary in the same directory.

### What if the contest forbids native binaries?

This is an unlikely edge case (the framework is performance-oriented
and doesn't restrict implementation language), but if you hit it, your
options are:

1. **Compile on the server** — ship the source tree and a build script;
   the server runs `make` before evaluating. Requires `g++` on the
   server.
2. **Bind via `ctypes` or `cffi`** — turn the C++ solver into a shared
   library (`libfloorplanner.so`) and call it from `my_optimizer.py`
   without subprocess. Slightly faster (no fork overhead), still
   counts as "calling C++ from Python". Adds maybe 100 lines of
   C ABI shim.
3. **Reimplement in Python** — last resort. Would lose ~50× in speed
   and is not worth doing unless explicitly required.

Confirm the server's policy with the contest organisers before
going down route 1 or 2.

---

## How to run a submission

### 1. Build the solver
```bash
cd <floorplanner>
make           # produces ./floorplanner
make check     # toy smoke test
```

### 2. Set up the contest folder
```bash
git clone https://github.com/IntelLabs/FloorSet
cd FloorSet
python -m venv venv && source venv/bin/activate
pip install -r iccad2026contest/requirements.txt
# torch>=2.0, numpy>=1.24, shapely>=2.0, matplotlib, tqdm, requests
```

Download the validation set (100 samples) into `LiteTensorDataTest/`
per the contest README. The dataloaders auto-download from Hugging
Face on first use.

### 3. Drop in our optimizer
```bash
cp <floorplanner>/my_optimizer.py FloorSet/iccad2026contest/
cp <floorplanner>/floorplanner   FloorSet/iccad2026contest/
# OR set FLOORPLANNER_BIN=/abs/path/to/floorplanner
```

### 4. Validate format
```bash
cd FloorSet/iccad2026contest
python iccad2026_evaluate.py --validate my_optimizer.py
```

This calls `solve()` on a tiny dummy case (5 blocks). If it returns a
list of 5 tuples, you've passed the format check.

### 5. Smoke test on one case
```bash
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0
```

Look for `feasible=True`, `cost ≈ 1.0` ish. If `cost == 10.0`, the
solver is producing infeasible solutions — see "Common things that
go wrong" below.

### 6. Full run (100 cases)
```bash
python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions
# Generates my_optimizer_results.json + my_optimizer_solutions.json
```

The `--save-solutions` flag lets you re-score later without re-running:
```bash
python iccad2026_evaluate.py --score my_optimizer_solutions.json
```

### 7. Tuning per-case time

Use environment variables (no code change required):

```bash
FLOORPLANNER_TIME='10+0.5*n' \
FLOORPLANNER_THREADS=12 \
FLOORPLANNER_SEED=42 \
python iccad2026_evaluate.py --evaluate my_optimizer.py
```

The default `'5+0.5*n'` gives 16 s on n=21, ~65 s on n=120. Total
wall-clock for 100 cases ≈ 30-50 minutes on 8 threads. Remember:
the contest's `RuntimeFactor` is computed against the **median runtime
of all submissions**, capped at 0.7 (30% benefit). Slowness is
*uncapped*, so don't over-spend.

---

## Verification protocol (do this BEFORE submitting)

### Step A — format mapping is correct

Run on `--test-id 0` and check the printed metrics. If `feasible=True`
and `cost < 5.0` on a small case, format mapping is good.

If `cost == 10.0` (infeasible), one of these is wrong:
- `target_positions` not being honoured for fixed/preplaced
- Boundary bitmask mis-translated
- MIB / cluster group ids dropped or off-by-one

The framework will print which violation(s) tripped: overlap,
area, or dimension. Address the highest-impact one first.

### Step B — sanity-check against the Python verifier
```bash
# Set FLOORPLANNER_KEEP=1 then look at the .txt/.sol pair:
FLOORPLANNER_KEEP=1 python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0
# In our floorplanner repo:
python tools/verify_solution.py /tmp/my_optimizer_*/case_000.txt /tmp/my_optimizer_*/case_000.sol
```

The C++ contest_cost and `verify_solution.py` should agree to ~1e-3.
If they diverge, the C++ cost function and the Python reimpl have
drifted from v9 — fix that first because no tuning helps if the
solver optimises the wrong objective.

### Step C — full run, watch for outliers

In the per-case output, look for:
- Any `cost == 10.0` cases — these dominate the score
- Cases with very high runtime (RuntimeFactor > 2 ⇒ score multiplier ≥ 1.23)

The `--save-solutions` JSON makes it easy to grep for these.

---

## Common things that go wrong

1. **Cost = 10 on cases with fixed/preplaced blocks.**
   `target_positions` not being passed through correctly. Check that
   our text format gets the right `(w_in, h_in)` (and `(x_in, y_in)`
   for preplaced) and the C++ packer locks them.

2. **High V_relative (close to 1.0) → big `exp(2·V)` penalty.**
   Boundary bitmask probably mis-translated. Run with
   `FLOORPLANNER_KEEP=1` and inspect a `case_NNN.txt` against the
   tensor to confirm the `bedge` column matches the bitmask.

3. **Solver runs but produces overlap.**
   Check `--verbose` output; if SA's last-best `feas=0`, the SA
   couldn't find a feasible move under the time budget. Increase
   `FLOORPLANNER_TIME`, or examine the case's constraint mix —
   tightly-packed preplaced blocks can starve SA.

4. **Subprocess startup overhead inflates `RuntimeFactor` on small
   cases.** ~50 ms per call. Negligible at typical budgets but if
   you set `FLOORPLANNER_TIME=1.0` for n=21 it can matter. Don't
   undershoot the budget.

5. **`--validate` fails with `No optimizer class found`.**
   The class must subclass `FloorplanOptimizer` and be at module
   scope. `MyOptimizer` is the conventional name; the evaluator
   accepts any subclass.

6. **Conflicting MIB / cluster ids.** Group ids are 1-indexed in the
   official `constraints` tensor; 0 means "not in any group". Our
   text format uses 0-based ordinals (-1 for none), and we map
   between the two in `my_optimizer.py::_write_txt`.

---

## File map

| File | Role |
|---|---|
| `my_optimizer.py` | The actual contest submission. Wraps `./floorplanner` as a `FloorplanOptimizer` subclass. |
| `tools/verify_solution.py` | Local re-implementation of the v9 cost. Used as a cross-check, not for submission. |
| `tools/floorset_to_txt.py`, `tools/sol_to_fp_tensor.py`, `tools/run_official_testset.py`, `tools/iccad2026_adapter.py` | Earlier exploratory adapters that pre-date `my_optimizer.py`. They produce or consume tensor-format submissions. **Not used by the official contest framework.** Keep them around if useful for debugging or batch experiments. |

---

## Submission checklist

- [ ] `make` builds `floorplanner` cleanly.
- [ ] `make check` passes the toy benchmark (cost ≈ 1.0).
- [ ] `python iccad2026_evaluate.py --validate my_optimizer.py` ⇒ PASSED.
- [ ] `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0`
      reports `is_feasible=True` and `cost < 5`.
- [ ] First 10 cases all feasible with cost < 5.
- [ ] Full 100-case run, `total_score < 2.0`.
- [ ] `solutions.json` saved for re-scoring.
- [ ] `requirements.txt` matches what `my_optimizer.py` needs (just
      `torch` — already in the contest's requirements).
- [ ] Whatever submission packaging the contest specifies (zip,
      git tag, leaderboard upload, etc.) is followed.

If those tick off, you're ready to submit.
