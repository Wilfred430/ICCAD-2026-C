# CLAUDE.md

Internal handover notes for AI assistants (Claude Code, ChatGPT, etc.) and
new team members. Read this before making non-trivial changes to the
codebase.

If you're a human and just want to submit, read **`START_HERE.md`** instead.

---

## TL;DR

- **Contest**: ICCAD 2026 Contest C, FloorSet Challenge, FloorSet-Lite track.
- **Spec**: `FloorplanningContest_ICCAD_2026_v9.pdf` (2026-03-25, with
  important April-19 updates — see "v9 gotchas" below).
- **Approach**: PARSAC (arXiv 2405.05495) + B*-tree + Fast-SA, multi-thread,
  multi-seed.
- **Status**: Phase 1 baseline complete. C++ solver compiles, smoke-tests
  pass on the toy benchmark, integrates cleanly with the official Python
  framework via `my_optimizer.py`. Phase 2 (GNN move proposer) and Phase 3
  (diffusion post-processing) not started — see `EVALUATION.md`.
- **Hard deadline**: alpha-test code freeze 2026-05-26.

---

## What lives where

| Path | Role | Touch when |
|---|---|---|
| `include/`, `src/`     | C++ solver core (~2100 lines) | Improving SA, cost, packer, moves |
| `my_optimizer.py`      | **The contest submission file.** Wraps the C++ binary as a `FloorplanOptimizer` subclass | Changing tensor↔text format mapping |
| `Makefile`             | `make` / `make static` / `make check` / `make submit` | Build configuration |
| `tools/floorset_to_txt.py` | Standalone helper: pkl → our text format | Bulk debug runs outside the contest framework |
| `tools/verify_solution.py` | Pure-Python v9 cost reimpl. — sanity cross-check | Whenever `cost.cpp` changes |
| `benchmarks/toy.txt`   | 6-block synthetic exercising every constraint | Adding new constraint types |
| `START_HERE.md`        | Step-by-step "how to submit" guide for humans | Steps actually change |
| `README.md`            | Project overview, build commands, file map | New files added |
| `SUBMISSION.md`        | Integration protocol details, deployment, troubleshooting | Contest framework changes |
| `EVALUATION.md`        | Method evaluation, ML extension recommendations | Trying new approaches |
| `CLAUDE.md`            | This file | Lessons learned, gotchas |

---

## Build & smoke-test

```bash
make             # release build, dynamic-linked
make static      # release build, statically linked (use this for submission)
make debug       # -O0 -g -DDEBUG
make check       # runs ./floorplanner on benchmarks/toy.txt
make submit      # packages my_optimizer.py + binary into submit/...zip
make clean
```

Smoke test must produce `feasible=1` and `contest_cost ≈ 1.00` on
`benchmarks/toy.txt`. If it doesn't, *something is broken* — that
benchmark is constructed so a correct solver cannot fail it.

Always cross-check with the Python verifier:

```bash
python3 tools/verify_solution.py benchmarks/toy.txt benchmarks/toy.sol
```

The two `contest_cost` values should agree to ≤ 1e-3. If they don't, the
C++ cost function and the Python reimpl have diverged from v9 — **fix
that before doing anything else.** Tuning a solver that optimises the
wrong objective is wasted effort.

---

## v9 gotchas (READ BEFORE TOUCHING `cost.cpp` or `my_optimizer.py`)

These are the spec details that are easy to get wrong and silently
produce 10× worse scores. All of them are verified against the actual
`iccad2026_evaluate.py` source code (not just the spec PDF).

### 1. HPWL is centroid-to-centroid Manhattan
Both `b2b` and `p2b` nets use weighted `|cx_i − cx_j| + |cy_i − cy_j|`
where `cx_i = x_i + w_i/2`. The bbox half-perimeter formula was an
*older* draft. See `cost.cpp::compute_hpwl_int()` and
`compute_hpwl_ext()`.

### 2. Fixed-shape and preplaced are HARD constraints
Changed April 19, 2026. Any deviation from the input dimensions
(or input location, for preplaced) ⇒ `Cost = 10` (M penalty). They
live in the **packer** (immutable dims, anchored placement), not in
the soft `V` term. The `V` term in `cost.cpp` only carries grouping,
MIB, and boundary.

### 3. Soft-block area tolerance is 1%
`|w·h − a| / a ≤ 0.01`. Older drafts said 5%. See
`check_hard_constraints()` in `cost.cpp`.

### 4. Boundary code in `constraints[:, 4]` is a BITMASK, not a sequential enum
Verified from `iccad2026_evaluate.py` boundary check:
- `1` = left, `2` = right, `4` = top, `8` = bottom
- corners are sums: `5` = TL (1+4), `9` = BL (1+8), `6` = TR (2+4), `10` = BR (2+8)

Our C++ uses a sequential enum (`E_LEFT=0 ... C_TR=7`). The translation
lives in `my_optimizer.py::_BOUNDARY_BITMASK_TO_ENUM` and is unit-tested
(all 9 codes round-trip correctly). **Do not change the C++ enum;**
update the Python conversion table if anything ever changes.

### 5. `solve()` returns `List[(x, y, w, h)]`
**NOT** the `(w, h, x, y)` order of `fp_sol`. Our `.sol` writes
`id x y w h`, so it round-trips correctly. Easy to flip if you write
a new tool or adapter.

### 6. The cost formula
```
Cost = (1 + 0.5·(HPWL_gap + Area_gap)) · exp(2·V_rel) · max(0.7, RT^0.3)
       if feasible, else 10
```
- `HPWL_gap`, `Area_gap` are **signed** relative gaps:
  `(actual − baseline) / baseline`. Negative gaps are *good*.
- `V_rel ∈ [0, 1]`, so `exp(2·V_rel) ∈ [1, e²≈7.39]`.
- `RT^0.3` capped at `0.7` lower bound (max 30% speed benefit).
  Slowness penalty is *uncapped*.

### 7. Total score weights cases by `e^n`
n ranges from 21 to 120. A 120-block case is `e^99 ≈ 8·10^42` times
heavier than a 21-block case in the score sum. The big cases are
*everything*. Plan compute budget accordingly.

### 8. Contest framework expects a Python module, not a tensor file
`my_optimizer.py` is the actual submission artefact. The framework
imports it via `importlib`, finds a `FloorplanOptimizer` subclass,
and calls `solve()` once per test case (timing each call for the
`RuntimeFactor`).

### 9. `target_positions` is the framework's way to enforce hard constraints
The framework passes a 4-column tensor:
- All rows default to `(-1, -1, -1, -1)` (free).
- For **fixed-shape**: cols 2,3 (`w, h`) set to required dims; cols 0,1 stay `-1`.
- For **preplaced**: all four `(x, y, w, h)` set.
- For **soft**: all four stay `-1`.

`my_optimizer.py::_write_txt` reads this and passes the locked geometry
through to our C++ packer. The packer must respect it or hard
constraints will break.

### 10. Reverse-engineering the FloorSet generator is disqualifying
v9 footnote 6. Don't even joke about it.

If `cost.cpp` changes, `tools/verify_solution.py` MUST change too.
The two implementing the same formula is our only guard against
silently optimising the wrong thing.

---

## Code map (where to look for what)

| If you want to … | Look at |
|---|---|
| Change HPWL / area / V formula      | `src/cost.cpp` AND `tools/verify_solution.py` (both!) |
| Add or modify an SA move            | `include/moves.hpp` + `src/moves.cpp` |
| Change SA temperature schedule      | `src/sa.cpp::run_sa()` |
| Change initial-floorplan construction | `src/parallel.cpp::make_initial()` |
| Change packing geometry             | `src/packer.cpp::pack_btree()` |
| Add new constraint type             | `include/types.hpp` (Block fields) → `src/cost.cpp` (V term) → `src/parser.cpp` (I/O) → possibly new move in `src/moves.cpp` → `my_optimizer.py::_write_txt` |
| Change tensor↔text mapping          | `my_optimizer.py::_write_txt` |
| Change boundary encoding            | `my_optimizer.py::_BOUNDARY_BITMASK_TO_ENUM` (NOT the C++ enum) |
| Tune SA hyperparameters             | `include/sa.hpp::SAConfig` defaults, or via CLI flags in `src/main.cpp` |
| Change per-case time budget         | `FLOORPLANNER_TIME` env var (no rebuild needed) |
| Change parallel strategy            | `src/parallel.cpp::worker()` |

`include/types.hpp` is the canonical schema. If you add a Block field
there, `parser.cpp`, `cost.cpp`, `my_optimizer.py::_write_txt` and any
move that mutates Blocks all need to be audited for the new field.

---

## Architectural invariants (DO NOT VIOLATE)

These keep the SA correct. Breaking any of them silently produces wrong
answers that still look plausible — the worst kind of bug.

1. **Packer is deterministic given the B*-tree topology + every block's
   `(w, h)`.** No RNG, no SA state, no move history inside the packer.
   If broken, repeated cost evaluations of the same solution disagree
   and SA goes off the rails.

2. **Every move stores enough state to revert exactly.** We use a full
   topology snapshot (`saved_w_vec`, `saved_h_vec`) — wasteful but
   bulletproof. Don't try clever-incremental this without a
   regression test that calls `apply` then `revert` 10⁵ times and
   checks the B*-tree is bit-identical at the end.

3. **Hard constraints are checked in the packer or in
   `check_hard_constraints()`, NEVER in the soft `V` term.** If you
   find yourself adding a fixed-shape or preplaced contribution to
   `V`, stop — that's a v9 bug.

4. **MIB synchronisation must apply to all blocks in the group
   atomically.** Partial update + pack reflects an illegal state.
   `op_mib_sync` in `moves.cpp` does the whole group, then packs,
   then checks.

5. **`always_accept = true` moves bypass the Metropolis criterion** and
   exist *only* for constraint-fixing (the FixBoundary move). Adding
   more always-accept moves breaks SA's mixing guarantees.

6. **Threads are independent.** No shared state between worker chains
   except the read-only `FloorplanInstance`. If you need cross-thread
   info (population-style SA, e.g.), do it as explicit message
   passing; don't sneak in a global.

7. **Per-thread RNG is seeded from `seed + thread_id`.** Reproducibility
   matters when debugging. If you change RNG plumbing, expose seed
   control end-to-end.

8. **`my_optimizer.py` is stateless across `solve()` calls** (other
   than a workdir for intermediate files). The framework may run
   cases in any order; don't depend on previous-call state.

---

## Style & conventions

- C++17, no exceptions in the hot path. Errors during parsing throw;
  the SA loop assumes valid input.
- All geometry is `Real = double`. No plan to switch to float; n ≤ 120
  so memory is irrelevant and double avoids accumulated cancellation
  in the centroid HPWL computation.
- File comments at the top of each `.cpp`/`.hpp` explain that file's
  role in 1–2 lines. Keep this convention if you add files.
- Logging uses `std::fprintf(stderr, "[tag] ...")` to keep verbose
  mode thread-interleaved but readable. Don't switch to a logging
  library.
- We *deliberately* don't use templates or virtual dispatch in the hot
  path. The code is concrete and easy to step through. Keep it that way.
- The `Makefile` is plain GNU make. Don't pull in CMake unless someone
  actually needs it; it just adds onboarding friction for the team.
- The Python wrapper (`my_optimizer.py`) avoids any non-stdlib import
  besides `torch` (which the contest's requirements.txt already
  installs). No external Python deps for the integration layer.

---

## Where the wins are (priority for new work)

In rough order of effort-vs-score-impact:

1. **Run on validation set, identify worst-cost cases.** Look at
   `my_optimizer_results.json` after a full run. The cases with
   `cost > 2.0` are where most of the weighted score is bleeding.
   Profile a few of them individually with `--test-id N --verbose`.

2. **Per-instance time budget tuning.** Currently `'5+0.5*n'`. Sweep
   over a few formulas to find one that maximizes `Total Score` on
   the validation set. Likely sweet spot somewhere in `'5+0.3*n'` to
   `'10+0.8*n'`.

3. **Quasi-Newton geometry refinement** after SA converges (Ji 2021).
   Topology is fixed by then; only `(w, h, x, y)` continuous → 3-5%
   improvement at near-zero runtime cost. ~150 lines of new code.

4. **GNN move proposer** trained on the supplied 1M ground-truth
   `(constraint_graph → tree_sol)` pairs. Replace uniform random
   move sampling in `moves.cpp` with the GNN's distribution; SA
   acceptance criterion stays the same. See `EVALUATION.md` Phase 2
   for details.

5. **Hyperparameter sweep on validation set.** P_AR, P_MIB, P_FIX,
   `iters_per_block`, T-schedule constants. Use `n ∈ {30, 60, 90, 120}`
   slices.

6. **Diffusion post-processing.** See `EVALUATION.md` Phase 3.
   Out of scope for alpha-test.

---

## Things to NOT do

- **Don't switch the topology representation** away from B*-tree
  without a very strong reason. Sequence-pair / TCG / CBL all require
  redesigning the move set, the packer, the constraint handling, and
  the I/O layer. PARSAC's anchored-blocks trick is B*-tree-specific.
  Switching costs a week of work for no expected score gain at our
  `n` range.

- **Don't fold ML into the C++ main loop.** Keep the GNN proposer
  as a separate Python service that sends move suggestions over a
  pipe (or precomputed batches written to disk), or as a
  TorchScript model loaded via libtorch. Don't entangle Python
  build deps into the contest binary.

- **Don't tune to the 100 public validation cases.** They're public;
  the test set is hidden. Train ML on the 1M training set, validate
  on the 100, and treat the 100 only as a smoke-screen.

- **Don't introduce float32** just because diffusion models use it.
  Centroid HPWL has cancellation; double is right. Memory cost is
  irrelevant at n ≤ 120.

- **Don't relax the 1% area tolerance** "to make SA converge faster".
  That's a hard constraint in v9; relaxing it produces submissions
  that pass our local checks and fail on the official evaluator
  with `Cost = 10`.

- **Don't change the boundary enum in C++.** The conversion lives in
  `my_optimizer.py`. Touching the C++ enum requires updating
  `cost.cpp::boundary_ok()`, `parser.cpp`, every `.txt` benchmark,
  and the Python conversion table — 4 places to keep in sync.

- **Don't remove the `verify_solution.py` cross-check.** It's our
  only safeguard against `cost.cpp` and the v9 spec drifting apart.

---

## Team context

- 隊員 3 人，指導老師 陳老師 / 趙家佐。
- 4/28 會議決議：採用 PARSAC + B*-tree + Fast-SA，分階段加 ML。
- 5/5 整體架構 + alpha test 可行性 ← we are here / just past.
- 5/12 + 5/19 演算法優化。
- **5/26 alpha test code 可交版本（hard deadline）.**
- 中後期可能加上 diffusion 或 GNN 強化（contest 主辦方有暗示）。

If a request conflicts with the 4/28 plan or the v9 spec, flag it
explicitly rather than silently doing something different. The team is
working on a tight schedule and surprises waste their time.

---

## When stuck

- Re-read v9 spec eq. (2)–(4) and PARSAC §3.2.
- Run `make check` — if that breaks, you've broken cost or feasibility.
- Compare C++ output against `tools/verify_solution.py` — divergence
  means a formula bug.
- Look at `[SA] iter=...` traces with `--verbose`. If `cost` doesn't
  trend downward and `T` is decaying normally, something in the
  move / cost / pack pipeline is broken.
- For Python-side bugs (the `my_optimizer.py` integration), set
  `FLOORPLANNER_KEEP=1` and inspect `/tmp/my_optimizer_*/case_NNN.txt`
  to see what the wrapper is feeding the C++ binary.

For things `START_HERE.md` doesn't cover (rare/advanced situations),
check `SUBMISSION.md`. For "what should we do next" questions, check
`EVALUATION.md`.
