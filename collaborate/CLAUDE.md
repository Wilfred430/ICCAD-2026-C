# CLAUDE.md

Internal handover notes for AI assistants (Claude Code, ChatGPT, etc.) and
new team members. Read this before making non-trivial changes to the
codebase.

If you're a human and just want to submit, read **`START_HERE.md`** instead.

---

## TL;DR

- **Contest**: ICCAD 2026 Contest C, FloorSet Challenge, FloorSet-Lite track.
- **Spec**: `Papers/FloorplanningContest_ICCAD_2026_v9.pdf` (2026-03-25, with
  important April-19 updates — see "v9 gotchas" below).
- **Approach A (Primary)**: PARSAC-style B*-tree + Fast-SA, multi-thread,
  multi-seed. C++ solver, Python wrapper. See `my_optimizer.py`.
- **Approach B (ML-augmented)**: Same C++ solver, but front-loaded with a
  Graph-Transformer warm-start (~250K params, ~50 ms/case). See
  `my_optimizer_ml.py` and `ml/`.
- **Approach C (Electrostatic)**: Pure-Python gradient-based global placer
  (ePlace/DREAMPlace paradigm), no C++ required. Total Score = 2.966 on
  100-case validation (100% feasible). See `electro_submission/`.
- **Status as of 2026-06-30**:
  - Phase 1 (C++ SA solver): Complete and mature. Alpha-test deadline (5/26) passed.
  - Phase 2 (ML warm-start): Implemented. Three trained weight checkpoints
    (`ml/weights/floorplan_v1.pt`, `v2.pt`, `v3.pt`). Drop-in via `my_optimizer_ml.py`.
  - Phase 3 (diffusion post-processing): Not started.

---

## What lives where

| Path | Role | Touch when |
|---|---|---|
| `include/`, `src/`           | C++ solver core | Improving SA, cost, packer, moves |
| `my_optimizer.py`            | **Primary submission file.** Python wrapper → `FloorplanOptimizer` subclass | Changing tensor↔text format mapping |
| `my_optimizer_ml.py`         | ML-augmented drop-in replacement for `my_optimizer.py`; adds Graph-Transformer warm-start | Swapping ML weights or tuning ML pipeline |
| `ml/`                        | Graph-Transformer model, data loader, trainer, predictor; trained weights in `ml/weights/` | Retraining or architecture changes |
| `electro_submission/`        | Alternative pure-Python electrostatic placer (Approach C) | Independent submission variant |
| `Makefile`                   | `make` / `make static` / `make check` / `make submit` | Build configuration |
| `benchmarks/toy.txt`         | 6-block synthetic exercising every constraint | Adding new constraint types |
| `benchmarks/toy.sol`         | Reference solution for toy benchmark | Regenerate if cost formula changes |
| `START_HERE.md`              | Step-by-step "how to submit" guide for humans | Steps actually change |
| `README.md`                  | Project overview, build commands, file map | New files added |
| `SUBMISSION.md`              | Integration protocol details, deployment, troubleshooting | Contest framework changes |
| `EVALUATION.md`              | Method evaluation, approach comparison, ML extension notes | Trying new approaches |
| `SA_TUNING_GUIDE.md`         | SA cost-curve debugging, logging bug explanation, tuning knobs | SA behavioural questions |
| `ALGORITHM_GUIDE.md`         | End-to-end pipeline diagram, per-file roles, parameter quick-ref | Architecture questions |
| `Papers/`                    | Contest spec + reference papers | Reading only |
| `log_thread/`                | Per-thread SA convergence CSVs from recent runs | Post-run analysis |
| `CLAUDE.md`                  | This file | Lessons learned, gotchas |

> **`tools/` does not exist** in this checkout. Ignore any docs that reference
> `tools/floorset_to_txt.py` or `tools/verify_solution.py` — those have been
> removed.

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

> **Note**: There is no `tools/verify_solution.py`. Use the contest evaluator
> (`iccad2026_evaluate.py`) directly for cost cross-checks, or re-run
> `make check` and compare printed `contest_cost` values.

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
- **`baseline` (HPWL/area) is the dataset's own ground-truth/optimal
  solution**, not a reference SA solver's output (confirmed both by
  `iccad2026_evaluate.py::_extract_baseline()`, which reads it straight
  from the label geometry, and by independently re-deriving
  `metrics[0]`/`metrics[6]`/`metrics[7]` from `LiteTensorDataTest` label
  geometry — exact match). Practical upshot: beating baseline
  (`HPWL_gap + Area_gap < 0`, i.e. `Q < 1`) is genuinely hard since
  baseline is already near-optimal; for realistic targets, budget for
  `Q` slightly `> 1` and put the marginal effort into `V_rel = 0` and
  low `RT` instead.
- `V_rel ∈ [0, 1]`, so `exp(2·V_rel) ∈ [1, e²≈7.39]`.
- `RT^0.3` capped at `0.7` lower bound (max 30% speed benefit).
  Slowness penalty is *uncapped*.
- **`RuntimeFactor = Your Runtime / Median Runtime of All Submissions`,
  computed independently per test case** (per-spec footnote: "using
  that individual test case's median runtime as the sole reference
  point") — this is a **cross-team, per-case** median, not something
  you can compute or target precisely offline. `iccad2026_evaluate.py`'s
  local `--evaluate` mode approximates it with a **self-median across
  your own 100 validation-case runtimes** (it has no access to other
  teams' data) — that's a local practice proxy only, not the real
  mechanism. Don't over-index on hitting an exact local RT number;
  the actionable takeaway is just "minimize wall-clock time on every
  case," since the real reference point is an unknowable, competitive
  moving target.

### 6a-2. `iccad2026_evaluate.py --evaluate`'s Total Score has real run-to-run noise from RT
Because the local `--evaluate` mode's RuntimeFactor is a **self-median of your own
100 cases' measured wall-clock time** (see 6a above), and `RT^0.3` is uncapped on
the slow side, **the SAME deterministic algorithm can report different Total
Score across identical runs** purely from OS scheduling/system-load jitter in the
timing measurement — not from any actual difference in the computed geometry.
Confirmed 2026-07-14 (`electro_optimized/`): three consecutive full-100 runs of
the byte-identical algorithm (coordinate diff between runs = 0.00000000) gave
Total Score 2.0158 / 2.0912 / 2.1029 — a ~5% spread from RT noise alone,
worse when multiple processes (e.g. two people/agents benchmarking on the same
machine at once) compete for CPU. **Practical upshot**: for "does this specific
algorithmic change help or hurt" questions, compare using a **neutral/fixed RT**
(RT=1.0, or whatever your harness's "Offline Neutral RT" mode is) — not the
raw Contest-Grading Total Score, which bakes in this timing noise. Reserve the
real-RT Total Score for a final, big-picture "is this ready to submit" check,
and expect it to jitter a few percent between runs even with zero code changes.
Large, clearly-explained deltas (driven by actual violation-count or geometry
changes) are still trustworthy; single-digit-percent deltas measured this way
are not, on their own, reliable evidence that a change helped.

### 6b. `ml/contest_cost.py`'s overlap/boundary tolerance was 10x too strict (fixed 2026-07-14)
`TOUCH_EPS` was `1e-7`; the real evaluator (`iccad2026_evaluate.py::check_overlap()`,
line ~223: `if overlap_x > 1e-6 and overlap_y > 1e-6`) uses `1e-6`. This matters
for **continuous/gradient-based placers** (electro's analytical optimization
lands blocks at values like `x=118.2616...`, not tidy round numbers) —
`legalize()`'s push/compaction arithmetic routinely leaves a residual gap in
the `1e-7`–`9e-7` range between two blocks that are meant to exactly touch.
The real evaluator tolerates this (`> 1e-6` required to flag); our stricter
`1e-7` threshold was flagging these as `overlap_violation=True` → `Cost=10`,
a **false infeasibility**. Found while building `ml/case_report_electro.py`
(a case_report.py-style Excel report for pop's electro pipeline): a case
independently confirmed `feasible=True, Cost=3.816` via the real evaluator
came back `feasible=False, Cost=10.000` through `contest_cost.py` — traced
to 4 block pairs with `overlap_area` in the `1e-7`–`9e-7` range on one axis.
Fixed by changing `TOUCH_EPS` to `1e-6` (`ml/contest_cost.py:26`), which also
fixes `_boundary_ok`'s edge-touching tolerance to match the real evaluator's
own `eps = 1e-6` (line ~511) for boundary checks. **Discrete/B\*-tree-contour
outputs are much less likely to have hit this** (the packer's arithmetic
tends to produce exact touching coordinates, not continuous-optimization
noise), so this probably didn't silently invalidate the generative B\*-tree
line's "100/100 feasible" numbers reported earlier this session — but if a
future case ever reports a surprising `Cost=10` with `overlap_violation=True`
and the geometry looks visually fine, check for a sub-`1e-6` residual gap
before assuming it's a real overlap bug in the placer itself.

### 7. Total score weights cases by `e^(n/12)`, NOT `e^n`
Verified 2026-07-01 against the actual spec PDF (Objective Function /
Total Score section): `Total Score = Σ Cost[i]·e^(n_i/12) / Σ e^(n_j/12)`.
n ranges from 21 to 120. A 120-block case is `e^(120/12)/e^(21/12) =
e^8.25 ≈ 3820` times heavier than a 21-block case — big, but nowhere
near the `e^99 ≈ 8·10^42` an earlier (wrong) reading of this file once
claimed. Mid-range cases (n≈60–90) still contribute meaningfully to
the total; don't write them off entirely. Big cases still dominate and
should get priority, just not to the point of total tunnel vision.

> **Caution**: `iccad2026_evaluate.py::compute_total_score()` in the
> local eval harness implements `math.exp(n - max_n)` — plain `e^n`,
> not `e^(n/12)`. This does NOT match the spec PDF. Treat the spec PDF
> as authoritative; the local script's Total Score number is a rough
> local proxy, not the real grading formula. (Discovered by cross-
> checking Fable-5-drafted analysis against both the local script and
> screenshots of the actual spec — the spec should always be pulled
> in preference to inferring intent from repo code.)

### 8. Contest framework expects a Python module, not a tensor file
`my_optimizer.py` (or `my_optimizer_ml.py`) is the actual submission
artefact. The framework imports it via `importlib`, finds a
`FloorplanOptimizer` subclass, and calls `solve()` once per test case
(timing each call for the `RuntimeFactor`).

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

---

## Code map (where to look for what)

| If you want to … | Look at |
|---|---|
| Change HPWL / area / V formula        | `src/cost.cpp` (verify against contest evaluator) |
| Add or modify an SA move              | `include/moves.hpp` + `src/moves.cpp` |
| Change SA temperature schedule        | `src/sa.cpp::run()` |
| Change initial-floorplan construction | `src/parallel.cpp::make_initial()` |
| Change packing geometry               | `src/packer.cpp::pack_btree()` |
| Change B*-tree primitives             | `src/btree.cpp` |
| Add new constraint type               | `include/types.hpp` → `src/cost.cpp` → `src/parser.cpp` → possibly `src/moves.cpp` → `my_optimizer.py::_write_txt` |
| Change tensor↔text mapping            | `my_optimizer.py::_write_txt` |
| Change boundary encoding              | `my_optimizer.py::_BOUNDARY_BITMASK_TO_ENUM` (NOT the C++ enum) |
| Tune SA hyperparameters               | `include/sa.hpp::SAConfig` defaults, or via CLI flags in `src/main.cpp` |
| Tune SA cost weights                  | `include/cost.hpp::SAWeights` — **most commonly changed file** |
| Change per-case time budget           | `FLOORPLANNER_TIME` env var (no rebuild needed) |
| Change parallel strategy              | `src/parallel.cpp::worker()` |
| Understand SA tuning knobs            | `SA_TUNING_GUIDE.md` |
| Understand full pipeline + params     | `ALGORITHM_GUIDE.md` |
| Swap / retrain ML warm-start model    | `ml/model.py`, `ml/train.py`, `ml/weights/` |
| Use electrostatic placer              | `electro_submission/electro_optimizer.py` |
| Run/retrain the generative B*-tree model | `python -m ml.run_pipeline` (see "Generative B*-tree model" below) |

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
   exist *only* for constraint-fixing (FixBoundary and FixGrouping).
   Adding more always-accept moves breaks SA's mixing guarantees.

6. **Threads are independent.** No shared state between worker chains
   except the read-only `FloorplanInstance`. If you need cross-thread
   info (population-style SA), do it as explicit message passing; don't
   sneak in a global.

7. **Per-thread RNG is seeded from `seed + thread_id`.** Reproducibility
   matters when debugging. If you change RNG plumbing, expose seed
   control end-to-end.

8. **`my_optimizer.py` is stateless across `solve()` calls** (other
   than a workdir for intermediate files). The framework may run
   cases in any order; don't depend on previous-call state.

---

## SA diagnostic notes (see also `SA_TUNING_GUIDE.md`)

The cost curves in `log_thread/*.csv` record `CurrentCost` (the SA's
actual accepted cost) and `BestCost`. A known earlier bug logged the
*candidate* cost `nc` instead of the accepted cost `cost`, making
curves look wildly noisy — this has been fixed in `src/sa.cpp`.

When reading convergence plots, **look at `BestCost` (column 4), not
`CurrentCost`** — `BestCost` is the stair-step envelope that actually
matters. Floorplan SA will always have higher short-term variance than
partition SA because a single tree-restructuring move can displace 30+
blocks simultaneously.

---

## The ML warm-start (`my_optimizer_ml.py` + `ml/`)

`my_optimizer_ml.py` is a drop-in replacement for `my_optimizer.py`.
It adds a Graph-Transformer inference step (~50 ms/case) that predicts
`(cx, cy, w, h)` for every block, then appends a `WARM_POSITIONS`
section to the C++ input file. The C++ parser currently silently ignores
unknown sections, so the warm-start only takes effect if the C++ side
has been patched to read it (see `ML_FLOORPLAN.md §4`).

**Model architecture** (`ml/model.py`):
- Block features [N, 16] + terminal positions [T, 2] fed as a unified
  sequence through L TransformerEncoder layers (full self-attention).
- Output heads: `pos_head` → (cx, cy), `dim_head` → (w, h).
- ~250K parameters; trains in 30–60 min on GPU over 10K cases.

**Trained checkpoints** (in `ml/weights/`):
- `floorplan_v1.pt`, `floorplan_v2.pt`, `floorplan_v3.pt` — different
  training epochs / hyperparameter settings.

**Configuration** (env vars):
- `FLOORPLANNER_ML_WEIGHTS` — path to `.pt` file (default: `ml/weights/floorplan_v1.pt`).
- `FLOORPLANNER_ML_DEVICE` — `cpu` or `cuda` (default `cpu`).
- `FLOORPLANNER_ML_VERBOSE` — `1` for predictor diagnostics.

If the weights file is missing, `my_optimizer_ml.py` silently falls back
to the baseline C++-only pipeline.

---

## Generative B*-tree model (`ml/model_tree.py`, `ml/train_tree.py`, 2026-07-01)

A second, independent ML approach that predicts *topology* instead of
*coordinates* — this is the fix for `model.py`'s mode-collapse failure mode
(averaging two valid but different B*-tree layouts produces overlapping,
invalid geometry; cross-entropy training over discrete tree-construction
choices doesn't have that failure mode, because the model must pick one
option rather than blend them).

**One command to train + sample + score + observe:**

```bash
cd collaborate
python -m ml.run_pipeline                       # defaults: quick-trains if
                                                  # ml/weights/tree_v1.pt is
                                                  # missing, then demos case 0
python -m ml.run_pipeline --case 5 --samples 16  # demo a different case
python -m ml.run_pipeline --retrain --train-cases 20000 --epochs 5  # real training run
```

This prints a ranked table of sampled topologies (bbox area / HPWL vs. the
dataset's own baseline `metrics`) and writes the best one as a `.sol`-style
text file. **Soft-block dimensions in this demo are a placeholder square
(`w=h=sqrt(area)`)** — the tree model only decides topology; pairing it with
`model.py`'s `dim_head` (or a dedicated dimension head) is the natural next
step, not yet wired in.

**Data pipeline** (`ml/data.py`):

- `tree_sol` (the 1M-set's near-optimal B*-tree edge list, `[N-1, 3]` =
  `(parent_id, child_id, direction)`) is loaded for TRAIN-format cases and
  converted by `tree_sol_to_sequence()` into `gen_order` / `parent_step` /
  `direction` teacher-forcing targets. TEST-format cases (the 100-case
  validation set) have no `tree_sol`; `has_tree=False` for those.
- `direction` semantics (confirmed against `src/packer.cpp::Packer::pack()`,
  our own C++ packer, NOT reverse-engineered): `0` = left child (touches
  parent's right edge), `1` = right child (touches parent's top edge); both
  use the running horizontal contour for y. Decoding `tree_sol` this way and
  comparing to `fp_sol` reproduces 20–77% of blocks exactly per case — the
  rest is Intel's own post-pack compaction, which doesn't matter for
  training since we only need the topology labels, not the geometry.

**Model architecture** (`ml/model_tree.py`, `TreeGenerator`): shares the
same block+terminal context encoder design as `model.py`, then a causal
Transformer decoder with THREE pointer-network heads per generation step:
which block goes next (over remaining blocks), which earlier step is its
parent (over already-placed steps), and which side it attaches on. All
three are pointer networks (not fixed-size classifiers), so the model works
for any block count. `generate()` runs this fully autoregressively with no
ground truth — that's what makes it usable on genuinely unseen cases.

**Packing sampled topologies** (`ml/pack_tree.py`): a pure-Python port of
`packer.cpp`'s contour-DFS placement + `compact_left_down`, used for fast
scoring/prototyping. It does NOT port `bbox_balance_pass` / `holes_fill_pass`
/ `grouping_repair_pass` / `boundary_repair_pass` — the real submission path
still goes through the actual C++ `floorplanner` binary, which has those.
`build_lc_rc()` deterministically repairs slot conflicts (two blocks
predicted for the same parent+direction), which an undertrained model
produces regularly — this keeps the pipeline crash-free instead of failing
on a bad sample.

**Status**: architecture + data pipeline validated end-to-end (smoke-tested
on real 1M-set data, loss/accuracy improve with training, `generate()`
produces valid full-coverage trees on unseen validation cases). Not yet
trained to convergence — see `WINNING_STRATEGY.md` §2 for the full plan
(reward fine-tuning against real contest Cost comes after supervised
pretraining is solid).

---

## Electrostatic alternative (`electro_submission/`)

A fully independent approach using continuous gradient-based global
placement (ePlace / DREAMPlace paradigm), pure Python + PyTorch, no
C++ binary required.

**Entry point**: `electro_submission/electro_optimizer.py`

**Pipeline**: analytical_place (PyTorch/Adam) → legalize (constraint-graph
compaction) → soft_repair (boundary/grouping passes)

**Validated score**: Total Score = **2.966**, 100/100 feasible, all
coordinates ≥ 0 on the 100-case validation set.

**Configuration** (env vars):
- `ELECTRO_SEEDS=N` — multi-start seeds (default 1).
- `ELECTRO_ITERS=K` — placement iterations (default 600).
- `ELECTRO_CLAMP=0 ELECTRO_NONNEG=0` — allow negative coords (Total ≈ 2.334 but lower cost).

All five `.py` files must be in the same directory. No rebuild needed.

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
  actually needs it.
- The Python wrapper (`my_optimizer.py`) avoids any non-stdlib import
  besides `torch`. No external Python deps for the integration layer.
- `src/*_backup.cpp` files (`btree_backup.cpp`, `moves_backup.cpp`,
  `packer_backup.cpp`) are reference snapshots — do not compile or
  include them in the build.

---

## Where the wins are (priority for new work)

As of 2026-06-30, Phase 1 is done. In rough order of effort-vs-score-impact:

1. **Benchmark the three ML weights on the validation set.** Run
   `my_optimizer_ml.py` with v1/v2/v3 and compare Total Score to the
   baseline (C++ only). Determine whether the warm-start is helping.

2. **Compare Approach A vs Approach C.** The electrostatic placer scores
   2.966 Total on validation. Measure Approach A's Total Score under the
   same conditions to decide which to submit.

3. **Per-instance time budget tuning.** Currently `8+1.0*n` (see
   `ALGORITHM_GUIDE.md §3.2`). Sweep over a few formulas to find one
   that maximises `Total Score` on the validation set.

4. **Identify worst-cost cases.** Look at `my_optimizer_results.json`
   after a full run. Cases with `cost > 2.0` dominate the weighted
   total due to the `e^(n/12)` weighting (see gotcha #7). Profile
   individually with `--test-id N --verbose`.

5. **Quasi-Newton geometry refinement** after SA converges (Ji 2021).
   Topology is fixed by then; only `(w, h, x, y)` continuous → 3–5%
   improvement at near-zero runtime cost.

6. **Hyperparameter sweep** on validation set: move probabilities,
   `c_fastsa`, `n_iters_per_block`, T-schedule constants. Use
   `n ∈ {30, 60, 90, 120}` slices. See `SA_TUNING_GUIDE.md` for
   guidance on which knobs to turn first.

7. **Diffusion post-processing.** See `EVALUATION.md` Phase 3.
   Lowest priority — out of scope unless score gap remains large.

---

## Things to NOT do

- **Don't switch the topology representation** away from B*-tree
  without a very strong reason. Sequence-pair / TCG / CBL all require
  redesigning the move set, the packer, the constraint handling, and
  the I/O layer.

- **Don't fold ML into the C++ main loop.** Keep the GNN proposer
  as a separate Python service (pipe or precomputed batches), or as a
  TorchScript model loaded via libtorch. Don't entangle Python
  build deps into the contest binary.

- **Don't tune to the 100 public validation cases.** They're public;
  the test set is hidden. Train ML on the 1M training set, validate
  on the 100, and treat the 100 only as a smoke-screen.

- **Don't introduce float32** just because ML models use it.
  Centroid HPWL has cancellation; double is right.

- **Don't relax the 1% area tolerance** "to make SA converge faster".
  That's a hard constraint in v9; relaxing it produces submissions
  that pass local checks and fail on the official evaluator with `Cost = 10`.

- **Don't change the boundary enum in C++.** The conversion lives in
  `my_optimizer.py`. Touching the C++ enum requires updating
  `cost.cpp::boundary_ok()`, `parser.cpp`, every `.txt` benchmark,
  and the Python conversion table — 4 places to keep in sync.

- **Don't compile `src/*_backup.cpp` files.** They are reference
  snapshots only; the Makefile already excludes them.

---

## Team context

- 隊員 3 人，指導老師 陳老師 / 趙家佐。
- 4/28 會議決議：採用 PARSAC + B*-tree + Fast-SA，分階段加 ML。
- 5/26 alpha test 截止日已通過（2026-06-30 現在是後期優化階段）。
- 現狀：C++ SA solver 穩定，ML warm-start (v1/v2/v3) 已訓練，
  電靜力法替代路線已驗證（Total 2.966 on validation）。
- 下一步決策：確定最終提交用哪個 Approach（A / A+ML / C）。

If a request conflicts with the 4/28 plan or the v9 spec, flag it
explicitly rather than silently doing something different.

---

## When stuck

- Re-read v9 spec eq. (2)–(4) and PARSAC §3.2.
- Run `make check` — if that breaks, you've broken cost or feasibility.
- If SA cost curves look wrong: read `SA_TUNING_GUIDE.md` — there was
  a logging bug (candidate cost logged instead of current cost); verify
  `src/sa.cpp` logs `cost`, not `nc`.
- Look at `[SA] iter=...` traces with `--verbose`. If `best_sa_cost`
  doesn't trend downward and `T` is decaying normally, something in
  the move / cost / pack pipeline is broken.
- For Python-side bugs, set `FLOORPLANNER_KEEP=1` and inspect
  `/tmp/my_optimizer_*/case_NNN.txt` to see what the wrapper feeds
  to the C++ binary.
- For parameter quick-reference, see `ALGORITHM_GUIDE.md §3`.
- For "what should we do next" questions, see `EVALUATION.md`.
