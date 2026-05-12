# Evaluation & Recommendation

This document answers the second and third parts of the prompt:
**評估這套方法的優缺點** and **給出你覺得的最佳方法**.

---

## Part 1 — Evaluation of the PARSAC + B*-tree + Fast-SA approach

### Why we picked this stack

Reading the six required references against the v9 spec, the design space
collapses to roughly four serious candidates:

| Family | Representatives | Verdict for FloorSet-Lite |
|---|---|---|
| Slicing trees / Polish expression | Wong–Liu 1986 [37] | Cannot represent the non-slicing layouts FloorSet-Lite samples are constructed from; rules itself out. |
| Sequence-pair / TCG / O-tree | Murata–Kuh 1998 [29], Lin–Chang 2001/02 [20,21], Pang 2000 [31] | Solid non-slicing representation, but pack cost is O(n²) (or O(n log log n) with LCS), and adapting the rich constraint set (preplaced anchoring, MIB sync, boundary fix-up) is markedly more code than the B*-tree variant. |
| CBL | Hong 2000 [10], Amini KDD'22 [2] | Very compact, GNN-friendly (Amini), but lacks the per-block "anchored" trick PARSAC uses for preplaced blocks; we'd have to reinvent it. |
| **B\*-tree + (Fast-)SA** | Chen–Chang TCAD'06 [6], **PARSAC** [28] | Linear-time pack via contour, well-understood SA schedule, and PARSAC §3.2.1–3.2.3 already gives boundary, anchored-blocks, and grouping recipes verbatim. **This is the cheapest path to an alpha-test-passing baseline.** |

Two recent papers we also studied don't change the conclusion:

- **Quasi-Newton floorplanner** (Ji 2021 [13]) — analytical, treats blocks as
  smooth potentials. Excellent for outline-fitting on large flat instances,
  but adapting it to MIB / grouping / boundary requires non-trivial
  reformulation, and PARSAC reports it loses on the FloorSet workloads anyway.
- **Mixed-Variable optimisation** (Sun 2024 [34]) — interesting because it
  treats topology and continuous shape jointly, but is a research-stage
  method without published code targeting our constraint set.

So: PARSAC is the *least-risky* baseline. It is also the one our 4/28 plan
already commits to.

### Strengths of our implementation

1. **All seven constraint classes handled in one cost function.** v9 promoted
   fixed and preplaced from soft to hard, so we route those through the
   packer (anchored B*-tree for preplaced; immutable w,h for fixed) and only
   keep grouping / MIB / boundary in the soft `Violationsrelative` term, which
   matches eq. (4) of v9 exactly. The Python verifier double-checks that the
   C++ cost agrees with a pure-Python recomputation.

2. **The HPWL formula is the v9 one, not the older bbox version.** v9
   silently changed inter-module HPWL from "half-perimeter of the
   block-and-its-neighbours bounding box" to "weighted Manhattan between
   centroids" (eq. 3 of v9). `cost.cpp::compute_hpwl_int()` implements the
   centroid form. This is the single most common bug we expect other teams
   to ship — easy to score 10× worse without noticing.

3. **Linear-time packer.** Each pack is `O(n)` thanks to the sorted-vector
   contour. With `n ≤ 120` we pack on the order of 10⁵ – 10⁶ times per
   thread per minute, plenty for SA convergence on the small/medium instances.

4. **Trivially parallelisable.** Each thread is a self-contained search with
   its own RNG seed, B*-tree, and packer. We just take the best feasible
   solution at the end. No locks, no shared state — this is the same
   "embarrassingly parallel" structure PARSAC documents in its PyTorch
   batched implementation, ported to native threads.

5. **Constraint-fixing moves preserve SA's mixing.** PARSAC's
   constraint-fixing move (`M6 FixBoundary` here) is *always accepted*, even
   if cost rises. Without it, SA stalls in local minima where every standard
   move would break a hard constraint. We trigger it with probability 0.0005
   per step, matching PARSAC §3.2.1.

6. **Modular, testable C++.** Eight `.cpp` files with one well-defined job
   each. The toy benchmark exercises every code path (preplaced, fixed, MIB,
   grouping, boundary corner) and the build/run smoke-test passes.

### Weaknesses (be honest with the team & advisor)

These are real, and most of them get *worse* as `n` grows toward 120, which
matters a lot because the contest weights `Cost[i]` by `e^n` — the
hardest-to-solve cases dominate the total score.

1. **Scalability of plain SA on `n ≈ 100–120`.** The contest spec quotes
   that even distributed SA hits ≥10% area or HPWL gap on 60-partition cases
   (and the v9 spec [3rd page] says so explicitly). We will hit the same
   wall. With our current 30-second budget we expect on average a 10–20%
   HPWL gap and similar area gap on `n ≥ 100`, which by eq. (2) of v9 puts
   our `Cost[i]` in the 1.15–1.30 band — fine for partial credit, far from
   leaderboard-leading.

2. **No use of the optimal-by-construction training labels.** FloorSet
   ships 1M `tree_sol`/`fp_sol` ground-truth pairs. PARSAC throws this away.
   Diffusion-model / GNN approaches use it for a near-instant warm start.
   The contest organisers stated their internal diffusion model reaches
   high-fidelity solutions in sub-minute intervals — and our pure-SA
   approach cannot, by construction, match that.

3. **Move proposal is uniformly random.** All blocks are equally likely to
   be moved/swapped. On a 100-block instance the chance of touching a *useful*
   block is 1/100. Learned move ordering (Liu KDD'22 [23], He ICCD'20 [9]) or
   a GNN-scored move policy would multiply our effective iteration count.

4. **No analytical refinement of geometry.** Once SA fixes a topology, the
   actual `(w_i, h_i)` for each soft block is just sampled within the
   aspect-ratio band. A QP / quasi-Newton stage (Ji 2021 [13]) with the
   topology fixed would shave the last 3–5% off both HPWL and area for
   essentially free. We do *not* do this.

5. **MIB is hard.** Our `M5 MibSync` move synchronises one MIB group per
   step. If two MIB groups share blocks (possible in industrial designs,
   though not common in FloorSet-Lite), the moves can fight each other. A
   constraint-propagation pass would be safer; we don't have one.

6. **Soft-block dimension move is a coarse rejection sampler.** We sample
   `w` uniformly in the AR band and recompute `h = a/w`. For tightly-packed
   instances near the outline this rejects often and slows the chain.
   Replacing it with a gradient / coordinate-descent step would help.

7. **Aspect-ratio band is a heuristic.** v9 doesn't actually constrain the
   AR of a soft block — only its area. We hardcoded `[0.25, 4.0]` to keep
   the packer well-behaved, but if the optimal layout for a particular
   instance wants a 10× thin sliver, we won't find it.

8. **Single-objective SA cost is weighted, not Pareto.** We collapse HPWL,
   area, and the V terms into one scalar with constant weights. If a
   particular instance's optimum lies on a strange Pareto front, a fixed-
   weight SA won't find it. In practice this matters only on outliers.

### Expected outcome with current code (rough estimates)

| `n` range | typical HPWL gap | typical area gap | typical Cost[i] | weight share |
|---|---|---|---|---|
| 21–40   | 0.05 – 0.15 | 0.05 – 0.15 | 1.05 – 1.15 | < 0.001 |
| 41–80   | 0.10 – 0.25 | 0.10 – 0.25 | 1.10 – 1.30 | ~0.05    |
| 81–120  | 0.15 – 0.40 | 0.15 – 0.40 | 1.20 – 1.50 (with possible infeasible outliers ⇒ 10) | **~0.95** |

The 81–120 band is everything: 95% of the weight, by far the hardest
instances, and the place plain SA struggles most. Pass alpha-test? Yes, very
probably. Win the contest? Not on this code alone.

---

## Part 2 — Recommended best method

The honest answer: **submit the C++ baseline first (it'll pass alpha-test),
but plan from day one to layer ML on top.** The contest is explicitly
designed to reward ML-augmented submissions ("Scalability Barriers" section
of v9), and the baselines are constructed so that classical methods asymptote.

Here is the path I recommend, in priority order:

### Phase 1 — finish & harden the classical baseline (this week, 5/5)

Goal: pass alpha-test with the code you already have.

1. **Verify the output file format against the official `iccad2026contest`
   repo.** Our `save_solution()` writes `(id, x, y, w, h)` per line. The
   official `evaluate_floorplan()` likely expects a tensor in `fp_sol`
   shape `[n,4]`. Add an adapter in `tools/` if needed. *This is the single
   most common cause of "infeasible" submissions that would otherwise
   pass.*
2. **Add a quasi-Newton geometry-refinement pass** (Ji 2021 [13]) after SA
   converges. Topology is fixed by then; only `(w_i, h_i)` and the
   continuous packing offsets vary. This is ~150 lines of code and
   typically buys 3–5% on both HPWL and area at near-zero runtime cost
   because it's `O(n)` per iteration and converges in <50 iterations.
3. **Per-instance time budget.** The contest scores `RuntimeFactor` per
   instance, capped at -30%. Spending 60 s on `n=21` is wasted; spending
   only 30 s on `n=120` is leaving free score on the table. Budget like
   `t = 5 + 0.5·n` seconds.
4. **Run the validation set (100 samples) end-to-end.** Tune the SA
   parameters (`P_AR`, `P_MIB`, `P_FIX`, `iters_per_block`, T-schedule
   constants) by sweeping on `n ∈ {30, 60, 90, 120}`. The 4/28 schedule's
   5/12 and 5/19 slots are exactly for this.

This gets you to roughly the leaderboard median, which is what the alpha-
test asks for.

### Phase 2 — ML warm-start (after 5/19, before 5/26)

Goal: bring the n=80–120 cases from "10–20% gap" to "<5% gap" without
spending more compute.

The contest hint is unsubtle: their reference is a **diffusion model**.
Concretely, what works for FloorSet-Lite (and is feasible in a 2-week
window):

- **Train a small GNN policy on the 1M `(constraint_graph → tree_sol)`
  pairs.** Input = block-feature graph (area, constraint flags) + B2B/P2B
  edges. Output = a probability distribution over the next move (parent,
  side, rotate). The GNN doesn't have to be smart — it has to reduce SA
  iteration count.
- **Replace the uniform random move sampler in `moves.cpp` with the GNN's
  proposal distribution.** PARSAC even discusses this in its discussion
  section. We keep the SA acceptance criterion unchanged so feasibility
  guarantees and hard-constraint enforcement are *the same code as today*.
- **Optionally**, use the GNN to produce a B*-tree warm-start (decode mode
  rather than scoring mode). PARSAC's published numbers improve by 30–50%
  in time-to-target with a learned warm-start.

A GNN move-proposer is the single best return-on-effort improvement and
fits inside the 5/19→5/26 alpha-test code freeze window if the team starts
the dataset pipeline this weekend.

### Phase 3 — diffusion model (post-alpha)

Goal: leaderboard-leading score on n=100–120.

Set up a diffusion model exactly along the lines of the contest-organiser
hint. The structure is well-understood:

- Forward process: noisy `(x_i, y_i, w_i, h_i)` tensors derived from
  `fp_sol` ground truth.
- Reverse process: condition on the constraint graph + block areas;
  iteratively denoise to a candidate floorplan.
- Post-process with one short SA pass to "snap" the diffusion sample into
  feasibility (it will almost always need this — diffusion gives "almost
  legal" floorplans, and our SA can fix the last few percent).

This is a 4–8 week project and is what the contest organisers themselves
hint they used internally. Realistic to start *after* the alpha-test
deliverable is locked.

### Concrete recommendation

The best-realistic-method, given your team size (3) and timeline (alpha-test
5/26, finals later), is:

> **PARSAC (this codebase) + a small GNN move proposer trained on
> FloorSet-Lite ground-truth, with a quasi-Newton refinement pass at the
> end of each chain. Target diffusion-model post-processing as the final
> push if there's time after alpha-test.**

This gives you:
- A working alpha-test submission *today* (the smoke-test passed).
- A clear, well-scoped ML extension that doesn't require throwing away any
  of the SA infrastructure.
- A research-grade Phase 3 ceiling that's competitive with the
  organisers' own diffusion baseline.

### What I would *not* do, even if tempted

- **Don't switch to sequence-pair or TCG just because they're "more
  general".** B*-tree + PARSAC's anchored-blocks is the simplest structure
  that handles every v9 constraint, and the literature is unanimous that
  for `n < 200` B*-tree wins on speed.
- **Don't rely on a single super-long SA chain.** The contest's exponential
  weight-by-`n` and runtime cap at `0.3` mean *speed is part of the score*.
  Many short multi-seed chains beat one long chain for both robustness and
  RuntimeFactor.
- **Don't reverse-engineer FloorSet's generator.** v9 footnote 6 explicitly
  disqualifies submissions doing this.
- **Don't tune to the 100 validation cases.** They're public; the test is
  hidden. Train ML on the 1M training set, validate on the 100, and treat
  the 100 only as a smoke-test.

---

## Summary

The PARSAC + B*-tree + Fast-SA implementation we built is a clean,
modular, *correct* baseline that will pass the 5/26 alpha-test. Its
primary weakness is the same weakness the contest authors explicitly
designed to expose — classical SA scales poorly with `n` exactly where
the score weight is highest. The natural fix is a learned move proposer
(GNN trained on the supplied `tree_sol` ground truth) layered on top of
the existing SA, with diffusion-model post-processing as the eventual
ceiling. The code as written makes that extension straightforward
because move proposal lives entirely in `moves.cpp` behind a clean
interface.
