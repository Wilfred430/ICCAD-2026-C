# FloorSet-Lite SA Floorplanner (ICCAD 2026 Contest C)

A C++17 implementation of a PARSAC-style B*-tree + Fast Simulated Annealing
floorplanner targeted at the **ICCAD 2026 FloorSet Challenge** (FloorSet-Lite,
spec v9, 2026-03-25).

The codebase is structured to match the design we agreed on in the 4/28 meeting
notes:

- **Framework** — PARSAC (Mostafa et al., arXiv 2405.05495) augmented with
  Fast-SA (Chen & Chang, TCAD 2006).
- **Representation** — B*-tree (indexed array, no pointer chasing) so a
  per-search snapshot is a single `memcpy`.
- **Per-block soft-shape change** and **MIB synchronisation** — handled as SA
  neighbourhood moves (no fixed aspect-ratio limits hardcoded in the cost).
- **Solution-quality strategy** — multi-thread, multi-seed independent SA
  chains; the best feasible result wins.
- **Constraint mapping** (one item per row, per the 4/28 notes):
  | Constraint | Method |
  |---|---|
  | internal / external connect, terminals | v9 centroid-Manhattan HPWL plugged into the PARSAC SA cost |
  | grouping | PARSAC grouping cost term + abutment check |
  | preplaced | PARSAC anchored-block B*-tree (§3.2.2) |
  | fixed-shape | dims locked in the packer (treated as hard) |
  | MIB | dedicated SA move that synchronises all blocks in a group |
  | boundary | PARSAC constraint-fixing moves (§3.2.1) |
  | soft-block w/h change | SA aspect-ratio move (no AR-band hardcoded in cost) |
  | parallelism | `std::thread` chains, one seed per thread |
  | overlap-free check | post-pack assertion + Python `verify_solution.py` |

## Build

```bash
cd floorplanner
make           # release: -O3
make debug     # -O0 -g -DDEBUG
make check     # quick smoke-test on benchmarks/toy.txt
```

The build is a plain C++17 GNU/Clang compile, no external dependencies; it
needs only `pthread` (already in the Makefile).

## Run

```
./floorplanner <input.txt> <output.sol> [options]

  --time    S     wall-clock seconds for each chain   (default 30)
  --threads N     number of independent SA chains     (default 8)
  --seed    S     base seed; thread i uses seed+i     (default 1)
  --iters-per-block K   inner iterations per temp     (default 30)
  --verbose       periodic [SA] traces during the run
```

Example:
```
./floorplanner benchmarks/toy.txt benchmarks/toy.sol --time 5 --threads 4 --verbose
```

## Pipeline

```
       FloorSet pkl                            our solver                       contest
  (HF / IntelLabs/FloorSet)                   floorplanner                  evaluator (Python)
        │                                          ▲                              ▲
        │  python tools/floorset_to_txt.py         │                              │
        ▼                                          │                              │
   *.txt instance ─ benchmarks/<name>.txt ─────────┘                              │
                                                                                  │
                            *.sol  ───────────────────────────────────────────────┘
                                              python tools/verify_solution.py
                                              (independent v9 cost reimpl.)
```

1. `tools/floorset_to_txt.py` converts a FloorSet-Lite `*.pkl` sample into the
   plain-text format we consume. It auto-fills `BASELINE_HPWL` from
   `metrics_sol[6] + metrics_sol[7]` and `BASELINE_AREA` from `metrics_sol[0]`.
2. `floorplanner <in.txt> <out.sol>` runs the SA optimiser.
3. `tools/verify_solution.py` reimplements the v9 cost in pure Python and is a
   sanity check independent of the C++ code.

## Project layout

```
floorplanner/
├── Makefile
├── README.md
├── EVALUATION.md          ← pros/cons of this approach + recommended next steps
├── include/
│   ├── types.hpp          Block, Terminal, Net, FloorplanInstance, BoundaryEdge
│   ├── btree.hpp          BTree (indexed array)
│   ├── packer.hpp         Contour-based packer
│   ├── cost.hpp           v9 cost & feasibility check
│   ├── moves.hpp          SA neighbourhood (M1–M6) + revert
│   ├── sa.hpp             Fast-SA driver
│   ├── parser.hpp         Plain-text I/O for instances
│   └── parallel.hpp       Multi-seed driver
├── src/
│   ├── btree.cpp
│   ├── packer.cpp         contour DFS pack; preplaced (anchored) blocks; overlap check
│   ├── cost.cpp           HPWL (centroid-Manhattan), Areabbox, V_grouping/V_mib/V_boundary,
│   │                       SA cost (weighted sum) + contest cost (eq. 2 of v9 spec)
│   ├── moves.cpp          M1 Rotate, M2 Move, M3 Swap, M4 AspectRatio,
│   │                       M5 MibSync, M6 FixBoundary (always-accept)
│   ├── sa.cpp             3-stage Fast-SA, T1 calibrated to p_accept = 0.99
│   ├── parallel.cpp       N std::thread chains with different seeds
│   ├── parser.cpp         text-format reader/writer
│   └── main.cpp           CLI
├── tools/
│   ├── floorset_to_txt.py  pkl  ─►  our text format
│   └── verify_solution.py  pure-Python v9 cost reimpl.
└── benchmarks/
    └── toy.txt             6-block synthetic test (preplaced, fixed, MIB, grouping,
                            boundary corner — exercises every code path)
```

## Input format (txt)

We use a plain-text serialisation (the actual FloorSet repository uses
PyTorch tensors, hence the `floorset_to_txt.py` adapter):

```
N_BLOCKS    <k>
N_TERMINALS <r>
BASELINE_HPWL <real>            # from metrics_sol[6] + metrics_sol[7]
BASELINE_AREA <real>            # from metrics_sol[0]
OUTLINE     <W> <H>             # baseline outline (advisory)

TERMINALS
<id> <x> <y>   ... r lines

BLOCKS
# id area is_fixed is_preplaced  w_in h_in x_in y_in  mib_grp grp_id bedge ar_min ar_max
<row per block, k lines>

B2B <m_int>
<i> <j> <weight>   ... m_int lines

P2B <m_ext>
<terminal_id> <block_id> <weight>   ... m_ext lines

GROUPS <P>
<size> <id1> <id2> ...     ... P lines

MIB <Q>
<size> <id1> <id2> ...     ... Q lines

END
```

`bedge` is encoded as in `BoundaryEdge` of `types.hpp` (`-1` = no constraint,
`0..3` = L/R/B/T edge, `4..7` = corners BL/BR/TL/TR).

## Output format (sol)

```
N_BLOCKS <k>
# id  x  y  w  h
0  <x0>  <y0>  <w0>  <h0>
1  <x1>  <y1>  <w1>  <h1>
...
```

Coordinates are the lower-left corner of each block, matching the contest
spec. Convert to whatever final form the official `iccad2026_evaluate.py`
expects when submitting; the (id, x, y, w, h) tuple is the only information
needed.

## Smoke-test result

On the bundled `benchmarks/toy.txt` (6 blocks, 5 s wall-clock, 4 threads):

```
[main] best thread=2 feasible=1 contest_cost=1.00175
       hpwl_gap=-0.59  area_gap=+0.59  V_rel=0
       (n_feasible_threads=4/4)  elapsed=5.01 s
[main] wrote benchmarks/toy.sol
```

The Python verifier independently reproduces `contest_cost = 1.0017`. The
toy is dominated by the small canvas (the OUTLINE constraint forces the area
gap up because the toy doesn't actually have any optimal area — it's
pseudo-baselined to 200), but the run shows: all hard constraints met
(preplaced 4 at (16,0) 9×5; fixed 5 at 6×5), MIB pair `{0,3}` shares
identical dims, group `{1,2}` is abutted, block 0 sits in the bottom-left
corner.

## What's next

See **EVALUATION.md** for a discussion of the strengths and weaknesses of
this approach and a concrete recommendation for the alpha-test deliverable.
