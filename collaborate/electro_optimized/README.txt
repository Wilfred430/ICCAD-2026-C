ICCAD 2026 CAD Contest - Problem C (The FloorSet Challenge, FloorSet-Lite)
=========================================================================
Submission: analytical / electrostatic floorplacement (source-code form)

----------------------------------------------------------------------
1. WHAT THIS IS
----------------------------------------------------------------------
A continuous, gradient-based global placer (the ePlace / DREAMPlace
paradigm) specialised for the small FloorSet instances (n <= 120),
followed by exact legalisation and soft-constraint repair.

Entry point : electro_optimizer.py
Optimizer   : class MyOptimizer(FloorplanOptimizer) -- implements solve().

The contest harness imports electro_optimizer.py and calls solve() once
per test case, exactly as in the provided evaluation script:

    python iccad2026_evaluate.py --evaluate electro_optimizer.py

All five .py files must sit in the SAME directory (electro_optimizer.py
inserts its own directory on sys.path and imports the other four).

----------------------------------------------------------------------
2. FILES
----------------------------------------------------------------------
electro_optimizer.py   Entry point (FloorplanOptimizer subclass, multi-seed
                       driver, seed ranking).
analytical_place.py    Continuous global placement (PyTorch / Adam).
legalize.py            Constraint-graph compaction + push-apart legalisation
                       (exact zero overlap; shapes unchanged).
soft_repair.py         Boundary / grouping soft-constraint repair passes.
electro_parallel.py    Per-seed worker (place -> legalize -> repair).
requirements.txt       Python dependencies (torch, numpy).
README.txt             This file.

iccad2026_evaluate.py is NOT bundled -- it is provided by the contest
environment and imported at evaluation time.

----------------------------------------------------------------------
3. DEPENDENCIES
----------------------------------------------------------------------
Python 3.10+ , PyTorch >= 2.0 , NumPy >= 1.24  (see requirements.txt).
No internet access required at run time. CPU-only by default; a GPU is
NOT needed (this is a small problem and runs on CPU).

    pip install -r requirements.txt

----------------------------------------------------------------------
4. DEFAULT CONFIGURATION (as submitted)
----------------------------------------------------------------------
The submitted default GUARANTEES every block lands in the first quadrant
(x, y >= 0, the contest's (0,0)=lower-left origin convention). This is
enabled by two built-in defaults set at the top of electro_optimizer.py:

    ELECTRO_CLAMP  = 1   (in-optimization lower-wall clamp)
    ELECTRO_NONNEG = 1   (floor-aware legalize + repair chain)

Measured on the 100-case validation set (local evaluator, RuntimeFactor=1):
    Total Score = 2.966 , 100/100 feasible , all coordinates >= 0.

All behaviour is overridable via environment variables (optional):
    ELECTRO_CLAMP=0 ELECTRO_NONNEG=0  -> lower-cost config that allows
                                         negative coordinates (Total 2.334).
    ELECTRO_SEEDS=N                   -> N multi-start seeds (default 1).
    ELECTRO_ITERS=K                   -> placement iterations (default 600).

No code change is needed to run -- the defaults are production-ready.
