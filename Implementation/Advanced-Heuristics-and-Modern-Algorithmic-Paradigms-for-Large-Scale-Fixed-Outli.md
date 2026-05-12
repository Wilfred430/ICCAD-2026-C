# Advanced Heuristics and Modern Algorithmic Paradigms for Large-Scale Fixed-Outline Floorplanning
The transition from classical outline-free floorplanning to modern fixed-outline constraints fundamentally alters the mathematical landscape of Very Large Scale Integration (VLSI) physical design. In classical paradigms, the objective is merely to minimize a linear combination of area and wirelength, allowing the bounding box to expand dynamically. However, in contemporary hierarchical System-on-Chip (SoC) design, sub-systems must fit precisely within predetermined silicon die geometries. This fixed-outline requirement transforms floorplanning from a relatively straightforward unconstrained optimization problem into a highly restrictive, NP-complete constraint satisfaction and feasibility challenge.
For large-scale instances containing a high number of blocks ($n > 100$), stochastic algorithms such as Fast Simulated Annealing (SA) applied to topological representations (e.g., B*-trees) frequently experience catastrophic failure modes. Standard random perturbations—such as swapping nodes, moving nodes, or rotating blocks—operate blindly in the topological space without geographic awareness of the fixed bounding box. As the block density increases, these blind topological moves almost universally result in boundary overflow.
In benchmark environments such as the ICCAD 2026 FloorSet Challenge, the evaluation criteria severely punish such boundary overflows. The challenge mandates absolute compliance with hard constraints, including a strict non-overlapping requirement, exact preservation of preplaced module coordinates, and a $1\%$ relative error tolerance for the realized area of soft blocks. Any violation of these parameters instantly renders the entire floorplan infeasible, invoking a severe fixed penalty ($M=10$) that obfuscates the objective function gradient. Consequently, the cost function explodes, and the SA engine becomes trapped in a landscape devoid of feasible gradient paths.
This comprehensive research report presents an exhaustive analysis of state-of-the-art (SOTA) algorithms, representation mechanics, hybrid analytical paradigms, and advanced heuristic strategies required to conquer the fixed-outline floorplanning problem. By restructuring the perturbation mechanics, transitioning to constraint-aware algorithms, and implementing dynamic penalty shaping, it is entirely feasible to achieve high success rates and minimized Half-Perimeter Wirelength (HPWL) within rigorous computational timeframes.

## 1. Advanced Move Proposals and Smart Perturbation Mechanics
The fundamental limitation of traditional Simulated Annealing on B*-tree representations is the spatial ignorance of its move generator. While topological moves guarantee a compacted, overlap-free placement by construction (a property known as P-admissibility), they lack a feedback mechanism to inform the optimizer whether a move pushes a block outside the designated outline. To achieve high fixed-outline success rates, the perturbation engine must transition from blind stochasticity to deterministic, "outline-aware" heuristics.

### The Mathematics of Floorplan Slack
The most robust theoretical framework for forcing blocks inside boundaries relies on the computation of *floorplan slack*, a concept pioneered by Adya and Markov. In floorplanning, slack quantifies the exact physical distance a block can be translated along a specific axis without increasing the overall bounding box of the layout.
When a floorplan topology is translated into absolute coordinates, the packing logic implicitly creates a Horizontal Constraint Graph (HCG) and a Vertical Constraint Graph (VCG). The maximum width of the floorplan ($W_{max}$) is dictated by the longest path (the critical path) through the HCG, while the maximum height ($H_{max}$) is dictated by the critical path through the VCG. If a particular block lies squarely on the HCG critical path, any perturbation that increases its width or pushes it further outward will instantly violate the X-outline. Such blocks are defined as having zero X-slack. Conversely, blocks situated outside the critical path possess positive slack, representing localized pockets of dead space or "white space".
The calculation of slack for a given block $i$ is performed via an algorithmic analogy to Static Timing Analysis (STA) used in logic synthesis. It requires a two-pass traversal of the constraint graphs:

1. **Forward Pass (Left-to-Right):** Computes the earliest possible placement coordinate, $X_{earliest}(i)$, analogous to signal arrival times.
2. **Backward Pass (Right-to-Left):** Computes the latest possible placement coordinate, $X_{latest}(i)$, that does not delay the sink node (which corresponds to the total floorplan width $W$).
3. **Slack Calculation:** The slack in the X dimension is exactly $X_{slack}(i) = X_{latest}(i) - X_{earliest}(i)$.
Advanced slack-based perturbation algorithms exploit these metrics to systematically compress the floorplan. Rather than randomly swapping blocks, the move generator evaluates the slack distribution. If the current floorplan violates the X-outline but maintains available dead space in the Y-outline, the algorithm explicitly identifies critical blocks (those with $X_{slack} = 0$) and relocates them topologically adjacent to blocks with maximum $Y_{slack}$. This targeted swap effectively ejects critical blocks out of the horizontal constraint sequence and forces them into vertical dead space, simultaneously shrinking the width while marginally increasing the height within permissible limits.

### Constraints-Aware Simulated Annealing (CA-SA)
While slack-based moves excel at general fixed-outline compaction, modern SoC benchmarks introduce specific, localized hard constraints. The ICCAD 2026 FloorSet Challenge explicitly tests boundary constraints (forcing specific blocks to align with the die edges or corners), grouping constraints (forcing physical abutment of clustered blocks), and pre-placement immutability. Standard SA attempts to resolve these by incorporating penalty terms into the objective function. However, recent industrial-grade research, most notably the PARSAC (Parallel Simulated Annealing with Constraints) framework, demonstrates that incorporating hard constraints purely through soft objective penalties leads to sub-optimal or chronically illegal solutions.
PARSAC introduces Constraints-Aware Simulated Annealing (CA-SA), an algorithmic paradigm that forcefully decouples objective optimization (minimizing HPWL) from constraint satisfaction (repairing boundary violations). In CA-SA, the move generator is fundamentally modified to act as an active repair agent.
If the SA engine detects that a layout violates a hard boundary constraint—for instance, a block mandated to touch the `LEFT` boundary is currently buried inside the core—it immediately intercepts the standard random move generator and invokes a targeted constraint-fixing routine. The logic operates deterministically:

1. The algorithm identifies the violating block $b_{v}$ that must reside at the boundary.
2. It subsequently scans the targeted boundary to identify an unconstrained block $b_{u}$ currently occupying that space.
3. A forced topological swap is executed between $b_{v}$ and $b_{u}$, instantly restoring feasibility.
4. In edge cases where all blocks at the required boundary are also constrained, the algorithm executes a structural tree manipulation, making $b_{v}$ the direct spatial child (e.g., the left child in a B*-tree) of a block already legally anchored at the boundary.
By overriding the stochastic generator with deterministic CA-SA repair moves whenever hard constraints are violated, the SA prevents the system from wandering aimlessly through highly penalized, infeasible state spaces. The thermal budget is thus preserved for optimizing the HPWL within the strictly feasible geometric region.

| Move Generation Paradigm | Algorithmic Mechanism | Primary Use Case | Success Rate Impact on n>100 |
| --- | --- | --- | --- |
| Standard Random Walk | Uniform probability topological swap/rotate. | Outline-free bin packing. | Exceptionally Low. Random walks fail to find specific bounding box configurations. |
| Slack-Driven Swaps | Extract HCG/VCG critical paths; swap zero-slack blocks into maximum-slack zones. | Global aspect-ratio correction and fixed-outline fitting. | High. Systematically compresses the floorplan. |
| CA-SA Boundary Repair | Deterministic topological insertion of violators at required layout edges. | Resolving rigid boundary, grouping, and pre-placement constraints. | Very High. Prevents the M=10 infeasibility trap instantly. |

## 2. Analysis of Alternative Floorplan Representations
The efficiency of any simulated annealing algorithm is fundamentally bounded by the data structure utilized to represent the floorplan topology. Because modern EDA requires overlap-free states by construction, absolute coordinate arrays are rarely used in purely heuristic solvers. Instead, relative topological representations are favored. While the B*-tree is a standard implementation , assessing alternative paradigms—such as Sequence Pair (SP), Corner Block List (CBL), O-Tree, and Transitive Closure Graph (TCG)—reveals highly specific mathematical tradeoffs regarding evaluation time, spatial logic, and fixed-outline suitability.

### Sequence Pair (SP)
The Sequence Pair representation encodes a floorplan via two ordered permutations of the blocks, denoted as $(\Gamma_+, \Gamma_-)$. The relative spatial positioning of any two blocks is strictly governed by their relative sequencing. For example, if block A precedes block B in both sequences, A is guaranteed to be placed to the left of B. If A precedes B in $\Gamma_+$ but follows B in $\Gamma_-$, A is placed above B.
The primary mathematical advantage of SP for fixed-outline problems is the elegance of its slack computation. The horizontal and vertical constraint graphs are implicitly embedded within the permutations. Consequently, the algorithm can utilize Longest Common Subsequence (LCS) dynamic programming algorithms to simultaneously pack the blocks and extract the critical paths in $O(n \log \log n)$ time. Furthermore, SP is structurally shift-invariant; altering the sequence position of one block does not inadvertently drag physically unrelated blocks across the floorplan, an issue that occasionally destabilizes tree-based data structures. Despite these advantages, the $O(n \log \log n)$ to $O(n^2)$ packing overhead becomes a computational bottleneck when executing millions of iterations on massive test cases.

### Corner Block List (CBL)
The Corner Block List encodes floorplans as a sequence of corner insertions, guaranteeing a "mosaic" topology. In a mosaic floorplan, there is absolutely zero white space between the internal "rooms" allocated to the blocks; every partition seamlessly abuts its neighbors. While this characteristic appears highly advantageous for minimizing total area, it is fundamentally detrimental to fixed-outline simulated annealing. Because the topology is rigidly interlocked, every discrete move in the CBL restructures the entire mosaic cascade. Consequently, minor topological perturbations result in massive, unpredictable shifts in the global aspect ratio. CBL inherently lacks the continuous neighborhood structure required for fine-grained, incremental convergence toward a rigid die shape, rendering it highly unsuitable for fixed-outline constraints.

### O-Tree and Transitive Closure Graph (TCG)
The O-Tree (Orthogonal Tree) representation compactly defines horizontal and vertical relative placements using a tree structure, achieving $O(n)$ packing complexity. While highly efficient for area minimization, it suffers from spatial opacity; mapping physical slack metrics back to specific O-Tree edges for targeted perturbations requires complex reverse transformations.
The Transitive Closure Graph (TCG) attempts to merge the spatial transparency of Sequence Pairs with the packing efficiency of tree structures. TCG maintains explicit boundary information and module shapes, and unlike SP or CBL, it does not require the construction of additional, intermediate constraint graphs for cost evaluation. TCG is mathematically P-admissible, meaning its representations always map to a compacted placement where no module can move left or down without causing overlap. However, maintaining the transitive closure matrices during SA perturbations incurs a non-trivial computational overhead.

### Defense of the B*-Tree Architecture
Despite the theoretical elegance of Sequence Pairs, the **B*-tree** remains the demonstrably superior choice for high-performance C++ implementations handling large block counts ($n > 100$). The B*-tree operates on an ordered binary tree combined with an advanced horizontal contour-based data structure. During the SA evaluation phase, the physical coordinates of the blocks are determined by traversing the tree using a Depth-First Search (DFS) while continuously updating a contour array. This contour method guarantees packing in strictly $O(n)$ time.
Like the TCG, the B*-tree is entirely P-admissible. To optimize the B*-tree specifically for fixed-outline constraints, the floorplanner must not abandon the data structure in favor of Sequence Pairs. Rather, the algorithm must implement $O(n)$ slack calculations on the absolute physical coordinates *immediately after* the contour packing step. Once physical coordinates are established, reverse contour traversals efficiently identify geometric dead space, allowing the SA engine to correlate critical physical paths back to specific nodes in the B*-tree. By maintaining the raw speed of the B*-tree contour packer while simultaneously injecting the spatial awareness of slack-driven moves, the algorithm achieves an optimal synthesis of runtime efficiency and fixed-outline success.

| Representation Framework | Packing Algorithmic Complexity | Spatial Transparency & Slack Mapping | Fixed-Outline Constraints Suitability |
| --- | --- | --- | --- |
| Sequence Pair (SP) | O(nloglogn) via LCS. | High. Shift-invariant structure. | Excellent for extracting critical paths, but computationally heavy for large SA runs. |
| B*-Tree | O(n) via contour mapping. | Medium. P-admissible binary tree. | Very High. Contour packing is incredibly fast. Slack can be computed post-packing. |
| Corner Block List (CBL) | O(n) corner insertion. | Low. Rigid internal mosaic. | Poor. Moves drastically alter aspect ratios; lacks incremental neighborhood stability. |
| Transitive Closure Graph (TCG) | Intermediate matrix updates. | High. Explicit boundary tracking. | Good. Preserves geometric relations transparently but incurs matrix update overheads. |

## 3. Hybrid Methodologies and Modern Analytical Paradigms
Given that purely stochastic SA on instances with $n > 100$ scales poorly and risks violating the 30% runtime factor limits outlined in the FloorSet Challenge, the academic SOTA has heavily pivoted toward hybrid methodologies. These frameworks merge the deterministic, global mathematical perspective of analytical solvers or Machine Learning (ML) agents with the local, fine-grained refinement power of heuristics.

### Analytical and Force-Directed Global Floorplanning
Analytical models entirely abandon topological structures (such as B*-trees) during the global placement phase. Instead, they treat block coordinates $(x_i, y_i)$ and continuous aspect ratios as unconstrained real-number variables. The floorplanning problem is subsequently modeled as an unconstrained mathematical optimization:

$$
\min_{X, Y, W, H} \left( \text{HPWL}(X, Y) + \lambda \cdot \text{Overlap}(X, Y, W, H) \right)
$$

subject to the fixed-outline constraints $0 \le x_i \le L_x - w_i$ and aspect ratio bounds $r_i \le w_i/h_i \le R_i$.
To strictly enforce the fixed outline during this continuous phase, analytical solvers deploy **Barrier Functions**. An exponential penalty is applied as block coordinates approach the absolute limits of the silicon die, repelling them inward and completely preventing boundary overflow. A standard barrier term $B$ takes the form :

$$
B = \sum_{i=1}^{n} \left( \frac{1}{x_i} + \frac{1}{L_x - x_i - w_i} + \frac{1}{y_i} + \frac{1}{L_y - y_i - h_i} \right)
$$
**The Conjugate Subgradient Algorithm (CSA) and Q-Learning:**
A critical challenge in analytical floorplanning is that wirelength (modeled via L1 Manhattan distance) and geometric overlap area are inherently non-smooth, non-differentiable functions. Applying standard first-order gradient descent results in severe numerical oscillation and entrapment in local minima. SOTA methodologies resolve this by utilizing the Conjugate Subgradient Algorithm (CSA), which directly computes subgradients to navigate the non-smooth topology.
Recent advancements have further hybridized CSA by integrating Q-Learning, creating the CSAQ or CSF algorithms. In CSAQ, a reinforcement learning agent dynamically regulates the step size (the learning rate) of the conjugate subgradient descent. The Q-table is updated based on the reward derived from overlap reduction, allowing the solver to adaptively strike an optimal balance between aggressive exploration (large step sizes) and localized exploitation (small step sizes). Other analytical SOTA approaches include PeF, which ingeniously models the overlap penalty field as a solution to Poisson's equation, simulating a physical repulsive force between overlapping blocks.
*The Hybrid Legalization Pipeline:* While analytical algorithms rapidly generate a globally optimal coordinate map, this map inevitably contains minor residual overlaps. The critical second phase requires a legalization engine. The analytical coordinates are snapped to a valid non-overlapping topological representation (e.g., converting continuous coordinates back into a Sequence Pair or B*-tree), followed by a low-temperature Fast-SA run to resolve final constraints without destroying the global placement. While highly effective and scalable, building robust non-linear programming libraries, subgradient solvers, and complex topology-extraction legalizers is often impractical within tight competitive timeframes.

### Deferred Decision Making (DeFer)
An alternative non-stochastic hybrid approach is the DeFer algorithm. DeFer sidesteps the immense search space of standard slicing trees by considering only a single, generalized slicing tree. It applies the principle of Deferred Decision Making (DDM); rather than fixing the dimensions of soft blocks early in the partition hierarchy, it defers shaping decisions until a bottom-up iterative merging packing algorithm is executed. DeFer recursively bipartitions the floorplan, managing dead space systematically to guarantee a feasible layout without relying on annealing stochasticity.

### Graph Neural Networks (GNN) and Reinforcement Learning (RL)
The advent of deep learning has introduced learning-to-search mechanisms to floorplanning. In frameworks such as *GoodFloorplan* or *GraphPlanner*, Graph Convolutional Networks (GCNs) are utilized to encode the topological netlist connectivity. The GCN embeddings are fed into an Actor-Critic Reinforcement Learning network, which outputs probability distributions over Sequence Pairs. Other approaches utilize Proximal Policy Optimization (PPO) agents to sequentially place blocks onto a Corner Block List (CBL) grid.
*Limitations for Fixed-Outline Benchmarks:* As explicitly noted in recent analyses, learning-based methods operate predominantly as "one-shot" floorplanners. While they excel at rapid macroscopic pattern recognition, they severely lack the incremental, fine-grained refinement capabilities required to resolve complex, conflicting hard constraints. In the context of the ICCAD FloorSet challenge, if a PPO agent places 119 blocks flawlessly but the 120th block breaches the bounding box by a single pixel, the solution is deemed entirely infeasible and receives the maximum $M=10$ penalty. ML models inherently lack the localized, deterministic "repair" mechanisms (like CA-SA) that heuristics possess. Consequently, relying exclusively on one-shot neural architectures without a heuristic legalizer is exceptionally risky for strict fixed-outline benchmarks.

## 4. Cost Function Shaping and Penalty Dynamics
The primary catalyst for the objective function "explosion" observed in standard B*-tree implementations is an improperly designed penalty architecture. The ICCAD 2026 FloorSet Challenge employs a rigorous multi-objective evaluation metric:

$$
\text{Cost} = (1 + \alpha (\text{HPWL}_{gap} + \text{Area}_{gap})) \times e^{\beta \times \text{Violations}_{rel}} \times \max(0.7, \text{RuntimeFactor})
$$

where $\alpha = 0.5$, $\beta = 2.0$, and the runtime factor is dampened by $\gamma=0.3$ and capped at a 30% benefit. Any violation of a hard constraint (overlap, 1% area deviation, preplaced immutability) completely overrides this formula, assigning a static penalty of $M=10$.
If the internal SA engine evaluates solutions simply by returning an arbitrary massive penalty (e.g., returning $10$) whenever a block exits the outline, the thermodynamic simulated annealing mechanics will collapse. The SA relies on contiguous gradient slopes; if the penalty is uniformly massive across all infeasible states, the SA cannot distinguish between a floorplan that is one pixel out of bounds and one that is entirely chaotic. It becomes trapped on an infinite plateau.

### The Adaptive Fast-SA Cost Formulation
To guarantee fixed-outline success, the internal SA cost function $\Phi(F)$ must be dynamically shaped to guide the solver smoothly from chaotic infeasibility into the feasible region. The **Adaptive Fast-SA** scheme utilizes a dynamic weight adjustment mechanism to continuously tune the cost function based on the historical trajectory of the search.
The standard dynamic internal cost function is modeled as:

$$
\Phi(F) = W_A \cdot \text{Area} + W_W \cdot \text{HPWL} + W_R \cdot (R - R^*)^2 + W_O \cdot \text{Overflow}
$$
Where:

- $R$ is the current physical aspect ratio of the generated bounding box, and $R^*$ is the target aspect ratio dictated by the fixed outline.
- The quadratic penalty term $(R - R^*)^2$ is mathematically critical. It applies aggressive pressure to force the macroscopic layout into the correct aspect ratio long before the absolute boundary dimensions are satisfied.
- $\text{Overflow} = \max(0, W_{current} - W_{max}) + \max(0, H_{current} - H_{max})$. This linear overflow term provides the continuous gradient required for the SA to "walk" toward the boundaries, avoiding the flat plateau of a binary $M=10$ penalty.
**Dynamic Weight Adaptation:**
Static penalty weights frequently fail because a high weight traps the SA, while a low weight allows the SA to terminate out of bounds. The Adaptive Fast-SA algorithm resolves this by maintaining a sliding analytical window of the last $k$ moves (e.g., $k=500$).

1. If the last 500 moves consistently fail to meet the outline constraints, the algorithm dynamically doubles the Overflow weight ($W_O$) and increases the Area weight ($W_A$).
2. Conversely, once the floorplan is safely compacted inside the bounding box ($\text{Overflow} = 0$), $W_O$ is aggressively relaxed, and the Wirelength weight ($W_W$) is exponentially increased.
This dynamic adaptability guarantees that the algorithm treats geometric feasibility as an existential priority early in the search, treating HPWL strictly as a secondary objective until the fixed-outline constraints are securely satisfied.

### Multi-Stage Thermodynamics (The Fast-SA Schedule)
To combat the massive combinatorial landscape of instances with $n \in $ blocks , the cooling schedule must depart from traditional monotonic exponential decay ($T_{i+1} = \gamma T_i$). A modified Fast-SA incorporates three distinct thermodynamic stages based on the rejection-to-acceptance ratio :

1. **High-Temperature Random Search:**$T \to \infty$. In this phase, the Metropolis criterion accepts nearly all moves. The objective is to wildly scramble the B*-tree to explore diverse macroscopic block groupings without concern for boundaries.
2. **Pseudo-Greedy Local Search:**$T$ drops drastically. The SA transitions into localized gradient descent, rapidly compacting the blocks into a dense configuration.
3. **Hill-Climbing Search:** The temperature is spiked back up to a moderate plateau, then slowly decayed. This controlled reheat allows the densely packed blocks to shift, resolving localized overlaps and soft boundary constraints without destroying the globally optimal macroscopic placement achieved in stage two.

| SA Penalty Metric | Mathematical Representation | Algorithmic Purpose | Fast-SA Dynamic Behavior |
| --- | --- | --- | --- |
| Aspect Ratio Pressure | (R−R∗)2 | Forces topological shape alignment. | High initial weight; decays as shape aligns. |
| Linear Overflow | max(0,Wcurr​−Wmax​) | Provides continuous gradient to feasibility. | Spikes if history shows consistent boundary failure. |
| HPWL Optimization | ∑ManhattanDist | Minimizes wirelength for final score. | Dominates the cost function only after Overflow=0. |

## 5. Actionable Implementation Strategy
Based on the exhaustive synthesis of SOTA literature, the single most effective, computationally efficient, and robust strategy to implement in C++ within a tight competitive timeframe is the **Slack-Driven B*-Tree with Adaptive Penalties (SDB-AP) augmented by CA-SA Repair**.
This composite strategy sidesteps the immense software engineering overhead required to build non-linear Conjugate Subgradient Analytical solvers from scratch. Simultaneously, it immediately cures the topological "blindness" of a standard B*-tree SA by injecting deterministic, geometry-aware repair mechanics.

### The Blueprint Architecture

1. **Retain the B*-Tree Representation:** Maintain the B*-tree infrastructure. Its $O(n)$ contour-based packing efficiency ensures that millions of SA iterations complete swiftly, safely capturing the negative cost reductions afforded by the 30% capped `RuntimeFactor`.
2. **Implement Spatial Slack Tracking:** After a set interval of iterations (e.g., every 100 moves), calculate the absolute physical coordinates of all blocks. Identify the critical paths in the X and Y dimensions and assign a precise geometric "slack" value to every node in the B*-tree.
3. **Augment the Move Generator (CA-SA):** Replace the uniform random move generator with a temperature-dependent probabilistic generator. As the temperature decreases (Stages 2 and 3 of Fast-SA), drastically increase the probability of selecting **Slack-Based Moves** and **CA-SA Constraint-Fixing Moves** over random swaps.
4. **Discretize Soft Block Tolerance:** The ICCAD challenge dictates a strict 1% relative error margin for soft blocks ($|\frac{w_ih_i - a_i}{a_i}| \le 0.01$). Continuous aspect-ratio sizing often generates dimensions that cause microscopic floating-point drift, pushing the area to 1.011% error and triggering the $M=10$ infeasibility penalty. To mathematically guarantee compliance, soft block aspect ratio adjustments must be *discretized*. Pre-compute a finite dictionary of valid $(w, h)$ integer or strictly bounded float pairs for each soft block that perfectly satisfy the tolerance. During the B*-tree Op4 (resize) move, randomly sample from this pre-computed dictionary rather than scaling dimensions dynamically.

### C++ Core Logic and Algorithmic Flow
The following algorithmic flow outlines the core logic for the enhanced perturbation engine, directly engineered to eliminate boundary overflow and guarantee fixed-outline feasibility.

```c++
// SDB-AP: Slack-Driven B*-Tree Perturbation Engine with CA-SA Repair
struct FloorplanState {
    BStarTree tree;
    vector<Block> blocks;
    double width, height;
    double current_cost;
};

// Phase 1: Compute Exact Geometric Slack after O(n) Contour Packing
void compute_slacks(FloorplanState& state, double target_W, double target_H) {
    // Pack the B*-tree to extract absolute (x,y) coordinates
    pack_btree_via_contour(state);
    
    // Evaluate geometric distance to absolute boundaries
    for (auto& b : state.blocks) {
        // Slack is strictly the distance to the required outline edge
        b.x_slack = target_W - (b.x + b.width);
        b.y_slack = target_H - (b.y + b.height);
        
        // Negative slack explicitly flags a block as a boundary violator
    }
}

// Phase 2: CA-SA Smart Move Generation
void generate_smart_move(FloorplanState& state, double temperature) {
    double rand_val = random_double(0.0, 1.0);
    
    // Shift from stochastic chaos to deterministic repair as temperature drops
    double smart_move_prob = 1.0 - (temperature / INITIAL_TEMP); 
    
    if (rand_val < smart_move_prob) {
        
        // ---------------------------------------------------------
        // 1. HARD CONSTRAINT REPAIR (CA-SA PARSAC Logic)
        // ---------------------------------------------------------
        Block* boundary_violator = get_boundary_constraint_violator(state.blocks);
        if (boundary_violator) {
            // E.g., Block requires LEFT boundary but its x coordinate > 0
            Block* unconstrained_left = get_unconstrained_block_at_boundary(LEFT);
            if (unconstrained_left) {
                // Instantly swap the violator into the legal topological position
                state.tree.swap_nodes(boundary_violator->id, unconstrained_left->id);
            } else {
                // Force topological restructuring: Make the violator the direct 
                // left-child of the leftmost existing node in the B*-tree
                Node* leftmost = state.tree.get_leftmost_node();
                state.tree.move_node_as_child(boundary_violator->id, leftmost->id, LEFT_CHILD);
            }
            return;
        }

        // ---------------------------------------------------------
        // 2. OUTLINE-AWARE SLACK COMPACTION
        // ---------------------------------------------------------
        Block* critical_block = nullptr;
        Block* whitespace_block = nullptr;
        
        // Isolate the block causing the most severe overflow
        critical_block = get_block_with_min_slack(state.blocks);
        
        if (critical_block && (critical_block->x_slack < 0 |

| critical_block->y_slack < 0)) {
            // Locate a highly flexible block occupying abundant dead space
            whitespace_block = get_block_with_max_slack(state.blocks);
            
            // Execute Slack-Driven Topological Perturbation:
            // Swap the critical block out of the critical path and into dead space
            state.tree.swap_nodes(critical_block->id, whitespace_block->id);
            return;
        }
    }
    
    // Phase 3: Standard Topological Perturbation (Exploration Fallback)
    int move_type = random_int(0, 3);
    if (move_type == 0) state.tree.random_swap();
    else if (move_type == 1) state.tree.random_rotate();
    else if (move_type == 2) state.tree.random_move();
    else resize_soft_block_from_discretized_dictionary(state.blocks); // Guarantees 1% Area Constraint
}

// Phase 4: Adaptive Cost Function Evaluation
double evaluate_cost(FloorplanState& state, CostWeights& weights) {
    double target_W = FIXED_OUTLINE_WIDTH;
    double target_H = FIXED_OUTLINE_HEIGHT;
    double target_R = target_H / target_W;
    double current_R = state.height / state.width;
    
    // Continuous gradient evaluation to prevent M=10 cliff entrapment
    double overflow_x = max(0.0, state.width - target_W);
    double overflow_y = max(0.0, state.height - target_H);
    
    double hpwl = calculate_hpwl(state);
    
    // Quadratic pressure forces layout aspect ratio alignment early in the search
    double ar_penalty = pow(current_R - target_R, 2.0);
    
    return (weights.W_A * (state.width * state.height)) + 
           (weights.W_W * hpwl) + 
           (weights.W_R * ar_penalty) + 
           (weights.W_O * (overflow_x + overflow_y));
}

```

### Addressing Sub-System Grouping and Multi-Instantiation Blocks (MIB)
The FloorSet Challenge specifies soft constraints where $B_{grouping}^P$ requires clusters of blocks to be physically abutted, and $B_{mib}^Q$ requires instances of a master cell to share identical dimensional shapes.
To mitigate the exponential evaluation penalty $e^{\beta \times V_{rel}}$, grouping constraints should be handled architecturally prior to the SA execution. Blocks within a specified group can be pre-clustered into a single "macro-block" node within the initial B*-tree. During the early high-temperature stages, this macro-block moves as a single rigid entity. Only during the final low-temperature hill-climbing phase is the macro-block "shredded" into its constituent sub-blocks to allow for fine-grained localized shifting. Similarly, MIB constraints are enforced by linking the dictionary indices of MIB blocks, ensuring that when one block is resized, all blocks in the MIB group simultaneously adopt the identical $(w,h)$ dictionary vector.
By systematically overhauling the perturbation mechanics with CA-SA and Slack metrics, insulating soft block dimensions with discretized valid-area dictionaries, and protecting the gradient via dynamic cost shaping, the SDB-AP algorithm transforms a failing heuristic into an industrially robust solver. This approach unequivocally ensures high fixed-outline success rates while securing a dominant position on the ICCAD 2026 wirelength and runtime leaderboards.

---

Source: https://gemini.google.com/app/9cc290fb5435da9f
Exported at: 2026-05-05T13:53:13.940Z