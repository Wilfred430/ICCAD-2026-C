"""Analytical / electrostatic-style global placement for FloorSet (Problem C).

Continuous, gradient-based global placement (the DREAMPlace / ePlace paradigm)
specialised for small FloorSet instances (n <= 120):

    minimise  WL  +  lam_ov*Overlap  +  lam_bb*BBoxArea
            +  lam_grp*Grouping  +  lam_bnd*Boundary

  * WL       : weighted L1 (Manhattan) centroid distance over b2b/p2b edges
               (matches the contest HPWL).
  * Overlap  : sum of pairwise differentiable overlap-area (the spreading force).
  * BBoxArea : compactness penalty.
  * Grouping : pull each cluster's members toward their common centroid so they
               end up abutting (soft-constraint V_grouping).
  * Boundary : pull each boundary block's required edge to the layout extreme
               (soft-constraint V_boundary).

Hard constraints handled by construction:
  * soft block area : exact, via shape = (sqrt(a)*exp(la/2), sqrt(a)*exp(-la/2)).
  * MIB same-shape  : members of a MIB group SHARE one log-aspect AND one (mean)
                      area, so they get byte-identical (w,h)  -> V_mib = 0.
  * fixed           : w,h locked to target; position free.
  * preplaced       : w,h and position locked; a static obstacle.
"""
from __future__ import annotations

import math
import os

import torch


def _valid(t, pad: float = -1.0):
    if t is None or t.numel() == 0:
        return None
    r = t[t[:, 0] != pad]
    return r if r.numel() > 0 else None


def _sabs(d, g):
    """Smooth differentiable |d| = sqrt(d^2 + g^2)  (g==0 -> exact abs).

    The raw L1 abs has a +/-1 subgradient with a kink at 0, so the wirelength
    term keeps jittering near convergence (Adam never sees curvature).  The
    g-smoothing gives a gradient d/sqrt(d^2+g^2) that shrinks to 0 as d->0, so
    connected centroids settle precisely on top of each other -- the smooth /
    weighted-average wirelength idea behind ePlace and DREAMPlace.  g is annealed
    toward 0 over the run so the final wirelength matches the true HPWL metric."""
    if g <= 0.0:
        return d.abs()
    return torch.sqrt(d * d + g * g)


def place(
    block_count: int,
    area_targets: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
    target_positions=None,
    iters: int = 600,
    lr: float = 0.02,
    seed: int = 0,
    device: str = "cpu",
    init_centers=None,
):
    N = int(block_count)
    if N == 0:
        return [], {}
    dev = torch.device(device)

    wl_smooth = float(os.environ.get("ELECTRO_WL_SMOOTH", "0"))  # 0 = exact L1
    # Module-area-growing: soft blocks start at 10% area and reach exact area by
    # 70% of the run.  Big win (full-100 3.545 -> 2.745) -- the shrunk blocks pack
    # into a tight outline before filling out, cutting area_gap and HPWL.
    area_grow0 = float(os.environ.get("ELECTRO_AREA_GROW", "0.1"))  # 1.0 = off
    grow_end = float(os.environ.get("ELECTRO_GROW_END", "0.7"))     # frac at full area
    ov0 = float(os.environ.get("ELECTRO_OV0", "0.1"))   # overlap penalty start
    ov1 = float(os.environ.get("ELECTRO_OV1", "2.5"))   # overlap penalty end
    bb0 = float(os.environ.get("ELECTRO_BB0", "0.24"))  # bbox penalty start
    bb1 = float(os.environ.get("ELECTRO_BB1", "0.04"))  # bbox penalty end
    # Fixed-outline containment pull -> denser packing, lower area_gap
    # (subset 2.604 -> 2.537).  0 = off.
    lam_out = float(os.environ.get("ELECTRO_LAM_OUT", "2.0"))
    target_util = float(os.environ.get("ELECTRO_TARGET_UTIL", "0.85"))
    # Hard canvas walls at x=0, y=0 (projected gradient): after each step clamp
    # every movable block's lower-left corner to >= 0, so the layout stays in the
    # first quadrant (the contest's origin convention) and blocks can sit flush on
    # the X / Y axes -- a hard wall packs tighter than a soft repulsion.
    clamp_canvas = os.environ.get("ELECTRO_CLAMP", "0") == "1"
    # Engage the first-quadrant clamp only after this fraction of the run, so the
    # global structure forms unconstrained first and blocks then slide smoothly
    # onto the axes (rather than being pinned to the x=0/y=0 walls from iter 0,
    # which piles them in the corner and inflates area).  0 = clamp from the start.
    clamp_start = float(os.environ.get("ELECTRO_CLAMP_START", "0.0"))
    # Quadratic boundary penalty (canvas origin walls at x=0, y=0): a smooth,
    # differentiable confinement that pushes blocks back into the first quadrant
    # during optimization (the standard analytical fixed-outline technique --
    # gentler than a hard clamp).  Weight ramps up over the run.  0 = off.
    lam_wall = float(os.environ.get("ELECTRO_WALL", "0"))
    lam_wall_lin = float(os.environ.get("ELECTRO_WALL_LIN", "0"))  # L1 exact-penalty wall
    # External (pin/terminal) wirelength weight.  Boosting it drags pin-connected
    # blocks onto their fixed terminals -> lower HPWLext AND anchors the layout to
    # the (positive-coordinate) terminal frame.  Subset 2.537 (w=1) -> 2.300 (w=11);
    # smooth basin ~8-18, overshoot >=25.  10 = robust default.
    ext_wl = float(os.environ.get("ELECTRO_EXT_WL", "10.0"))
    # eDensity FFT density field (ePlace / DREAMPlace fence-region style).  A
    # single electrostatic density penalty on a FIXED canvas [0,Wc]x[0,Hc]
    # anchored at the origin, with Neumann BC solved by DCT (cosine transform).
    # It (a) spreads blocks to uniformly fill the canvas -> drives utilization to
    # ed_util (cuts area_gap), and (b) confines the whole layout into the first
    # quadrant -> kills the negative-coord drift WITHOUT the score cost of the
    # rejected WALL/CLAMP hacks.  Replaces the floating `out` containment box.
    # 0 = off (legacy pairwise-overlap-only spreading).
    edensity = float(os.environ.get("ELECTRO_EDENSITY", "0.0"))
    ed_grid = int(os.environ.get("ELECTRO_EDENSITY_GRID", "64"))
    # Canvas utilization target = total_block_area / canvas_area.  GT dies pack to
    # ~0.965; aim a touch higher so the achieved util (always a bit below target,
    # since blocks never perfectly tile) lands near GT and area_gap ~ 0.
    ed_util = float(os.environ.get("ELECTRO_EDENSITY_UTIL", "0.98"))
    
    # Custom optimizations for V_mib and V_boundary
    # ELECTRO_MIB_SHAPE (2026-07-17): re-swept given everything stacked on top
    # since this was first tuned. The end-of-run exact copy (`la.data.copy_`,
    # below) guarantees V_mib=0 regardless of this weight -- it only shapes how
    # much freedom the optimizer has DURING placement before being snapped.
    # Sharp, reproducible, non-monotonic optimum found at 0.05 (weaker guidance
    # than the original 0.1): full-100 Neutral RT (portfolio off, so zero extra
    # runtime) 2.2281 -> 2.1731 (-2.5%); stacks cleanly with ELECTRO_GRP_WEIGHT
    # and ELECTRO_ITERS_PORTFOLIO (combined: 2.1014 -> 2.0559).
    # For Jacobi warm-start, the optimal parameters are slightly shifted compared to Random init.
    # Default to 0.03 for Jacobi, 0.05 for Random.
    default_mib_shape = "0.03" if os.environ.get("ELECTRO_INIT", "random") == "jacobi" else "0.05"
    lam_mib_shape = float(os.environ.get("ELECTRO_MIB_SHAPE", default_mib_shape))
    boundary_l2 = os.environ.get("ELECTRO_BOUNDARY_L2", "0") == "1"
    grouping_l2 = os.environ.get("ELECTRO_GROUPING_L2", "0") == "1"


    a = area_targets[:N].float().to(dev).clamp(min=1e-9)
    cons = constraints[:N].to(dev)
    ncol = cons.shape[1]
    is_fixed = cons[:, 0] != 0
    is_pre = cons[:, 1] != 0
    is_soft = ~(is_fixed | is_pre)
    mib_id = cons[:, 2].long() if ncol > 2 else torch.zeros(N, dtype=torch.long, device=dev)
    clust_id = cons[:, 3].long() if ncol > 3 else torch.zeros(N, dtype=torch.long, device=dev)
    bcode = cons[:, 4].long() if ncol > 4 else torch.zeros(N, dtype=torch.long, device=dev)

    if target_positions is not None and target_positions.numel() > 0:
        tp = target_positions[:N].float().to(dev)
    else:
        tp = torch.full((N, 4), -1.0, device=dev)

    S = float(torch.sqrt(a.sum()).item()) or 1.0
    an = a / (S * S)
    sqrt_an = torch.sqrt(an)   # per-block, so EVERY block keeps its exact area

    tw = (tp[:, 2] / S).clamp(min=0.0)
    th = (tp[:, 3] / S).clamp(min=0.0)
    pre_cx = tp[:, 0] / S + 0.5 * tw
    pre_cy = tp[:, 1] / S + 0.5 * th

    # ---- MIB shape-groups: members share one (la, area) -> identical (w,h) ----
    sg = torch.arange(N, device=dev)
    gmax = int(mib_id.max().item()) if mib_id.numel() else 0
    for g in range(1, gmax + 1):
        mem = (mib_id == g).nonzero().flatten()
        if mem.numel() > 0:
            sg[mem] = mem[0]
    _, inv = torch.unique(sg, return_inverse=True)
    K = int(inv.max().item()) + 1
    cnt_sg = torch.zeros(K, device=dev).index_add_(0, inv, torch.ones(N, device=dev))
    area_sg = torch.zeros(K, device=dev).index_add_(0, inv, an) / cnt_sg.clamp(min=1.0)
    sqrt_area_sg = torch.sqrt(area_sg.clamp(min=1e-12))

    # ---- cluster membership matrix (constant) for grouping centroids ----
    Gc = int(clust_id.max().item()) if clust_id.numel() else 0
    if Gc > 0:
        Mc = torch.zeros(Gc, N, device=dev)
        for g in range(1, Gc + 1):
            mem = (clust_id == g).nonzero().flatten()
            if mem.numel() > 0:
                Mc[g - 1, mem] = 1.0 / mem.numel()
        has_clust = clust_id > 0
        gi = (clust_id[has_clust] - 1)
        n_clust_mem = max(1, int(has_clust.sum().item()))

    Lb = (bcode & 1) > 0
    Rb = (bcode & 2) > 0
    Tb = (bcode & 4) > 0
    Bb = (bcode & 8) > 0

    # ---- edge extraction (moved before init so Jacobi warm-start can use it) ----
    eb = _valid(b2b_connectivity)
    ep = _valid(p2b_connectivity)
    pv = None
    if pins_pos is not None and pins_pos.numel() > 0:
        pv = pins_pos[pins_pos[:, 0] != -1].float().to(dev) / S
    if eb is not None:
        ia = eb[:, 0].long().clamp(0, N - 1)
        ib = eb[:, 1].long().clamp(0, N - 1)
        wb = eb[:, 2].float().to(dev)
    if ep is not None and pv is not None and pv.shape[0] > 0:
        et = ep[:, 0].long().clamp(0, pv.shape[0] - 1)
        ebk = ep[:, 1].long().clamp(0, N - 1)
        wp = ep[:, 2].float().to(dev)
        tx, ty = pv[et, 0], pv[et, 1]
    else:
        ep = None
    total_w = 1.0 + (float(wb.sum()) if eb is not None else 0.0) + (float(wp.sum()) if ep is not None else 0.0)

    # ---- parameters ----
    gen = torch.Generator(device=dev).manual_seed(seed)
    if init_centers is not None:
        # ML warm-start: use the Transformer's predicted (cx, cy) -- given in raw
        # coords, so normalise by S -- as the gradient-descent starting point.
        # seed 0 = the pure prediction; further seeds jitter it for multi-start
        # diversity while staying in the good basin the model points to.
        ic = init_centers.to(dev).float()
        cx = ic[:, 0] / S
        cy = ic[:, 1] / S
        if seed != 0:
            j = float(os.environ.get("ELECTRO_ML_JITTER", "0.15"))
            cx = cx + j * torch.randn(N, generator=gen, device=dev)
            cy = cy + j * torch.randn(N, generator=gen, device=dev)
    else:
        cx = torch.rand(N, generator=gen, device=dev)
        cy = torch.rand(N, generator=gen, device=dev)

    # ---- Jacobi warm-start: graph-layout initialization (2026-07-18) ----
    # When ELECTRO_INIT=jacobi, refine the random (cx,cy) with K rounds of
    # neighbor-averaging on the b2b connectivity graph.  Preplaced blocks act
    # as fixed anchors.  This gives connected blocks a head start near each
    # other, reducing the variance that multi-start (seeds>1) compensates for.
    # Zero extra runtime (same iters=600, same single seed).
    jacobi_init = os.environ.get("ELECTRO_INIT", "random")
    jacobi_rounds = int(os.environ.get("ELECTRO_JACOBI_ROUNDS", "20"))
    jacobi_alpha = float(os.environ.get("ELECTRO_JACOBI_ALPHA", "0.3"))

    if jacobi_init == "jacobi" and eb is not None and init_centers is None:
        with torch.no_grad():
            cx = torch.where(is_pre, pre_cx, cx)
            cy = torch.where(is_pre, pre_cy, cy)
            for _ in range(jacobi_rounds):
                # Weighted sum of neighbor positions via scatter-add
                sum_x = torch.zeros(N, device=dev)
                sum_y = torch.zeros(N, device=dev)
                sum_w = torch.zeros(N, device=dev)
                sum_x.index_add_(0, ia, wb * cx[ib])
                sum_x.index_add_(0, ib, wb * cx[ia])
                sum_y.index_add_(0, ia, wb * cy[ib])
                sum_y.index_add_(0, ib, wb * cy[ia])
                sum_w.index_add_(0, ia, wb)
                sum_w.index_add_(0, ib, wb)
                # Add pin/terminal pull if available
                if ep is not None:
                    sum_x.index_add_(0, ebk, wp * tx)
                    sum_y.index_add_(0, ebk, wp * ty)
                    sum_w.index_add_(0, ebk, wp)
                # Blocks with neighbors: blend toward weighted average
                has_nbr = sum_w > 0
                avg_x = sum_x / sum_w.clamp(min=1e-9)
                avg_y = sum_y / sum_w.clamp(min=1e-9)
                new_cx = torch.where(has_nbr, (1 - jacobi_alpha) * cx + jacobi_alpha * avg_x, cx)
                new_cy = torch.where(has_nbr, (1 - jacobi_alpha) * cy + jacobi_alpha * avg_y, cy)
                # Preplaced blocks: always anchored at their fixed positions
                cx = torch.where(is_pre, pre_cx, new_cx)
                cy = torch.where(is_pre, pre_cy, new_cy)

    cx = torch.where(is_pre, pre_cx, cx).clone().requires_grad_(True)
    cy = torch.where(is_pre, pre_cy, cy).clone().requires_grad_(True)
    la = torch.zeros(K, device=dev, requires_grad=True)   # one log-aspect per shape-group

    AR_CAP = 4.0
    la_cap = float(torch.log(torch.tensor(AR_CAP)))

    # Precompute target log aspect ratio for MIB groups with anchor blocks
    la_target = torch.zeros(K, device=dev)
    has_anchor = torch.zeros(K, dtype=torch.bool, device=dev)
    for k in range(K):
        mems = (inv == k).nonzero().flatten()
        anchors = mems[is_fixed[mems] | is_pre[mems]]
        if anchors.numel() > 0:
            idx = anchors[0]
            la_target[k] = torch.clamp(
                torch.log(tw[idx].clamp(min=1e-6) / th[idx].clamp(min=1e-6)),
                -la_cap, la_cap
            )
            has_anchor[k] = True

    triu = torch.triu_indices(N, N, offset=1, device=dev)
    ti, tj = triu[0], triu[1]

    # ---- eDensity: fixed origin-anchored canvas + DCT Poisson operator ----
    if edensity > 0.0:
        import math as _m
        # Canvas aspect from the pin-terminal bbox (a strong proxy for the GT die
        # outline -- both anchor at origin and their aspect ratios track closely).
        # Fall back to square when there are too few terminals.
        if pv is not None and pv.shape[0] >= 2:
            pwx = float(pv[:, 0].max() - pv[:, 0].min())
            pwy = float(pv[:, 1].max() - pv[:, 1].min())
            aspect = pwx / pwy if pwy > 1e-6 else 1.0
        else:
            aspect = 1.0
        aspect = min(max(aspect, 0.25), 4.0)
        area_c = 1.0 / ed_util            # normalized total block area == 1
        Hc = (area_c / aspect) ** 0.5
        Wc = aspect * Hc
        M = ed_grid
        hx = Wc / M
        hy = Hc / M
        x_edges = torch.arange(M + 1, device=dev).float() * hx   # [M+1]
        y_edges = torch.arange(M + 1, device=dev).float() * hy
        xL, xR = x_edges[:-1], x_edges[1:]                       # [M] bin spans
        yL, yR = y_edges[:-1], y_edges[1:]
        # Orthonormal DCT-II basis  Cb[k,j] = s_k cos(pi k (2j+1)/(2M)).
        kk = torch.arange(M, device=dev).float()
        jj = torch.arange(M, device=dev).float()
        Cb = torch.cos(_m.pi * kk[:, None] * (2 * jj[None, :] + 1) / (2 * M))
        s = torch.full((M,), (2.0 / M) ** 0.5, device=dev)
        s[0] = (1.0 / M) ** 0.5
        Cb = Cb * s[:, None]
        # Discrete-Laplacian eigenvalues for Neumann BC (DCT-II):
        #   lam_k = 2(1 - cos(pi k / M)) / h^2.   Energy = sum a_hat^2 / (lam_x+lam_y)
        # weights low spatial frequencies (large-scale non-uniformity) most -> the
        # global spreading force; DC mode (0,0) carries no energy (zeroed below).
        lam = 2.0 * (1.0 - torch.cos(_m.pi * kk / M))
        denom = lam[:, None] / (hx * hx) + lam[None, :] / (hy * hy)
        denom[0, 0] = 1.0
        inv_bin = 1.0 / (hx * hy)

    def shapes(la_, ascale=1.0):
        # Aspect ratio is shared within a MIB group (la is per shape-group);
        # AREA is always per-block exact (sqrt_an), so the hard area constraint
        # is never violated.  Equal-area MIB members -> identical (w,h); members
        # with unequal target areas keep their own area (V_mib may be > 0, but
        # area stays feasible -- hard constraint beats the soft one).
        la_b = la_.clamp(-la_cap, la_cap)
        # Module-area-growing (ascale<1 early -> 1.0 at output): soft blocks start
        # shrunk so they slip into gaps and the layout packs into a tight outline,
        # then grow to exact area (the fixed-outline rectilinear-soft-module idea).
        # Only soft blocks grow; fixed/preplaced obstacles stay full size.
        sg_scale = ascale ** 0.5
        w_soft = (sqrt_area_sg * torch.exp(0.5 * la_b))[inv] * sg_scale
        h_soft = (sqrt_area_sg * torch.exp(-0.5 * la_b))[inv] * sg_scale
        w = torch.where(is_soft, w_soft, torch.where(is_fixed | is_pre, tw, w_soft))
        h = torch.where(is_soft, h_soft, torch.where(is_fixed | is_pre, th, h_soft))
        return w, h

    opt_name = os.environ.get("ELECTRO_OPT", "adam").lower()
    if opt_name == "nesterov":
        # ePlace-style accelerated gradient (SGD + Nesterov momentum).  Needs a
        # larger lr than Adam since it has no per-parameter scaling.
        opt = torch.optim.SGD([cx, cy, la], lr=lr, momentum=0.9, nesterov=True)
    else:
        opt = torch.optim.Adam([cx, cy, la], lr=lr)

    loss_history = []
    
    # elfPlace rollback + entropy injection (2026-07-18)
    enable_rollback = os.environ.get("ELECTRO_ROLLBACK", "0") == "1"
    rollback_count = 0
    last_checkpoint = None
    ov_history = []
    opt_lbfgs = None

    for it in range(iters):
        opt.zero_grad()
        frac = it / max(1, iters - 1)
        g_wl = wl_smooth * (1.0 - 0.9 * frac)   # anneal the WL smoothing -> 0
        # grow soft-block area from area_grow0 up to 1.0 by frac == grow_end
        area_scale = min(1.0, area_grow0 + (1.0 - area_grow0) * (frac / max(grow_end, 1e-6)))

        w, h = shapes(la, area_scale)
        ecx = torch.where(is_pre, pre_cx, cx)
        ecy = torch.where(is_pre, pre_cy, cy)

        wl = ecx.new_zeros(())
        if eb is not None:
            wl = wl + (wb * (_sabs(ecx[ia] - ecx[ib], g_wl) + _sabs(ecy[ia] - ecy[ib], g_wl))).sum()
        if ep is not None:
            # ext_wl scales the pin/terminal (external) wirelength pull, to test
            # dragging pin-connected blocks closer to their fixed terminals.
            wl = wl + ext_wl * (wp * (_sabs(tx - ecx[ebk], g_wl) + _sabs(ty - ecy[ebk], g_wl))).sum()
        wl = wl / total_w

        dx = (ecx[ti] - ecx[tj]).abs()
        dy = (ecy[ti] - ecy[tj]).abs()
        ov = (torch.relu(0.5 * (w[ti] + w[tj]) - dx) * torch.relu(0.5 * (h[ti] + h[tj]) - dy)).sum()

        left = (ecx - 0.5 * w).min(); right = (ecx + 0.5 * w).max()
        bot = (ecy - 0.5 * h).min(); top = (ecy + 0.5 * h).max()
        bbox = (right - left) * (top - bot)

        # Fixed-outline containment: pull every block edge inside a target square
        # of side L = sqrt(total_area / util) centred on the layout, forcing dense
        # packing (raise util -> shrink the box -> cut area_gap).  Normalised total
        # block area == 1, so L = sqrt(1/util).  (PeF / fixed-outline FP idea.)
        out = ecx.new_zeros(())
        if lam_out > 0.0 and edensity <= 0.0:   # eDensity supersedes the float box
            hL = 0.5 / (target_util ** 0.5)
            gx = ((left + right) * 0.5).detach()
            gy = ((bot + top) * 0.5).detach()
            ex = (torch.relu((ecx + 0.5 * w) - (gx + hL))
                  + torch.relu((gx - hL) - (ecx - 0.5 * w)))
            ey = (torch.relu((ecy + 0.5 * h) - (gy + hL))
                  + torch.relu((gy - hL) - (ecy - 0.5 * h)))
            out = (ex + ey).sum() / N

        # eDensity electrostatic penalty on the fixed origin-anchored canvas.
        # Differentiable bin charge = block-rect / bin-rect overlap area; the
        # density field's energy (solved via DCT, Neumann BC) is minimized when
        # the charge is spread uniformly over [0,Wc]x[0,Hc] -- spreading the
        # blocks to fill the canvas AND confining them inside it.  Preplaced
        # blocks bin at their fixed (ecx,ecy) -> act as static obstacles.
        den = ecx.new_zeros(())
        if edensity > 0.0:
            bl = ecx - 0.5 * w
            br = ecx + 0.5 * w
            bb = ecy - 0.5 * h
            bt = ecy + 0.5 * h
            ox = (torch.minimum(br[:, None], xR[None, :])
                  - torch.maximum(bl[:, None], xL[None, :])).clamp(min=0.0)  # [N,M]
            oy = (torch.minimum(bt[:, None], yR[None, :])
                  - torch.maximum(bb[:, None], yL[None, :])).clamp(min=0.0)  # [N,M]
            rho = (ox.transpose(0, 1) @ oy) * inv_bin   # [M,M] area-fraction density
            rho = rho - rho.mean()
            a_hat = Cb @ rho @ Cb.transpose(0, 1)       # 2-D DCT-II
            den = (a_hat * a_hat / denom).sum()

        # Walls at x=0 and y=0 confining blocks to the first quadrant.
        #  * quadratic (lam_wall): smooth, but the restoring force -> 0 at the
        #    boundary, so it leaves an equilibrium gap (never exactly non-negative).
        #  * linear / L1 (lam_wall_lin): a CONSTANT force the instant a corner goes
        #    negative, vanishing at >=0 -- an *exact* penalty (a finite weight that
        #    exceeds the opposing forces pins blocks exactly on the axis, no gap).
        wall = ecx.new_zeros(())
        if lam_wall > 0.0:
            wall = (torch.relu(0.5 * w - ecx) ** 2
                    + torch.relu(0.5 * h - ecy) ** 2).sum() / N
        wall_lin = ecx.new_zeros(())
        if lam_wall_lin > 0.0:
            wall_lin = (torch.relu(0.5 * w - ecx)
                        + torch.relu(0.5 * h - ecy)).sum() / N

        grp = ecx.new_zeros(())
        if Gc > 0:
            gcx = Mc @ ecx
            gcy = Mc @ ecy
            if grouping_l2:
                grp = (((ecx[has_clust] - gcx[gi]).pow(2)
                        + (ecy[has_clust] - gcy[gi]).pow(2)).sum()) / n_clust_mem
            else:
                grp = (((ecx[has_clust] - gcx[gi]).abs()
                        + (ecy[has_clust] - gcy[gi]).abs()).sum()) / n_clust_mem

        bnd = ecx.new_zeros(())
        xmn, xmx = left.detach(), right.detach()
        ymn, ymx = bot.detach(), top.detach()
            
        if boundary_l2:
            if Lb.any():
                bnd = bnd + ((ecx[Lb] - 0.5 * w[Lb]) - xmn).pow(2).sum()
            if Rb.any():
                bnd = bnd + ((ecx[Rb] + 0.5 * w[Rb]) - xmx).pow(2).sum()
            if Tb.any():
                bnd = bnd + ((ecy[Tb] + 0.5 * h[Tb]) - ymx).pow(2).sum()
            if Bb.any():
                bnd = bnd + ((ecy[Bb] - 0.5 * h[Bb]) - ymn).pow(2).sum()
        else:
            if Lb.any():
                bnd = bnd + ((ecx[Lb] - 0.5 * w[Lb]) - xmn).abs().sum()
            if Rb.any():
                bnd = bnd + ((ecx[Rb] + 0.5 * w[Rb]) - xmx).abs().sum()
            if Tb.any():
                bnd = bnd + ((ecy[Tb] + 0.5 * h[Tb]) - ymx).abs().sum()
            if Bb.any():
                bnd = bnd + ((ecy[Bb] - 0.5 * h[Bb]) - ymn).abs().sum()
        bnd = bnd / N

        mib_shape = ecx.new_zeros(())
        if has_anchor.any():
            mib_shape = ((la - la_target) ** 2)[has_anchor].sum()

        # Keep the layout TIGHT: the legalizer cleans up small residual overlap,
        # so we don't ramp the spreading force so high that blocks over-disperse
        # (which inflates HPWL and bbox area).  A modest final lam_ov leaves a
        # little overlap for the legalizer and keeps wirelength/area low.
        lam_ov = ov0 + (ov1 - ov0) * frac
        lam_bb = bb0 + (bb1 - bb0) * frac
        # ELECTRO_GRP_WEIGHT (2026-07-17): the grouping-centroid pull (below) was a
        # hardcoded constant with no external knob, unlike lam_bnd.  Sweeping it
        # (portfolio off, so this is a zero-extra-runtime change) found a sharp,
        # reproducible, NON-monotonic optimum at 0.4 -- weaker than the original
        # implicit 1.0.  Counter-intuitively the original strength over-pulled
        # cluster members toward one point, distorting HPWL/area more than it
        # saved on V_grouping; full-100 Neutral RT Total Score 2.4071 -> 2.2281
        # (-7.4%), stacks additively with ELECTRO_ITERS_PORTFOLIO (-> 2.1014).
        # Do NOT also crank ELECTRO_BND_WEIGHT up when this is <1.0: the two
        # losses compete for the same gradient budget -- combining each knob's
        # own individual optimum (grp=0.4, bnd=2.0) regressed to 2.3060, worse
        # than either alone. Needs a real joint grid search, not naive stacking.
        # Default to 0.5 for Jacobi init, 0.4 for Random init.
        default_grp_weight = "0.5" if os.environ.get("ELECTRO_INIT", "random") == "jacobi" else "0.4"
        grp_weight = float(os.environ.get("ELECTRO_GRP_WEIGHT", default_grp_weight))
        lam_grp = (0.2 + 1.6 * frac) * grp_weight
        bnd_weight = float(os.environ.get("ELECTRO_BND_WEIGHT", "1.0"))
        lam_bnd = (0.2 + 1.6 * frac) * bnd_weight


        loss = wl + lam_ov * ov + lam_bb * bbox + lam_grp * grp + lam_bnd * bnd + lam_out * out
        if lam_mib_shape > 0.0:
            loss = loss + (lam_mib_shape * (0.2 + 1.6 * frac)) * mib_shape
        if edensity > 0.0:
            loss = loss + edensity * den
        if lam_wall > 0.0:        # ramp the wall up so it firmly confines by the end
            loss = loss + (lam_wall * frac) * wall
        if lam_wall_lin > 0.0:    # constant (not ramped): strong from iter 0 so
            loss = loss + lam_wall_lin * wall_lin   # blocks never escape negative
        loss_history.append(float(loss.item()))
        loss.backward()
        
        enable_lbfgs = os.environ.get("ELECTRO_LBFGS_FINISH", "0") == "1"
        if enable_lbfgs and it >= iters - 50:
            if opt_lbfgs is None:
                opt_lbfgs = torch.optim.LBFGS([cx, cy, la], lr=0.05, max_iter=5, history_size=5, line_search_fn="strong_wolfe")
            
            def evaluate_loss_closure():
                opt_lbfgs.zero_grad()
                w_curr, h_curr = shapes(la, area_scale)
                ecx_curr = torch.where(is_pre, pre_cx, cx)
                ecy_curr = torch.where(is_pre, pre_cy, cy)
                
                wl_curr = ecx_curr.new_zeros(())
                if eb is not None:
                    wl_curr = wl_curr + (wb * (_sabs(ecx_curr[ia] - ecx_curr[ib], g_wl) + _sabs(ecy_curr[ia] - ecy_curr[ib], g_wl))).sum()
                if ep is not None:
                    wl_curr = wl_curr + ext_wl * (wp * (_sabs(tx - ecx_curr[ebk], g_wl) + _sabs(ty - ecy_curr[ebk], g_wl))).sum()
                wl_curr = wl_curr / total_w
                
                dx = (ecx_curr[ti] - ecx_curr[tj]).abs()
                dy = (ecy_curr[ti] - ecy_curr[tj]).abs()
                ov_curr = (torch.relu(0.5 * (w_curr[ti] + w_curr[tj]) - dx) * torch.relu(0.5 * (h_curr[ti] + h_curr[tj]) - dy)).sum()
                
                left_c = (ecx_curr - 0.5 * w_curr).min(); right_c = (ecx_curr + 0.5 * w_curr).max()
                bot_c = (ecy_curr - 0.5 * h_curr).min(); top_c = (ecy_curr + 0.5 * h_curr).max()
                bbox_curr = (right_c - left_c) * (top_c - bot_c)
                
                out_curr = ecx_curr.new_zeros(())
                if lam_out > 0.0 and edensity <= 0.0:
                    hL = 0.5 / (target_util ** 0.5)
                    gx = ((left_c + right_c) * 0.5).detach()
                    gy = ((bot_c + top_c) * 0.5).detach()
                    ex = (torch.relu((ecx_curr + 0.5 * w_curr) - (gx + hL))
                          + torch.relu((gx - hL) - (ecx_curr - 0.5 * w_curr)))
                    ey = (torch.relu((ecy_curr + 0.5 * h_curr) - (gy + hL))
                          + torch.relu((gy - hL) - (ecy_curr - 0.5 * h_curr)))
                    out_curr = (ex + ey).sum() / N
                    
                den_curr = ecx_curr.new_zeros(())
                if edensity > 0.0:
                    bl = ecx_curr - 0.5 * w_curr
                    br = ecx_curr + 0.5 * w_curr
                    bb = ecy_curr - 0.5 * h_curr
                    bt = ecy_curr + 0.5 * h_curr
                    ox_c = (torch.minimum(br[:, None], xR[None, :])
                          - torch.maximum(bl[:, None], xL[None, :])).clamp(min=0.0)
                    oy_c = (torch.minimum(bt[:, None], yR[None, :])
                          - torch.maximum(bb[:, None], yL[None, :])).clamp(min=0.0)
                    rho = (ox_c.transpose(0, 1) @ oy_c) * inv_bin
                    rho = rho - rho.mean()
                    a_hat = Cb @ rho @ Cb.transpose(0, 1)
                    den_curr = (a_hat * a_hat / denom).sum()
                    
                wall_curr = ecx_curr.new_zeros(())
                if lam_wall > 0.0:
                    wall_curr = (torch.relu(0.5 * w_curr - ecx_curr) ** 2
                            + torch.relu(0.5 * h_curr - ecy_curr) ** 2).sum() / N
                wall_lin_curr = ecx_curr.new_zeros(())
                if lam_wall_lin > 0.0:
                    wall_lin_curr = (torch.relu(0.5 * w_curr - ecx_curr)
                                + torch.relu(0.5 * h_curr - ecy_curr)).sum() / N
                                
                grp_curr = ecx_curr.new_zeros(())
                if Gc > 0:
                    gcx = Mc @ ecx_curr
                    gcy = Mc @ ecy_curr
                    if grouping_l2:
                        grp_curr = (((ecx_curr[has_clust] - gcx[gi]).pow(2)
                                + (ecy_curr[has_clust] - gcy[gi]).pow(2)).sum()) / n_clust_mem
                    else:
                        grp_curr = (((ecx_curr[has_clust] - gcx[gi]).abs()
                                + (ecy_curr[has_clust] - gcy[gi]).abs()).sum()) / n_clust_mem
                                
                bnd_curr = ecx_curr.new_zeros(())
                xmn_c, xmx_c = left_c.detach(), right_c.detach()
                ymn_c, ymx_c = bot_c.detach(), top_c.detach()
                if boundary_l2:
                    if Lb.any(): bnd_curr = bnd_curr + ((ecx_curr[Lb] - 0.5 * w_curr[Lb]) - xmn_c).pow(2).sum()
                    if Rb.any(): bnd_curr = bnd_curr + ((ecx_curr[Rb] + 0.5 * w_curr[Rb]) - xmx_c).pow(2).sum()
                    if Tb.any(): bnd_curr = bnd_curr + ((ecy_curr[Tb] + 0.5 * h_curr[Tb]) - ymx_c).pow(2).sum()
                    if Bb.any(): bnd_curr = bnd_curr + ((ecy_curr[Bb] - 0.5 * h_curr[Bb]) - ymn_c).pow(2).sum()
                else:
                    if Lb.any(): bnd_curr = bnd_curr + ((ecx_curr[Lb] - 0.5 * w_curr[Lb]) - xmn_c).abs().sum()
                    if Rb.any(): bnd_curr = bnd_curr + ((ecx_curr[Rb] + 0.5 * w_curr[Rb]) - xmx_c).abs().sum()
                    if Tb.any(): bnd_curr = bnd_curr + ((ecy_curr[Tb] + 0.5 * h_curr[Tb]) - ymx_c).abs().sum()
                    if Bb.any(): bnd_curr = bnd_curr + ((ecy_curr[Bb] - 0.5 * h_curr[Bb]) - ymn_c).abs().sum()
                bnd_curr = bnd_curr / N
                
                mib_shape_curr = ecx_curr.new_zeros(())
                if has_anchor.any():
                    mib_shape_curr = ((la - la_target) ** 2)[has_anchor].sum()
                    
                loss_curr = wl_curr + lam_ov * ov_curr + lam_bb * bbox_curr + lam_grp * grp_curr + lam_bnd * bnd_curr + lam_out * out_curr
                if lam_mib_shape > 0.0:
                    loss_curr = loss_curr + (lam_mib_shape * (0.2 + 1.6 * frac)) * mib_shape_curr
                if edensity > 0.0:
                    loss_curr = loss_curr + edensity * den_curr
                if lam_wall > 0.0:
                    loss_curr = loss_curr + (lam_wall * frac) * wall_curr
                if lam_wall_lin > 0.0:
                    loss_curr = loss_curr + lam_wall_lin * wall_lin_curr
                    
                loss_curr.backward()
                return loss_curr
                
            opt_lbfgs.step(evaluate_loss_closure)
        else:
            opt.step()


        # Checkpoint coordinates and optimizer state every 50 iterations
        if enable_rollback and it % 50 == 0:
            import copy
            last_checkpoint = (
                cx.data.clone(), 
                cy.data.clone(), 
                la.data.clone(), 
                copy.deepcopy(opt.state_dict())
            )
            
        ov_history.append(float(ov.item()))
        
        # Stagnation check and rollback trigger at the end of every 50-iter window
        if enable_rollback and it > 0 and it % 50 == 49 and rollback_count < 2:
            if len(ov_history) >= 100:
                mean_current = sum(ov_history[-50:]) / 50.0
                mean_prev = sum(ov_history[-100:-50]) / 50.0
                # Stagnation criterion: overlap reduction < 0.5%
                # and substantial remaining overlap (> 1% of total block area)
                tot_area = float(an.sum().item())
                if mean_current >= 0.995 * mean_prev and mean_current > 0.01 * tot_area:
                    with torch.no_grad():
                        cx.data.copy_(last_checkpoint[0])
                        cy.data.copy_(last_checkpoint[1])
                        la.data.copy_(last_checkpoint[2])
                        opt.load_state_dict(copy.deepcopy(last_checkpoint[3]))
                        # Inject entropy (noise) proportional to canvas size
                        w_canvas = float(Wc) if (edensity > 0.0) else 1.0
                        h_canvas = float(Hc) if (edensity > 0.0) else 1.0
                        cx.data.add_(torch.randn_like(cx) * 0.01 * w_canvas)
                        cy.data.add_(torch.randn_like(cy) * 0.01 * h_canvas)
                        rollback_count += 1
                        print(f"[debug-rollback] Case rollback triggered at iter {it}! Count: {rollback_count}")
                        # Clear history for this window to prevent immediate re-trigger
                        ov_history = []

        if clamp_canvas and frac >= clamp_start:  # confine to the first quadrant
            with torch.no_grad():
                cx.data.copy_(torch.maximum(cx.data, 0.5 * w.detach()))
                cy.data.copy_(torch.maximum(cy.data, 0.5 * h.detach()))

        if edensity > 0.0:
            # Hard canvas projection: eDensity is a *dispersal* force (it flattens
            # density and, unconstrained, would evacuate blocks to thin them out).
            # The fixed region is what turns "flatten" into "fill at ed_util":
            # clamp every movable center so its rect stays inside [0,Wc]x[0,Hc].
            # Charge is then conserved in the canvas -> mean density == ed_util ->
            # minimizing variance packs uniformly AND guarantees non-negative
            # output (the whole point).  (DREAMPlace fence-region projection.)
            with torch.no_grad():
                hw = 0.5 * w.detach()
                hh = 0.5 * h.detach()
                lox = torch.minimum(hw, torch.full_like(hw, 0.5 * Wc))
                hix = torch.maximum(Wc - hw, torch.full_like(hw, 0.5 * Wc))
                loy = torch.minimum(hh, torch.full_like(hh, 0.5 * Hc))
                hiy = torch.maximum(Hc - hh, torch.full_like(hh, 0.5 * Hc))
                cx.data.copy_(torch.clamp(cx.data, lox, hix))
                cy.data.copy_(torch.clamp(cy.data, loy, hiy))

    with torch.no_grad():
        if has_anchor.any():
            la.data.copy_(torch.where(has_anchor, la_target, la.data))
        w, h = shapes(la)
        ecx = torch.where(is_pre, pre_cx, cx)
        ecy = torch.where(is_pre, pre_cy, cy)
        x = (ecx - 0.5 * w) * S
        y = (ecy - 0.5 * h) * S
        W = w * S
        H = h * S
        # Output EXACT target geometry for locked blocks (avoids float round-trip
        # drift through the /S*S normalisation, which could shift a pinned
        # preplaced block by ~1e-5 and create a micro-overlap the contest flags).
        ispre_l = is_pre.cpu().tolist()
        isfix_l = is_fixed.cpu().tolist()
        tpl = tp.cpu().tolist()
        out = []
        for i in range(N):
            if ispre_l[i]:
                out.append((tpl[i][0], tpl[i][1], tpl[i][2], tpl[i][3]))
            elif isfix_l[i]:
                out.append((float(x[i]), float(y[i]), tpl[i][2], tpl[i][3]))
            else:
                out.append((float(x[i]), float(y[i]), float(W[i]), float(H[i])))
        diag = _measure(x, y, W, H, eb, ep,
                        (tx * S, ty * S) if ep is not None else None,
                        ia if eb is not None else None, ib if eb is not None else None,
                        wb if eb is not None else None,
                        ebk if ep is not None else None, wp if ep is not None else None, a)
        diag["loss_history"] = loss_history
        needs_ext = False
        if len(loss_history) >= 50:
            rel_dec = (loss_history[-50] - loss_history[-1]) / max(1e-9, abs(loss_history[-50]))
            thresh = float(os.environ.get("ELECTRO_ADAPTIVE_THRESH", "0.005"))
            if abs(rel_dec) > thresh:
                needs_ext = True
        diag["needs_extension"] = needs_ext
    return out, diag


@torch.no_grad()
def _measure(x, y, W, H, eb, ep, term_xy, ia, ib, wb, ebk, wp, a):
    cx = x + 0.5 * W
    cy = y + 0.5 * H
    hpwl = 0.0
    if eb is not None:
        hpwl += float((wb * ((cx[ia] - cx[ib]).abs() + (cy[ia] - cy[ib]).abs())).sum())
    if ep is not None:
        tx, ty = term_xy
        hpwl += float((wp * ((tx - cx[ebk]).abs() + (ty - cy[ebk]).abs())).sum())
    N = x.shape[0]
    tri = torch.triu_indices(N, N, offset=1)
    i, j = tri[0], tri[1]
    ox = torch.relu(0.5 * (W[i] + W[j]) - (cx[i] - cx[j]).abs())
    oy = torch.relu(0.5 * (H[i] + H[j]) - (cy[i] - cy[j]).abs())
    total_area = float(a.sum())
    bbox = float((x + W).max() - x.min()) * float((y + H).max() - y.min())
    return {
        "hpwl": hpwl,
        "overlap_pct": 100.0 * float((ox * oy).sum()) / max(total_area, 1e-9),
        "bbox_area": bbox,
        "total_block_area": total_area,
        "bbox_util_pct": 100.0 * total_area / max(bbox, 1e-9),
    }
