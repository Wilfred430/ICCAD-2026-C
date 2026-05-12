// btree.cpp -- B*-tree topology operations.
#include "btree.hpp"
#include <random>
#include <cassert>
#include <algorithm>
#include <stack>
#include <unordered_set>

namespace fp {

void BTree::init(int n_blocks) {
    nodes.assign(n_blocks, BNode{});
    x.assign(n_blocks, 0.0);
    y.assign(n_blocks, 0.0);
    w.assign(n_blocks, 0.0);
    h.assign(n_blocks, 0.0);
    root = NO_NODE;
}

void BTree::build_default() {
    const int n = static_cast<int>(nodes.size());
    if (n == 0) { root = NO_NODE; return; }
    root = 0;
    nodes[0] = BNode{};                 // no parent
    for (int i = 1; i < n; ++i) {
        nodes[i] = BNode{};
        nodes[i].parent = i - 1;
        nodes[i - 1].lc = i;            // left-spine: each block sits to the right of the previous one
    }
}

void BTree::build_random(uint64_t seed) {
    const int n = static_cast<int>(nodes.size());
    if (n == 0) { root = NO_NODE; return; }
    std::mt19937_64 rng(seed);
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::shuffle(order.begin(), order.end(), rng);

    root = order[0];
    nodes.assign(n, BNode{});
    // Insert each subsequent block as a child of a uniformly random existing node,
    // picking the left or right slot if free; if both occupied, descend.
    for (int k = 1; k < n; ++k) {
        int v = order[k];
        int u = order[std::uniform_int_distribution<int>(0, k - 1)(rng)];
        // Walk down until we find a free slot.
        while (true) {
            bool go_left = std::bernoulli_distribution(0.5)(rng);
            int& slot = go_left ? nodes[u].lc : nodes[u].rc;
            if (slot == NO_NODE) {
                slot = v;
                nodes[v].parent = u;
                break;
            }
            u = slot;
        }
    }
}

void BTree::detach(int v) {
    int p = nodes[v].parent;
    if (p == NO_NODE) return;          // detaching the root is a no-op here
    if (nodes[p].lc == v) nodes[p].lc = NO_NODE;
    if (nodes[p].rc == v) nodes[p].rc = NO_NODE;
    nodes[v].parent = NO_NODE;
}

void BTree::insert_left(int u, int v) {
    int prev = nodes[u].lc;
    nodes[u].lc = v;
    nodes[v].parent = u;
    // Graft prev onto v -- arbitrarily put it in v's left slot (if free)
    // or right slot, descending if both occupied.  This keeps all blocks
    // reachable.
    if (prev != NO_NODE) {
        // place prev into the deepest free slot of v's subtree
        int cur = v;
        while (true) {
            if (nodes[cur].lc == NO_NODE) { nodes[cur].lc = prev; nodes[prev].parent = cur; break; }
            if (nodes[cur].rc == NO_NODE) { nodes[cur].rc = prev; nodes[prev].parent = cur; break; }
            cur = nodes[cur].lc;
        }
    }
}

void BTree::insert_right(int u, int v) {
    int prev = nodes[u].rc;
    nodes[u].rc = v;
    nodes[v].parent = u;
    if (prev != NO_NODE) {
        int cur = v;
        while (true) {
            if (nodes[cur].lc == NO_NODE) { nodes[cur].lc = prev; nodes[prev].parent = cur; break; }
            if (nodes[cur].rc == NO_NODE) { nodes[cur].rc = prev; nodes[prev].parent = cur; break; }
            cur = nodes[cur].lc;
        }
    }
}

void BTree::op_swap(int a, int b) {
    if (a == b) return;
    // Swap the *labels*, i.e. swap their geometry and constraint indices but
    // keep them in their tree positions.  In practice the cleanest way is to
    // swap the .lc/.rc slots that point at a or b in their parents, swap
    // their children pointers, and swap the parent pointer of their children.
    //
    // Simpler: swap a <-> b in every reference (parent's child slot, and the
    // children's parent slot), then swap nodes[a] and nodes[b]'s topology
    // fields.
    BNode na = nodes[a], nb = nodes[b];

    auto retarget_child = [&](int parent, int from, int to) {
        if (parent == NO_NODE) return;
        if (nodes[parent].lc == from) nodes[parent].lc = to;
        else if (nodes[parent].rc == from) nodes[parent].rc = to;
    };

    // Handle the case where a and b are parent/child of each other.
    // After swap, identities switch -- reroute carefully.
    if (na.parent == b || nb.parent == a) {
        int parent_id = (na.parent == b) ? b : a;
        int child_id  = (na.parent == b) ? a : b;
        // Swap structure with parent-child in mind.
        BNode np = nodes[parent_id];
        BNode nc = nodes[child_id];
        bool child_was_left = (np.lc == child_id);

        // grandparent pointer
        retarget_child(np.parent, parent_id, child_id);
        // Reassign topology
        nodes[child_id].parent = np.parent;
        if (child_was_left) {
            nodes[child_id].lc = parent_id;
            nodes[child_id].rc = np.rc;
        } else {
            nodes[child_id].lc = np.lc;
            nodes[child_id].rc = parent_id;
        }
        nodes[parent_id].parent = child_id;
        nodes[parent_id].lc = nc.lc;
        nodes[parent_id].rc = nc.rc;
        // Reparent the children we just shuffled.
        if (nodes[child_id].lc != NO_NODE && nodes[child_id].lc != parent_id) nodes[nodes[child_id].lc].parent = child_id;
        if (nodes[child_id].rc != NO_NODE && nodes[child_id].rc != parent_id) nodes[nodes[child_id].rc].parent = child_id;
        if (nodes[parent_id].lc != NO_NODE) nodes[nodes[parent_id].lc].parent = parent_id;
        if (nodes[parent_id].rc != NO_NODE) nodes[nodes[parent_id].rc].parent = parent_id;
        if (root == parent_id) root = child_id;
        return;
    }

    // General case: a and b are unrelated.
    retarget_child(na.parent, a, b);
    retarget_child(nb.parent, b, a);

    std::swap(nodes[a].parent, nodes[b].parent);
    std::swap(nodes[a].lc, nodes[b].lc);
    std::swap(nodes[a].rc, nodes[b].rc);
    if (nodes[a].lc != NO_NODE) nodes[nodes[a].lc].parent = a;
    if (nodes[a].rc != NO_NODE) nodes[nodes[a].rc].parent = a;
    if (nodes[b].lc != NO_NODE) nodes[nodes[b].lc].parent = b;
    if (nodes[b].rc != NO_NODE) nodes[nodes[b].rc].parent = b;
    if (root == a) root = b;
    else if (root == b) root = a;
}

void BTree::op_rotate(int v) {
    std::swap(w[v], h[v]);
}

bool BTree::op_move(int v, int u, bool as_left) {
    // Disallow moving v to be its own descendant -- check by walking up from u.
    if (v == u) return false;
    if (v == root) return false;       // we don't move the root via this primitive
    int cur = u;
    while (cur != NO_NODE) {
        if (cur == v) return false;     // u is in v's subtree
        cur = nodes[cur].parent;
    }
    detach(v);
    if (as_left) insert_left(u, v);
    else         insert_right(u, v);
    return true;
}

bool BTree::validate() const {
    const int n = static_cast<int>(nodes.size());
    if (n == 0) return root == NO_NODE;
    if (root < 0 || root >= n) return false;
    if (nodes[root].parent != NO_NODE) return false;
    std::vector<int> seen(n, 0);
    std::stack<int> st; st.push(root);
    while (!st.empty()) {
        int u = st.top(); st.pop();
        if (u < 0 || u >= n) return false;
        if (seen[u]) return false;
        seen[u] = 1;
        if (nodes[u].lc != NO_NODE) {
            if (nodes[nodes[u].lc].parent != u) return false;
            st.push(nodes[u].lc);
        }
        if (nodes[u].rc != NO_NODE) {
            if (nodes[nodes[u].rc].parent != u) return false;
            st.push(nodes[u].rc);
        }
    }
    for (int i = 0; i < n; ++i) if (!seen[i]) return false;
    return true;
}

void BTree::copy_from(const BTree& o) {
    nodes = o.nodes;
    root  = o.root;
    x = o.x; y = o.y; w = o.w; h = o.h;
}

} // namespace fp
