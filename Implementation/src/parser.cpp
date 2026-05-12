// parser.cpp -- Plain-text input / output.
#include "parser.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>

namespace fp {

namespace {
inline bool is_comment_or_blank(const std::string& s) {
    for (char c : s) {
        if (c == '#') return true;
        if (!std::isspace(static_cast<unsigned char>(c))) return false;
    }
    return true;
}

class Lexer {
public:
    explicit Lexer(std::istream& is) : is_(is) {}
    bool next_token(std::string& tok) {
        for (;;) {
            if (cursor_ >= line_.size()) {
                if (!std::getline(is_, line_)) return false;
                ++lineno_;
                cursor_ = 0;
                if (is_comment_or_blank(line_)) continue;
            }
            // skip whitespace
            while (cursor_ < line_.size() && std::isspace(static_cast<unsigned char>(line_[cursor_]))) ++cursor_;
            if (cursor_ >= line_.size()) continue;
            // comment in the middle of a line?
            if (line_[cursor_] == '#') { cursor_ = line_.size(); continue; }
            size_t start = cursor_;
            while (cursor_ < line_.size() && !std::isspace(static_cast<unsigned char>(line_[cursor_])) && line_[cursor_] != '#') ++cursor_;
            tok = line_.substr(start, cursor_ - start);
            return true;
        }
    }
    int line() const { return lineno_; }
private:
    std::istream& is_;
    std::string line_;
    size_t cursor_ = 0;
    int    lineno_ = 0;
};

template <typename T>
bool read_typed(Lexer& L, T& out, std::string* err) {
    std::string tok;
    if (!L.next_token(tok)) { if (err) *err = "unexpected EOF"; return false; }
    std::stringstream ss(tok);
    ss >> out;
    if (!ss) { if (err) *err = "parse error at line " + std::to_string(L.line()) + " token '" + tok + "'"; return false; }
    return true;
}

bool expect_keyword(Lexer& L, const std::string& kw, std::string* err) {
    std::string tok;
    if (!L.next_token(tok)) { if (err) *err = "expected '" + kw + "', got EOF"; return false; }
    if (tok != kw) { if (err) *err = "expected '" + kw + "', got '" + tok + "' at line " + std::to_string(L.line()); return false; }
    return true;
}
} // anonymous

bool load_instance(const std::string& path, FloorplanInstance& inst, std::string* err) {
    std::ifstream is(path);
    if (!is) { if (err) *err = "cannot open " + path; return false; }
    Lexer L(is);

    auto rd = [&](auto& v){ return read_typed(L, v, err); };

    if (!expect_keyword(L, "N_BLOCKS", err) || !rd(inst.n_blocks)) return false;
    if (!expect_keyword(L, "N_TERMINALS", err) || !rd(inst.n_terminals)) return false;

    // optional fields
    std::string tok;
    while (L.next_token(tok)) {
        if (tok == "BASELINE_HPWL")     { if (!rd(inst.baseline_hpwl)) return false; }
        else if (tok == "BASELINE_AREA"){ if (!rd(inst.baseline_area)) return false; }
        else if (tok == "OUTLINE")      { if (!rd(inst.outline_w) || !rd(inst.outline_h)) return false; }
        else if (tok == "TERMINALS") {
            inst.terminals.resize(inst.n_terminals);
            for (int i = 0; i < inst.n_terminals; ++i) {
                int id; Real x, y;
                if (!rd(id) || !rd(x) || !rd(y)) return false;
                if (id < 0 || id >= inst.n_terminals) { if (err) *err = "terminal id out of range"; return false; }
                inst.terminals[id] = {id, x, y};
            }
        }
        else if (tok == "BLOCKS") {
            inst.blocks.assign(inst.n_blocks, {});
            for (int i = 0; i < inst.n_blocks; ++i) {
                int id; int isf, isp, gid, mib; int bedge_int;
                Real area, wi, hi, xi, yi, armin, armax;
                if (!rd(id) || !rd(area) || !rd(isf) || !rd(isp)
                    || !rd(wi) || !rd(hi) || !rd(xi) || !rd(yi)
                    || !rd(mib) || !rd(gid) || !rd(bedge_int)
                    || !rd(armin) || !rd(armax)) return false;
                Block& b = inst.blocks[id];
                b.id = id;
                b.area_target = area;
                b.is_fixed = (isf != 0);
                b.is_preplaced = (isp != 0);
                b.w_input = wi; b.h_input = hi;
                b.x_input = xi; b.y_input = yi;
                b.mib_group = mib;
                b.group_id  = gid;
                b.bedge = static_cast<BoundaryEdge>(bedge_int);
                b.ar_min = (armin > 0) ? armin : 0.25;
                b.ar_max = (armax > 0) ? armax : 4.0;
            }
        }
        else if (tok == "B2B") {
            int m; if (!rd(m)) return false;
            inst.b2b_nets.reserve(m);
            for (int i = 0; i < m; ++i) {
                Net n; if (!rd(n.a) || !rd(n.b) || !rd(n.w)) return false;
                if (n.a == n.b) continue;       // self-loops are no-ops in HPWL
                if (n.a > n.b) std::swap(n.a, n.b);  // canonical order
                inst.b2b_nets.push_back(n);
            }
        }
        else if (tok == "P2B") {
            int m; if (!rd(m)) return false;
            inst.p2b_nets.reserve(m);
            for (int i = 0; i < m; ++i) {
                Net n; if (!rd(n.a) || !rd(n.b) || !rd(n.w)) return false;
                inst.p2b_nets.push_back(n);
            }
        }
        else if (tok == "GROUPS") {
            int P; if (!rd(P)) return false;
            inst.grouping_groups.assign(P, {});
            for (int p = 0; p < P; ++p) {
                int sz; if (!rd(sz)) return false;
                inst.grouping_groups[p].reserve(sz);
                for (int j = 0; j < sz; ++j) {
                    int b; if (!rd(b)) return false;
                    inst.grouping_groups[p].push_back(b);
                    if (b >= 0 && b < (int)inst.blocks.size()) inst.blocks[b].group_id = p;
                }
            }
        }
        else if (tok == "MIB") {
            int Q; if (!rd(Q)) return false;
            inst.mib_groups.assign(Q, {});
            for (int q = 0; q < Q; ++q) {
                int sz; if (!rd(sz)) return false;
                inst.mib_groups[q].reserve(sz);
                for (int j = 0; j < sz; ++j) {
                    int b; if (!rd(b)) return false;
                    inst.mib_groups[q].push_back(b);
                    if (b >= 0 && b < (int)inst.blocks.size()) inst.blocks[b].mib_group = q;
                }
            }
        }
        else if (tok == "END") break;
        else {
            // tolerate unknown tokens (forward compat)
        }
    }

    if ((int)inst.blocks.size() != inst.n_blocks) {
        if (err) *err = "BLOCKS section missing or short";
        return false;
    }
    return true;
}

bool save_solution(const std::string& path, const FloorplanInstance& inst, const BTree& t) {
    std::ofstream os(path);
    if (!os) return false;
    os << "N_BLOCKS " << inst.n_blocks << "\n";
    os << "# id  x  y  w  h\n";
    os << std::fixed;
    os.precision(8);
    for (int i = 0; i < inst.n_blocks; ++i) {
        os << i << " " << t.x[i] << " " << t.y[i] << " " << t.w[i] << " " << t.h[i] << "\n";
    }
    return true;
}

} // namespace fp
