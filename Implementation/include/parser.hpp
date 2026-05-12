// parser.hpp -- Input / output for the floorplanner.
//
// Input format (plain text, line-oriented):
//
//   # comment lines start with '#'
//   N_BLOCKS    <k>
//   N_TERMINALS <r>
//   BASELINE_HPWL <h>
//   BASELINE_AREA <a>
//   OUTLINE     <W> <H>            # baseline outline (informational)
//   TERMINALS                       # 'r' rows
//   <id> <x> <y>
//   ...
//   BLOCKS                          # 'k' rows, one per block
//   <id> <area> <is_fixed> <is_preplaced> <w_input> <h_input> <x_input> <y_input>
//        <mib_group> <group_id> <bedge> <ar_min> <ar_max>
//   ...
//   B2B <m_b2b>
//   <i> <j> <w>
//   ...
//   P2B <m_p2b>
//   <ti> <j> <w>
//   ...
//   GROUPS <P>                       # grouping (cluster) groups
//   # for each group: <size> <id_1> ... <id_size>
//   ...
//   MIB <Q>                          # MIB groups
//   ...
//
// Field meanings:
//   bedge: -1 none, 0 left, 1 right, 2 bottom, 3 top, 4 BL, 5 BR, 6 TL, 7 TR
//
// Output format ("solution file"):
//   N_BLOCKS <k>
//   <id> <x> <y> <w> <h>
//
#pragma once
#include "types.hpp"
#include "btree.hpp"
#include <string>

namespace fp {

bool load_instance(const std::string& path, FloorplanInstance& out, std::string* err);
bool save_solution(const std::string& path, const FloorplanInstance& inst,
                   const BTree& t);

} // namespace fp
