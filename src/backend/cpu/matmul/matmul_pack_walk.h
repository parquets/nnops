#pragma once
/// @file matmul_pack_walk.h
/// @brief The M-panel decomposition both pack/MMA layers walk.
///
/// Shared by matmul_helper.cpp (float) and matmul_helper_i8.cpp (int8): the
/// cascade is the arch's MR ladder either way, and the cost model — panels
/// first, then duplicate rows, then kernel width — does not depend on the
/// element type. The int8 ladder keeps its panels exact (its packers take no
/// live width), which is the one difference and is named at the call site.

#include "matmul_helper.h"

#include <algorithm>

namespace nnops::backend::cpu {

/// One M panel: @p level selects the packer and MMA width, @p rows is how many
/// real rows it carries. A panel narrower than its kernel is legal because the
/// kernels alias every row past the live count onto the last live row, in A and
/// in C: the surplus rows redo a lane of arithmetic and store it back to the
/// same place. No padding, no bounds, no change in the result.
struct MrPanel {
    int level;
    int rows;
};

/// The rows below the level-0 stride, largest level first, then the last two
/// panels folded into the next wider kernel for as long as they both fit in it.
/// The fold is the point: 4+1 becomes one 6-row panel, 1+1+1 one 4-row panel —
/// one B pass instead of three, for two rows of duplicate arithmetic.
///
/// Caller passes fewer than 3 * levels[0] rows, so the tail is at most
/// 3 * levels[0] / levels[1] + 2 panels; MrPanel[10] covers every shipped table.
inline int tail_panels(int count, const int* levels, MrPanel* out)
{
    int n = 0;
    for (int lv = 1; lv < MR_LEVELS && levels[lv] > 0; ++lv) {
        for (const int w = levels[lv]; count >= w; count -= w) {
            out[n++] = {lv, w};
        }
    }
    while (n >= 2) {
        MrPanel& a = out[n - 2];
        const int merged = a.rows + out[n - 1].rows;
        if (a.level == 0 || merged > levels[a.level - 1]) {
            break;
        }
        a.rows = merged;
        --a.level;
        --n;
    }
    return n;
}

/// The M split, by three costs in order: the panel count (each panel re-streams
/// the whole B side, an order of magnitude above any row cost), the rows of
/// duplicate arithmetic the rounding rule pays for, and the widest kernel (a
/// wider panel does more work per row). Ties go to the greedy split.
///
/// Only the level-0 count is searched, and only a couple of steps below greedy:
/// giving up a level-0 panel frees under one stride, which one tail panel now
/// covers outright. Yields 1, 2, 4/3, 4, 6/5, 6, 6+1, 4+4 for 1..8 rows.
template <class Emit> inline void walk_levels(int& i, int count, const int* levels, Emit&& emit)
{
    const int l0 = levels[0];
    const int greedy = count / l0;
    int best_q0 = greedy, best_n = count + 1, best_waste = 0, best_wide = 0;
    MrPanel best_tail[10];
    int best_tail_n = 0;

    for (int q0 = greedy; q0 >= 0 && q0 >= greedy - 2; --q0) {
        MrPanel tail[10];
        const int tn = tail_panels(count - q0 * l0, levels, tail);
        int waste = 0, wide = l0;
        for (int t = 0; t < tn; ++t) {
            waste += levels[tail[t].level] - tail[t].rows;
            wide = std::max(wide, levels[tail[t].level]);
        }
        const int n = q0 + tn;
        if (n < best_n || (n == best_n && waste < best_waste) ||
            (n == best_n && waste == best_waste && wide <= best_wide)) {
            best_q0 = q0;
            best_n = n;
            best_waste = waste;
            best_wide = wide;
            best_tail_n = tn;
            for (int t = 0; t < tn; ++t) {
                best_tail[t] = tail[t];
            }
        }
    }

    for (int n = 0; n < best_q0; ++n) {
        emit(0, l0);
        i += l0;
    }
    for (int t = 0; t < best_tail_n; ++t) {
        emit(best_tail[t].level, best_tail[t].rows);
        i += best_tail[t].rows;
    }
}

/// The same count search for a ladder whose packers take no live width: an int8
/// LHS panel is its rung's full row count, so a panel can never be narrower than
/// the kernel that packed it and the tail stays exact. The count search still
/// applies (8 rows is 4+4, not 6+1+1); only the rounding does not.
template <class Emit>
inline void walk_levels_exact(int& i, int count, const int* levels, Emit&& emit)
{
    const int l0 = levels[0], l1 = levels[1];
    const int greedy = count / l0;
    int best = count + 1, best_q0 = greedy;
    for (int q0 = greedy; q0 >= 0 && q0 >= greedy - 2; --q0) {
        const int rest = count - q0 * l0;
        const int n = q0 + rest / l1 + rest % l1;
        if (n < best) {
            best = n;
            best_q0 = q0;
        }
    }
    for (int n = 0; n < best_q0; ++n) {
        emit(0, l0);
        i += l0;
    }
    for (int lv = 1; lv < MR_LEVELS && levels[lv] > 0; ++lv) {
        for (int w = levels[lv]; i + w <= count; i += w) {
            emit(lv, w);
        }
    }
}

/// Walk the N side in whole nr_max-wide panels, the remainder last.
template <class Pack> inline void for_each_n_panel(int nc, int nr_max, Pack&& pack)
{
    int n = 0;
    for (; n + nr_max <= nc; n += nr_max) {
        pack(nr_max);
    }
    if (n < nc) {
        pack(nc - n);
    }
}

} // namespace nnops::backend::cpu
