/// @file recurrent_plan.cpp
/// @brief build_plan — the tile grid, the chunk and the quantized scratch, from
///        the descriptor shapes alone.
///
/// Every number a kernel reads back is decided here: the interleaved gate width
/// per phase, the tile grid solve_tiles picks, the recurrent k-block, the input
/// projection's chunk and column tile, and — on the int8 route — the per-worker
/// scratch slots and their sizes. A plan and a kernel that disagree is exactly
/// the bug this file exists to prevent.

#include "backend/cpu/rnn/recurrent_internal.h"

namespace nnops::backend::cpu {

namespace {

int64_t gcd64(int64_t a, int64_t b) noexcept {
    while (b != 0) { const int64_t t = a % b; a = b; b = t; }
    return a;
}

int64_t lcm64(int64_t a, int64_t b) noexcept {
    return (a / gcd64(a, b)) * b;
}

/// The input projection's column tile. int8 has no prepacked B, so the panels
/// are addressed per k-block slab; a ~512-wide column tile keeps one task's
/// int32 accumulator small (and is a whole number of nr-wide panels, which is
/// what lets the kernel address a column block as a panel index — so it is
/// rounded down to one: 512 is 32 panels at nr 16, but not at nr 12).
constexpr int64_t kQ8ProjNc = (512 / NR_MAX_I8) * NR_MAX_I8;

/// A tile smaller than this does less MMA work than the dispatch around it
/// costs, and a step's grid is dispatched once per phase per timestep — so the
/// floor is what keeps a large team from being slower than a small one on the
/// shapes where B and the phase width together do not amount to much.
constexpr int64_t kMinTileMacs = int64_t{1} << 16;

/// Per-timestep tile grid: split B and the phase width so the padded cell count
/// per busy worker is smallest. A single column block takes the whole width even
/// when that is not a multiple of the grid; every other block starts on it.
void solve_tiles(int64_t rows, int64_t cols, int64_t k, int64_t n_align,
                 int64_t threads, PhaseGeom& g)
{
    threads = std::max<int64_t>(threads, 1);
    g.m_blocks = 1; g.n_blocks = 1; g.mc = rows; g.nc = cols;

    const int64_t max_tasks =
        std::max<int64_t>(1, rows * cols * k / kMinTileMacs);

    int64_t best_pad = -1, best_used = 1;
    const int64_t max_nb = std::max<int64_t>(
        1, std::min<int64_t>(std::min<int64_t>(threads, max_tasks), cols / n_align));

    for (int64_t nb = 1; nb <= max_nb; ++nb) {
        int64_t nc_t = cols;
        if (nb > 1) {
            const int64_t share = (cols + nb - 1) / nb;
            nc_t = std::min<int64_t>(
                ((share + n_align - 1) / n_align) * n_align, cols);
        }
        const int64_t nblk = (nc_t >= cols) ? 1 : (cols + nc_t - 1) / nc_t;

        const int64_t mb = std::max<int64_t>(
            1, std::min<int64_t>(std::min<int64_t>(threads / nblk, rows),
                                 max_tasks / nblk));
        const int64_t mc_t = (rows + mb - 1) / mb;
        const int64_t mblk = (rows + mc_t - 1) / mc_t;

        const int64_t used = std::min<int64_t>(mblk * nblk, threads);
        const int64_t pad = (mblk * mc_t) * (nblk * nc_t);

        if (best_pad < 0 || pad * best_used < best_pad * used) {
            best_pad = pad; best_used = used;
            g.m_blocks = mblk; g.n_blocks = nblk; g.mc = mc_t; g.nc = nc_t;
        }
    }
}

}  // namespace

Status build_plan(const RecurrentParams& p, const ComputeContext& ctx, RecPlan& plan)
{
    const int64_t threads = std::max<int64_t>(ctx.cpu.thread_count(), 1);
    const int64_t esz = static_cast<int64_t>(data_type_size(p.dtype));
    const int64_t H = p.hidden;
    plan.q8 = p.weight_dtype == DataType::s8;
    const int64_t nr = plan.q8 ? NR_MAX_I8 : nr_max_dt(p.dtype);
    const int64_t kc_cap =
        plan.q8 ? kQ8Kc
                : (p.dtype == DataType::f16  ? KC_CAP_F16
                 : p.dtype == DataType::bf16 ? KC_CAP_BF16
                                             : KC_CAP_F32);

    const bool lbr = p.mode == RecurrentMode::Gru && p.linear_before_reset;
    plan.nphases = num_phases(p.mode, lbr);
    plan.hp = (H + 7) / 8 * 8;
    plan.total_cols = 0;

    for (int64_t ph = 0; ph < plan.nphases; ++ph) {
        PhaseGeom& g = plan.phase[ph];
        g.kind = phase_kind(p.mode, lbr, ph);
        g.gates = phase_gates(g.kind);
        g.q = 8 * g.gates;
        g.cols = ((H + 7) / 8) * g.q;
        g.n_align = lcm64(g.q, nr);
        if (!plan.q8) {
            g.rgeo = matmul_packed_b_geom(g.cols, H, p.dtype);
            if (!g.rgeo.ok()) { return Status::invalid_argument; }
        }
        plan.col_base[ph] = plan.total_cols;
        plan.total_cols += g.cols;
        solve_tiles(p.batch, g.cols, std::min(H, kc_cap), g.n_align, threads, g);
    }

    if (!plan.q8) {
        plan.wgeo = matmul_packed_b_geom(plan.total_cols, p.input, p.dtype);
        if (!plan.wgeo.ok()) { return Status::invalid_argument; }
    }

    plan.kc = std::max<int64_t>(1, std::min<int64_t>(H, kc_cap));
    // bf16's B panels are k-group major (four k rows per 64-element group) and
    // a k block is addressed at its first group, so every block but the last has
    // to start on one. The caps are multiples of four already; this rounds a
    // short H, whose single block then starts at zero anyway.
    if (p.dtype == DataType::bf16) {
        plan.kc = (plan.kc + 3) & ~int64_t{3};
    }

    // Chunk the input projection so its buffer stays inside a shared L3.
    const int64_t budget = int64_t{8} << 20;
    plan.chunk = std::max<int64_t>(
        1, budget / (std::max<int64_t>(p.batch, 1) * plan.total_cols * esz));
    plan.chunk = std::min(plan.chunk, std::max<int64_t>(p.seq, 1));

    if (plan.q8) {
        // The projection's m-tile, by the same rule project_chunk uses: start
        // wide, then narrow until the grid is deep enough for the team.
        const int64_t rows = plan.chunk * std::max<int64_t>(p.batch, 1);
        const int64_t threads_n = std::max<int64_t>(threads, 1);
        int64_t mc = 64;
        for (;;) {
            const int64_t mb = (rows + mc - 1) / mc;
            if (mb >= threads_n || mc <= 8) { break; }
            mc -= 8;
        }
        plan.proj_mc = std::min<int64_t>(mc, std::max<int64_t>(rows, 1));
        // A column tile keeps one task's int32 accumulator bounded; the panels
        // are addressed per k-block slab, so any nr multiple is addressable.
        plan.proj_nc = std::min<int64_t>(plan.total_cols, kQ8ProjNc);

        // Scratch slots, by the library's rule: thread ids bound the slots when
        // the grid hands one worker several items, otherwise every task owns
        // its slice. Sized for the largest tile any phase and the projection
        // can produce, since the phases run one after another.
        const int64_t proj_n_blocks = (plan.total_cols + plan.proj_nc - 1) / plan.proj_nc;
        int64_t tasks_max = (rows + plan.proj_mc - 1) / plan.proj_mc * proj_n_blocks;
        for (int64_t ph = 0; ph < plan.nphases; ++ph) {
            tasks_max = std::max<int64_t>(
                tasks_max, plan.phase[ph].m_blocks * plan.phase[ph].n_blocks);
        }
        const int64_t workers = std::max<int64_t>(ctx.cpu.worker_count(), 1);
        plan.slots = (ctx.cpu.thread_slot_count() > 0 && workers < tasks_max)
                         ? workers : std::max<int64_t>(tasks_max, 1);
        plan.a_bytes = num_panels_max(static_cast<int>(plan.proj_mc), MR_I8, 3) *
                       geom_ldd_i8_a(std::min<int64_t>(kQ8Kc, std::max<int64_t>(p.input, 1)));
        plan.a_bytes = std::max<int64_t>(
            plan.a_bytes, num_panels_max(static_cast<int>(p.batch), MR_I8, 3) *
                          geom_ldd_i8_a(std::min<int64_t>(kQ8Kc, H)));
        plan.c_elems = plan.proj_mc * plan.proj_nc;
        plan.s_elems = plan.proj_mc;
        for (int64_t ph = 0; ph < plan.nphases; ++ph) {
            const PhaseGeom& g = plan.phase[ph];
            plan.c_elems = std::max<int64_t>(plan.c_elems, g.mc * g.nc);
            plan.s_elems = std::max<int64_t>(plan.s_elems, g.mc);
        }
    }
    return Status::ok;
}

}  // namespace nnops::backend::cpu
