// Round 031 full-cycle native root.
//
// This is a separate, additive C++20 implementation.  It keeps the original
// 8 MiB physical layout (256 heads * 128 bytes + 32,640 pair cells * 256
// bytes) but exposes the complete C(255,2) = 32,385 root-star cycle basis,
// instead of only the 16x16 = 256 triangle window used by native_cycle.cpp.
//
// Layout
// ------
//   heads[v]         128 bytes = 32 floats.  words [0,16) are the per-class
//                    root-star gauge coordinates gamma(v,c); words [16,32)
//                    are reserved zero identity words.
//   cells[cell(i,j)] 256 bytes = two 128-byte lanes.
//                    lane f words [0,16)  = cycle coefficient a(i,j,c)
//                    lane f words [16,32) = raw pair lane
//                                           x(i,j,c) = a + gamma(j,c)
//                                                        - gamma(i,c)
//                    lane r words [0,16)  = -a(i,j,c)
//                    lane r words [16,32) = -x(i,j,c)
//
// Cycle ids are the lexicographic root-star basis over heads 1..255:
//   id(i,j) = (i-1)*(509-(i-1))/2 + (j-i-1),  1 <= i < j <= 255.
// Root pair cells cell(0,j), 1 <= j <= 255, are kept exactly zero: in the
// root-star basis the root edges are gauge coordinates, stored in heads, not
// in pair cells.
//
// Gauge invariance: a gauge update changes gamma and therefore the raw lanes
// x(i,j,c), but never the coefficient words a(i,j,c).  Recomputing the
// root-star circulation gamma(i,c) + x(i,j,c) - gamma(j,c) recovers a.
//
// Build:
//   c++ -std=c++20 -O3 -shared -fPIC native_full_cycle.cpp -o libnative_full_cycle.so
//
// C ABI (all pointers are handles returned by nan_full_new):
//   void* nan_full_new(int mode);       // 0 or 32385 accepted; 256 refused
//   void  nan_full_free(void*);
//   int   nan_full_count(void*);        // always 32,385
//   int   nan_full_write(void*, int id, int channel, float value);
//   float nan_full_read(void*, int id, int channel); // NaN on bad arguments
//   int   nan_full_read_all(void*, float* out, int n); // n == 32385*16
//   int   nan_full_predict(void*, const float* features16, float* out16);
//   double nan_full_train_epoch(void*, const float* samples, const int* labels,
//                               int n, float lr, unsigned seed);
//   int   nan_full_gauge(void*, unsigned seed, float sigma);
//   int   nan_full_audit(void*);
//   int   nan_full_snapshot(void*, const char* path);
//   int   nan_full_restore(void*, const char* path);
// Extra, for independent mapping checks:
//   int   nan_full_cycle_id(int i, int j);          // -1 if not a cycle pair
//   int   nan_full_cycle_pair(int id, int* i, int* j);
//   int   nan_full_bytes(void*);                    // 8,388,608
//   int   nan_full_channels(void*);                 // 16
//   int   nan_full_feature_dim(void*);              // 16

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace {

constexpr int N = 256;          // heads / graph vertices
constexpr int CLASSES = 16;     // class channels per cycle
constexpr int FEATURES = 16;    // public feature dimension
constexpr size_t PAIR_CELLS = 32'640;
constexpr int FULL_CYCLES = 32'385;
constexpr size_t ROOT_BYTES = 8'388'608;

static_assert(FULL_CYCLES == (N - 1) * (N - 2) / 2);
static_assert(PAIR_CELLS == static_cast<size_t>(N) * (N - 1) / 2);
static_assert(N * 128 + PAIR_CELLS * 256 == ROOT_BYTES);

struct alignas(64) Head {
    float g[32];
};

struct alignas(64) Pair {
    float f[32];
    float r[32];
};

struct alignas(64) Memory {
    Head heads[N];
    Pair cells[PAIR_CELLS];
};

static_assert(sizeof(Head) == 128);
static_assert(sizeof(Pair) == 256);
static_assert(sizeof(Memory) == ROOT_BYTES);
static_assert(offsetof(Memory, cells) == static_cast<size_t>(N) * 128);

// Base offset of row i in the full N-node triangular pair table.
constexpr int row_base(int i) {
    return i * (2 * N - 1 - i) / 2;
}

// Dense cycle rank among heads 1..N-1, lexicographic by (i,j).
constexpr int rank_base(int a) {
    // a = i-1, and rank_base(a) = a*(N-1) - a*(a+1)/2.
    return a * (2 * (N - 1) - a - 1) / 2;
}

int cycle_id(int i, int j) {
    if (i > j) std::swap(i, j);
    if (i < 1 || i >= N - 1 || j <= i || j >= N) return -1;
    const int a = i - 1;
    const int b = j - 1;
    return rank_base(a) + (b - a - 1);
}

bool cycle_pair(int id, int& i, int& j) {
    if (id < 0 || id >= FULL_CYCLES) return false;
    // Largest a with rank_base(a) <= id; a lives in [0, N-3].
    int lo = 0;
    int hi = N - 3;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (rank_base(mid) <= id) lo = mid;
        else hi = mid - 1;
    }
    const int a = lo;
    const int off = id - rank_base(a);
    i = a + 1;
    j = i + 1 + off;
    return i >= 1 && i < j && j < N;
}

int cell_index(int i, int j) {
    if (i > j) std::swap(i, j);
    return row_base(i) + (j - i - 1);
}

// Raw non-root pair lane.  With root gauge coordinate gamma(root)=0 this is
// the canonical gauge presentation x(i,j,c) = gamma(i,c) + a - gamma(j,c).
inline float lane_value(float coefficient, float gamma_i, float gamma_j) {
    return coefficient + gamma_j - gamma_i;
}

inline void set_coefficient(Pair& cell, int channel, float coefficient,
                            float gamma_i, float gamma_j) {
    cell.f[channel] = coefficient;
    cell.r[channel] = -coefficient;
    const float lane = lane_value(coefficient, gamma_i, gamma_j);
    cell.f[CLASSES + channel] = lane;
    cell.r[CLASSES + channel] = -lane;
}

void zero_root_edges(Memory& memory) {
    for (int j = 1; j < N; ++j) {
        Pair& cell = memory.cells[cell_index(0, j)];
        std::memset(&cell, 0, sizeof(cell));
    }
}

void make_potentials(const float* x, float* potentials) {
    // Deterministic expansion of the 16 public features onto 255 non-root
    // heads.  Every cycle is therefore reachable by the readout.
    potentials[0] = 0.0f;
    for (int v = 1; v < N; ++v) {
        const float a = x[(v - 1) % FEATURES];
        const float b = x[(v + 5) % FEATURES];
        const float sign = (v & 1) ? 1.0f : -1.0f;
        potentials[v] = sign * a + 0.5f * b;
    }
}

// Predictions use all 32,385 cycles.  The 1/sqrt(32385) scale keeps the
// aggregate readout O(1) for O(1) coefficients and makes ordinary SGD steps
// effective without requiring a caller-specific learning-rate rescaling.
constexpr float READOUT_SCALE = 1.0f / 180.0f;  // ~= 1 / sqrt(32385)

void compute_logits(const Memory& memory, const float* potentials,
                    float* logits) {
    double acc[CLASSES] = {};
    for (int i = 1; i < N - 1; ++i) {
        const float left = potentials[i];
        const int base = row_base(i);
        for (int j = i + 1; j < N; ++j) {
            const float weight = left * potentials[j] * READOUT_SCALE;
            const Pair& cell = memory.cells[base + (j - i - 1)];
            for (int c = 0; c < CLASSES; ++c) {
                acc[c] += static_cast<double>(weight) * cell.f[c];
            }
        }
    }
    for (int c = 0; c < CLASSES; ++c) {
        logits[c] = static_cast<float>(acc[c]);
    }
}

}  // namespace

extern "C" {

void* nan_full_new(int mode) {
    // The truncated 16x16 / 256-cycle mode is deliberately refused.
    if (mode != 0 && mode != FULL_CYCLES) return nullptr;
    void* raw = std::aligned_alloc(4096, ROOT_BYTES);
    if (!raw) return nullptr;
    std::memset(raw, 0, ROOT_BYTES);
    return raw;
}

void nan_full_free(void* handle) {
    if (!handle) return;
#ifdef __linux__
    munlock(handle, ROOT_BYTES);
#endif
    std::free(handle);
}

int nan_full_count(void* handle) {
    (void)handle;
    return FULL_CYCLES;
}

int nan_full_write(void* handle, int id, int channel, float value) {
    if (!handle || id < 0 || id >= FULL_CYCLES ||
        channel < 0 || channel >= CLASSES) {
        return 0;
    }
    int i = 0;
    int j = 0;
    if (!cycle_pair(id, i, j)) return 0;
    Memory& memory = *static_cast<Memory*>(handle);
    Pair& cell = memory.cells[cell_index(i, j)];
    set_coefficient(cell, channel, value,
                    memory.heads[i].g[channel],
                    memory.heads[j].g[channel]);
    return 1;
}

float nan_full_read(void* handle, int id, int channel) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    if (!handle || id < 0 || id >= FULL_CYCLES ||
        channel < 0 || channel >= CLASSES) {
        return nan;
    }
    int i = 0;
    int j = 0;
    if (!cycle_pair(id, i, j)) return nan;
    const Memory& memory = *static_cast<const Memory*>(handle);
    return memory.cells[cell_index(i, j)].f[channel];
}

int nan_full_read_all(void* handle, float* out, int n) {
    if (!handle || !out || n != FULL_CYCLES * CLASSES) return 0;
    const Memory& memory = *static_cast<const Memory*>(handle);
    size_t at = 0;
    for (int id = 0; id < FULL_CYCLES; ++id) {
        int i = 0;
        int j = 0;
        if (!cycle_pair(id, i, j)) return 0;
        const Pair& cell = memory.cells[cell_index(i, j)];
        for (int c = 0; c < CLASSES; ++c) out[at++] = cell.f[c];
    }
    return 1;
}

int nan_full_predict(void* handle, const float* features, float* out) {
    if (!handle || !features || !out) return 0;
    float potentials[N];
    make_potentials(features, potentials);
    const Memory& memory = *static_cast<const Memory*>(handle);
    compute_logits(memory, potentials, out);
    return 1;
}

double nan_full_train_epoch(void* handle, const float* samples,
                            const int* labels, int n, float lr,
                            unsigned seed) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!handle || !samples || !labels || n <= 0 || !std::isfinite(lr)) {
        return nan;
    }
    for (int t = 0; t < n; ++t) {
        if (labels[t] < 0 || labels[t] >= CLASSES) return nan;
    }

    Memory& memory = *static_cast<Memory*>(handle);
    std::vector<int> order(static_cast<size_t>(n));
    for (int t = 0; t < n; ++t) order[static_cast<size_t>(t)] = t;
    std::mt19937 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);

    double total = 0.0;
    float potentials[N];
    float logits[CLASSES];
    float gradient[CLASSES];

    for (int index : order) {
        const float* x = samples + static_cast<size_t>(index) * FEATURES;
        const int label = labels[index];
        make_potentials(x, potentials);
        compute_logits(memory, potentials, logits);

        float maximum = logits[0];
        for (int c = 1; c < CLASSES; ++c) maximum = std::max(maximum, logits[c]);
        double denominator = 0.0;
        for (int c = 0; c < CLASSES; ++c) {
            gradient[c] = std::exp(logits[c] - maximum);
            denominator += gradient[c];
        }
        total += (maximum + std::log(denominator)) - logits[label];
        for (int c = 0; c < CLASSES; ++c) {
            gradient[c] = static_cast<float>(
                static_cast<double>(gradient[c]) / denominator -
                (c == label ? 1.0 : 0.0));
        }

        // Full-basis sparse-in-feature update: every one of the 32,385
        // coefficients receives its exact root-star gradient contribution.
        for (int i = 1; i < N - 1; ++i) {
            const float left = potentials[i];
            const int base = row_base(i);
            float gamma_i[CLASSES];
            for (int c = 0; c < CLASSES; ++c) gamma_i[c] = memory.heads[i].g[c];
            for (int j = i + 1; j < N; ++j) {
                const float weight = left * potentials[j] * READOUT_SCALE;
                Pair& cell = memory.cells[base + (j - i - 1)];
                for (int c = 0; c < CLASSES; ++c) {
                    const float step = lr * gradient[c] * weight;
                    set_coefficient(cell, c, cell.f[c] - step,
                                    gamma_i[c], memory.heads[j].g[c]);
                }
            }
        }
    }
    return total / static_cast<double>(n);
}

int nan_full_gauge(void* handle, unsigned seed, float sigma) {
    if (!handle || !std::isfinite(sigma) || sigma < 0.0f) return 0;
    Memory& memory = *static_cast<Memory*>(handle);

    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, sigma);

    // Root-star anchoring: gamma(root,c) == 0.  Every non-root head receives
    // its gauge coordinate; coefficients are intentionally untouched.
    for (int c = 0; c < CLASSES; ++c) {
        memory.heads[0].g[c] = 0.0f;
        for (int v = 1; v < N; ++v) {
            memory.heads[v].g[c] += normal(rng);
        }
    }
    for (int v = 0; v < N; ++v) {
        for (int c = CLASSES; c < 32; ++c) memory.heads[v].g[c] = 0.0f;
    }

    // Move only the raw pair-lane presentation.  The invariant coefficient
    // words f[0..15] and their mirrors stay exactly as they were.
    for (int i = 1; i < N - 1; ++i) {
        const int base = row_base(i);
        for (int j = i + 1; j < N; ++j) {
            Pair& cell = memory.cells[base + (j - i - 1)];
            for (int c = 0; c < CLASSES; ++c) {
                const float lane = lane_value(cell.f[c],
                                              memory.heads[i].g[c],
                                              memory.heads[j].g[c]);
                cell.f[CLASSES + c] = lane;
                cell.r[CLASSES + c] = -lane;
            }
        }
    }
    zero_root_edges(memory);
    return 1;
}

int nan_full_audit(void* handle) {
    if (!handle) return 0;
    Memory& memory = *static_cast<Memory*>(handle);

    // Exhaustive forward/inverse cycle-id mapping and one-cell-per-cycle.
    std::vector<unsigned char> cycle_cells(PAIR_CELLS, 0);
    for (int id = 0; id < FULL_CYCLES; ++id) {
        int i = 0;
        int j = 0;
        if (!cycle_pair(id, i, j)) return 0;
        if (i < 1 || i >= j || j >= N) return 0;
        if (cycle_id(i, j) != id) return 0;
        const int cell = cell_index(i, j);
        if (cell < N - 1 || static_cast<size_t>(cell) >= PAIR_CELLS) return 0;
        if (cycle_cells[static_cast<size_t>(cell)]) return 0;
        cycle_cells[static_cast<size_t>(cell)] = 1;
    }
    for (int i = 1; i < N - 1; ++i) {
        for (int j = i + 1; j < N; ++j) {
            if (!cycle_cells[static_cast<size_t>(cell_index(i, j))]) return 0;
        }
    }

    // Every unordered pair has one symmetric cell and a unique cell index.
    std::vector<unsigned char> all_cells(PAIR_CELLS, 0);
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            const int forward = cell_index(i, j);
            const int reverse = cell_index(j, i);
            if (forward != reverse) return 0;
            if (forward < 0 || static_cast<size_t>(forward) >= PAIR_CELLS) {
                return 0;
            }
            if (all_cells[static_cast<size_t>(forward)]) return 0;
            all_cells[static_cast<size_t>(forward)] = 1;
        }
    }

    // Root pair cells are gauge-fixed to zero.
    for (int j = 1; j < N; ++j) {
        const Pair& cell = memory.cells[cell_index(0, j)];
        for (int k = 0; k < 32; ++k) {
            if (!(cell.f[k] == 0.0f) || !(cell.r[k] == 0.0f)) return 0;
        }
    }

    // Head identity: root gauge is zero and reserved head words are zero.
    for (int c = 0; c < CLASSES; ++c) {
        if (!(memory.heads[0].g[c] == 0.0f)) return 0;
    }
    for (int v = 0; v < N; ++v) {
        for (int c = 0; c < 32; ++c) {
            if (!std::isfinite(memory.heads[v].g[c])) return 0;
        }
        for (int c = CLASSES; c < 32; ++c) {
            if (!(memory.heads[v].g[c] == 0.0f)) return 0;
        }
    }

    // Mirror and gauge identities for every non-root pair.
    for (int i = 1; i < N - 1; ++i) {
        for (int j = i + 1; j < N; ++j) {
            const Pair& cell = memory.cells[cell_index(i, j)];
            for (int k = 0; k < 32; ++k) {
                if (!(cell.r[k] == -cell.f[k])) return 0;
            }
            for (int c = 0; c < CLASSES; ++c) {
                const float coefficient = cell.f[c];
                const float gamma_i = memory.heads[i].g[c];
                const float gamma_j = memory.heads[j].g[c];
                const float expected = lane_value(coefficient, gamma_i, gamma_j);
                const float tolerance = 1.0e-5f *
                    (1.0f + std::fabs(coefficient) +
                     std::fabs(gamma_i) + std::fabs(gamma_j));
                if (!(std::fabs(cell.f[CLASSES + c] - expected) <= tolerance)) {
                    return 0;
                }
                if (!(cell.r[CLASSES + c] == -cell.f[CLASSES + c])) return 0;
                const float circulation =
                    gamma_i + cell.f[CLASSES + c] - gamma_j;
                if (!(std::fabs(circulation - coefficient) <= tolerance)) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

int nan_full_snapshot(void* handle, const char* path) {
    if (!handle || !path) return 0;
    FILE* file = std::fopen(path, "wb");
    if (!file) return 0;
    static const char magic[8] = {'N', 'A', 'N', 'F', 'U', 'L', 'L', '1'};
    int ok = std::fwrite(magic, 1, 8, file) == 8 &&
             std::fwrite(handle, 1, ROOT_BYTES, file) == ROOT_BYTES;
    if (std::fclose(file) != 0) ok = 0;
    return ok;
}

int nan_full_restore(void* handle, const char* path) {
    if (!handle || !path) return 0;
    FILE* file = std::fopen(path, "rb");
    if (!file) return 0;
    static const char magic[8] = {'N', 'A', 'N', 'F', 'U', 'L', 'L', '1'};
    char got_magic[8];
    int ok = std::fread(got_magic, 1, 8, file) == 8 &&
             std::memcmp(got_magic, magic, 8) == 0;
    if (ok) {
        // Restore atomically: a short/corrupt file must not partially overwrite
        // the live root.
        std::vector<unsigned char> staged(ROOT_BYTES);
        ok = std::fread(staged.data(), 1, ROOT_BYTES, file) == ROOT_BYTES &&
             std::fgetc(file) == EOF;
        if (ok) std::memcpy(handle, staged.data(), ROOT_BYTES);
    }
    std::fclose(file);
    return ok;
}

// Extra introspection helpers; they do not change the memory layout.
int nan_full_cycle_id(int i, int j) {
    return cycle_id(i, j);
}

int nan_full_cycle_pair(int id, int* i, int* j) {
    if (!i || !j) return 0;
    int left = 0;
    int right = 0;
    if (!cycle_pair(id, left, right)) return 0;
    *i = left;
    *j = right;
    return 1;
}

int nan_full_bytes(void* handle) {
    (void)handle;
    return static_cast<int>(ROOT_BYTES);
}

int nan_full_channels(void* handle) {
    (void)handle;
    return CLASSES;
}

int nan_full_feature_dim(void* handle) {
    (void)handle;
    return FEATURES;
}

}  // extern "C"
