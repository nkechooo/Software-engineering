// HIP blocked Floyd–Warshall with optimized kernels; keeps solve(std::vector<int>&, int) interface
#include <iostream>
#include <fstream>
#include <vector>
#include <hip/hip_runtime.h>
#include <algorithm>

#define INF 1073741823
#define BLOCK_SIZE 32

__constant__ int c_BLOCK_SIZE = BLOCK_SIZE;
__constant__ int c_INF = INF;

// Phase 1: update diagonal block (k,k) with shared-memory padding to avoid bank conflicts
__global__ void phase1_opt(int* __restrict__ d, int k, int V) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int B = c_BLOCK_SIZE;
    const int base = k * B;
    const int i = base + ty;
    const int j = base + tx;

    __shared__ int s[BLOCK_SIZE][BLOCK_SIZE + 1];

    if (i < V && j < V) s[ty][tx] = d[i * V + j]; else s[ty][tx] = c_INF;
    __syncthreads();

    const int limit = min(B, V - base);
    #pragma unroll
    for (int t = 0; t < BLOCK_SIZE; ++t) {
        if (t < limit) {
            const int a = s[ty][t];
            const int b = s[t][tx];
            const int sum = a + b;
            if (sum < s[ty][tx]) s[ty][tx] = sum;
        }
        __syncthreads();
    }

    if (i < V && j < V) d[i * V + j] = s[ty][tx];
}

// Phase 2 row blocks: update (k, j) for j != k, reuse pivot and (k, jBlock) tile in shared memory
__global__ void phase2_row_opt(int* __restrict__ d, int k, int V) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int B = c_BLOCK_SIZE;
    const int jBlock = blockIdx.x;
    if (jBlock == k) return;

    __shared__ int piv[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ int colTile[BLOCK_SIZE][BLOCK_SIZE + 1];

    const int base = k * B;
    const int pi = base + ty;
    const int pj = base + tx;
    if (pi < V && pj < V) piv[ty][tx] = d[pi * V + pj]; else piv[ty][tx] = c_INF;
    const int j = jBlock * B + tx;
    // preload (k, jBlock) tile columns into shared memory to avoid per-t iteration global loads
    if (pi < V && j < V) colTile[ty][tx] = d[pi * V + j]; else colTile[ty][tx] = c_INF;
    __syncthreads();

    const int i = base + ty;
    if (i >= V || j >= V) return;

    int curr = d[i * V + j];
    int best = curr;

    const int limit = min(B, V - base);
    #pragma unroll
    for (int t = 0; t < BLOCK_SIZE; ++t) {
        if (t < limit) {
            const int a = piv[ty][t];
            const int b = colTile[t][tx];
            const int sum = a + b;
            if (sum < best) best = sum;
        }
    }

    if (best < curr) d[i * V + j] = best;
}

// Phase 2 column blocks: update (i, k) for i != k, reuse pivot and (iBlock, k) tile in shared memory
__global__ void phase2_col_opt(int* __restrict__ d, int k, int V) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int B = c_BLOCK_SIZE;
    const int iBlock = blockIdx.x;
    if (iBlock == k) return;

    __shared__ int piv[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ int rowTile[BLOCK_SIZE][BLOCK_SIZE + 1];

    const int base = k * B;
    const int pi = base + ty;
    const int pj = base + tx;
    if (pi < V && pj < V) piv[ty][tx] = d[pi * V + pj]; else piv[ty][tx] = c_INF;
    const int i = iBlock * B + ty;
    // preload (iBlock, k) tile rows into shared memory to avoid per-t iteration global loads
    if (i < V && pj < V) rowTile[ty][tx] = d[i * V + pj]; else rowTile[ty][tx] = c_INF;
    __syncthreads();

    const int j = base + tx;
    if (i >= V || j >= V) return;

    int curr = d[i * V + j];
    int best = curr;

    const int limit = min(B, V - base);
    #pragma unroll
    for (int t = 0; t < BLOCK_SIZE; ++t) {
        if (t < limit) {
            const int a = rowTile[ty][t];
            const int b = piv[t][tx];
            const int sum = a + b;
            if (sum < best) best = sum;
        }
    }

    if (best < curr) d[i * V + j] = best;
}

// Phase 3: update remaining blocks (i, j) with i != k, j != k, caching row/col tiles
__global__ void phase3_opt(int* __restrict__ d, int k, int V) {
    const int bi = blockIdx.x;
    const int bj = blockIdx.y;
    if (bi == k || bj == k) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int B = c_BLOCK_SIZE;

    const int i = bi * B + ty;
    const int j = bj * B + tx;
    const int base = k * B;

    __shared__ int rowTile[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ int colTile[BLOCK_SIZE][BLOCK_SIZE + 1];

    // load row tile (i, k)
    const int r_i = i;
    const int r_j = base + tx;
    if (r_i < V && r_j < V) rowTile[ty][tx] = d[r_i * V + r_j]; else rowTile[ty][tx] = c_INF;

    // load col tile (k, j)
    const int c_i = base + ty;
    const int c_j = j;
    if (c_i < V && c_j < V) colTile[ty][tx] = d[c_i * V + c_j]; else colTile[ty][tx] = c_INF;
    __syncthreads();

    if (i >= V || j >= V) return;
    int curr = d[i * V + j];
    int best = curr;

    const int limit = min(B, V - base);
    #pragma unroll
    for (int t = 0; t < BLOCK_SIZE; ++t) {
        if (t < limit) {
            const int a = rowTile[ty][t];
            const int b = colTile[t][tx];
            const int sum = a + b;
            if (sum < best) best = sum;
        }
    }

    if (best < curr) d[i * V + j] = best;
}

static inline void hipCheck(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        fprintf(stderr, "%s: %s\n", msg, hipGetErrorString(err));
        abort();
    }
}

// main solve entry; interface must not change
void solve(std::vector<int>& h_dist, int V) {
    const int B = BLOCK_SIZE;
    const int nBlocks = (V + B - 1) / B;

    int* d_dist = nullptr;
    size_t bytes = static_cast<size_t>(V) * static_cast<size_t>(V) * sizeof(int);
    hipCheck(hipMalloc(&d_dist, bytes), "hipMalloc d_dist");

    hipStream_t stream0, streamRow, streamCol;
    hipCheck(hipStreamCreate(&stream0), "hipStreamCreate 0");
    hipCheck(hipStreamCreate(&streamRow), "hipStreamCreate row");
    hipCheck(hipStreamCreate(&streamCol), "hipStreamCreate col");

    hipEvent_t evtP1Done, evtRowDone, evtColDone;
    hipCheck(hipEventCreateWithFlags(&evtP1Done, hipEventDisableTiming), "event p1");
    hipCheck(hipEventCreateWithFlags(&evtRowDone, hipEventDisableTiming), "event row");
    hipCheck(hipEventCreateWithFlags(&evtColDone, hipEventDisableTiming), "event col");

    // Try to pin the host vector memory to accelerate H2D/D2H
    hipHostRegister(h_dist.data(), bytes, hipHostRegisterDefault);

    hipCheck(hipMemcpyAsync(d_dist, h_dist.data(), bytes, hipMemcpyHostToDevice, stream0), "H2D memcpy");

    dim3 block(B, B);

    for (int k = 0; k < nBlocks; ++k) {
        // Phase 1 on stream0
        phase1_opt<<<1, block, 0, stream0>>>(d_dist, k, V);
        hipCheck(hipGetLastError(), "phase1 launch");
        hipCheck(hipEventRecord(evtP1Done, stream0), "record p1 done");

        if (nBlocks > 1) {
            // Phase 2 row on streamRow (wait for phase1)
            hipCheck(hipStreamWaitEvent(streamRow, evtP1Done, 0), "row wait p1");
            phase2_row_opt<<<nBlocks, block, 0, streamRow>>>(d_dist, k, V);
            hipCheck(hipGetLastError(), "phase2_row launch");
            hipCheck(hipEventRecord(evtRowDone, streamRow), "record row done");

            // Phase 2 col on streamCol (wait for phase1)
            hipCheck(hipStreamWaitEvent(streamCol, evtP1Done, 0), "col wait p1");
            phase2_col_opt<<<nBlocks, block, 0, streamCol>>>(d_dist, k, V);
            hipCheck(hipGetLastError(), "phase2_col launch");
            hipCheck(hipEventRecord(evtColDone, streamCol), "record col done");

            // Phase 3 on stream0 waits for both row/col
            hipCheck(hipStreamWaitEvent(stream0, evtRowDone, 0), "p3 wait row");
            hipCheck(hipStreamWaitEvent(stream0, evtColDone, 0), "p3 wait col");
            dim3 grid(nBlocks, nBlocks);
            phase3_opt<<<grid, block, 0, stream0>>>(d_dist, k, V);
            hipCheck(hipGetLastError(), "phase3 launch");
        }
    }

    hipCheck(hipMemcpyAsync(h_dist.data(), d_dist, bytes, hipMemcpyDeviceToHost, stream0), "D2H memcpy");
    hipCheck(hipStreamSynchronize(stream0), "final sync");

    hipHostUnregister(h_dist.data());
    hipFree(d_dist);
    hipStreamDestroy(stream0);
    hipStreamDestroy(streamRow);
    hipStreamDestroy(streamCol);
    hipEventDestroy(evtP1Done);
    hipEventDestroy(evtRowDone);
    hipEventDestroy(evtColDone);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./apsp input.txt" << std::endl;
        return 1;
    }
    
    int V, E;
    std::ifstream fin(argv[1]);
    fin >> V >> E;

    std::vector<int> h_dist(static_cast<size_t>(V) * static_cast<size_t>(V), INF);
    for (int i = 0; i < V; ++i) h_dist[static_cast<size_t>(i) * V + i] = 0;

    for (int e = 0; e < E; ++e) {
        int u, v, w;
        fin >> u >> v >> w;
        h_dist[static_cast<size_t>(u) * V + v] = w;
    }
    fin.close();

    solve(h_dist, V);

    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            std::cout << h_dist[static_cast<size_t>(i) * V + j];
            if (j != V - 1) std::cout << " ";
        }
        std::cout << "\n";
    }
    return 0;
}
