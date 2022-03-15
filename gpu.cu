#include "common.h"
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <thrust/scan.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
int* bin_cnt_gpu;
int* bin_id_gpu;
int* bin_id_cp_gpu;
int* part_id_gpu;
int Nbin;
double bin_size;

// print all particles in the parts array for debugging
__global__ void print_part(particle_t const* parts, int num_parts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;
    printf("Particle %d: (x, y) = (%f, %f), (vx, vy) = (%f, %f), (ax, ay) = (%f, %f)\n", tid,
           parts[tid].x, parts[tid].y, parts[tid].vx, parts[tid].vy, parts[tid].ax, parts[tid].ay);
}

// print part ids in each bin, for debugging
void print_bins(int* part_id, int* bin_id, int Nbin, int num_parts) {
    for (int i = 0; i < Nbin * Nbin; ++i) {
        int row = i / Nbin;
        int col = i % Nbin;
        std::cout << "Bin (" << row << ", " << col << "): ";
        int end = (i == Nbin * Nbin - 1) ? num_parts : bin_id[i + 1];
        for (int j = bin_id[i]; j < end; ++j) {
            std::cout << part_id[j] << ", ";
        }
        std::cout << std::endl;
    }
}

__device__ int get_bin_idx(particle_t const& particle, int Nbin, double bin_size) {
    int idx_x = (int)(particle.x / bin_size);
    int idx_y = (int)(particle.y / bin_size);
    int idx = idx_x + idx_y * Nbin;
    return idx;
}

__global__ void init_bin_cnt(int* bin_cnt, int Nbin) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= Nbin * Nbin)
        return;
    bin_cnt[tid] = 0;
}

__global__ void sync_src_to_cp(int* src, int* cp, int Nbin) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= Nbin * Nbin)
        return;
    cp[tid] = src[tid];
}

// get #parts in each bin
__global__ void fill_bin_cnt(particle_t const* parts, int* bin_cnt, int num_parts, int Nbin,
                             double bin_size) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;
    int idx = get_bin_idx(parts[tid], Nbin, bin_size);
    //    printf("Particle %d is in bin %d\n", tid, idx);
    atomicAdd(&bin_cnt[idx], 1);
}

__global__ void fill_part_id(particle_t const* parts, int* bin_id, int* part_id, int num_parts,
                             int Nbin, double bin_size) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;
    int idx = get_bin_idx(parts[tid], Nbin, bin_size);
    int old = atomicAdd(&bin_id[idx], 1);
    part_id[old] = tid;
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    Nbin = (int)sqrt(num_parts);
    bin_size = size / Nbin;

    // init array on GPU
    cudaMalloc((void**)&bin_cnt_gpu, Nbin * Nbin * sizeof(int));
    cudaMalloc((void**)&bin_id_gpu, Nbin * Nbin * sizeof(int));
    cudaMalloc((void**)&bin_id_cp_gpu, Nbin * Nbin * sizeof(int));
    cudaMalloc((void**)&part_id_gpu, num_parts * sizeof(int));

    // init bin_cnt_gpu to be array of 0s
    init_bin_cnt<<<blks, NUM_THREADS>>>(bin_cnt_gpu, Nbin);
    cudaDeviceSynchronize();

    // get #parts in each bin
    fill_bin_cnt<<<blks, NUM_THREADS>>>(parts, bin_cnt_gpu, num_parts, Nbin, bin_size);
    cudaDeviceSynchronize();

    // copy bin_cnt_gpu to bin_id_gpu
    sync_src_to_cp<<<blks, NUM_THREADS>>>(bin_cnt_gpu, bin_id_gpu, Nbin);
    cudaDeviceSynchronize();
    // get the exclusive prefix sum on GPU. Must in-place
    thrust::exclusive_scan(thrust::device, bin_id_gpu, bin_id_gpu + Nbin * Nbin,
                           bin_id_gpu); // in-place scan

    // copy bin_id_gpu to bin_id_cp_gpu
    sync_src_to_cp<<<blks, NUM_THREADS>>>(bin_id_gpu, bin_id_cp_gpu, Nbin);
    cudaDeviceSynchronize();
    // get part_id_gpu
    fill_part_id<<<blks, NUM_THREADS>>>(parts, bin_id_cp_gpu, part_id_gpu, num_parts, Nbin,
                                        bin_size);
    cudaDeviceSynchronize();

    //    print_part<<<blks, NUM_THREADS>>>(parts, num_parts);
    //    cudaDeviceSynchronize();
}

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__device__ void compute_force_neigh(particle_t* parts, particle_t& p, int* part_id, int* bin_id,
                                    int Nbin, int num_parts, double bin_size) {
    // get bin idx for the particle
    int idx = get_bin_idx(p, Nbin, bin_size);
    int row = idx / Nbin;
    int col = idx % Nbin;
    int row_start = (row - 1 >= 0) ? row - 1 : 0;
    int row_end = (row + 1 < Nbin) ? row + 1 : Nbin - 1;
    int col_start = (col - 1 >= 0) ? col - 1 : 0;
    int col_end = (col + 1 < Nbin) ? col + 1 : Nbin - 1;

    for (int i = row_start; i <= row_end; ++i) {
        for (int j = col_start; j <= col_end; ++j) {
            int bin_idx = i * Nbin + j;
            int end = (bin_idx == Nbin * Nbin - 1) ? num_parts : bin_id[bin_idx + 1];
            for (int k = bin_id[bin_idx]; k < end; ++k) {
                int neigh_id = part_id[k];
                apply_force_gpu(p, parts[neigh_id]);
            }
        }
    }
}

__global__ void compute_forces_gpu(particle_t* parts, int* part_id, int* bin_cnt, int* bin_id,
                                   int num_parts, int Nbin, double bin_size) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    parts[tid].ax = parts[tid].ay = 0;

    // compute forces from neighboring parts
    compute_force_neigh(parts, parts[tid], part_id, bin_id, Nbin, num_parts, bin_size);
}

__global__ void move_gpu(particle_t* particles, int* bin_cnt, int num_parts, int Nbin,
                         double bin_size, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    // get the old bin idx before updating p
    int idx_old = get_bin_idx(*p, Nbin, bin_size);

    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }

    // get the new bin idx after moving p
    int idx_new = get_bin_idx(*p, Nbin, bin_size);

    if (idx_old != idx_new) {
        atomicAdd(&bin_cnt[idx_new], 1);
        atomicSub(&bin_cnt[idx_old], 1);
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, part_id_gpu, bin_cnt_gpu, bin_id_gpu,
                                              num_parts, Nbin, bin_size);
    cudaDeviceSynchronize();

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, bin_cnt_gpu, num_parts, Nbin, bin_size, size);
    cudaDeviceSynchronize();

    // rebinning all particles
    // copy bin_cnt_gpu to bin_id_gpu
    sync_src_to_cp<<<blks, NUM_THREADS>>>(bin_cnt_gpu, bin_id_gpu, Nbin);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, bin_id_gpu, bin_id_gpu + Nbin * Nbin,
                           bin_id_gpu); // in-place scan
    sync_src_to_cp<<<blks, NUM_THREADS>>>(bin_id_gpu, bin_id_cp_gpu, Nbin);
    cudaDeviceSynchronize();

    // get part_id_gpu
    fill_part_id<<<blks, NUM_THREADS>>>(parts, bin_id_cp_gpu, part_id_gpu, num_parts, Nbin,
                                        bin_size);
}
