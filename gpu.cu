#include "common.h"
#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <stdio.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
int* part_id;
int* bin_id;
int* bin_cnt;
int* part_id_gpu;
int* bin_id_gpu;
int* bin_cnt_gpu;
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
// static void print_bins(int* part_id, int* bin_id, int Nbin, int num_parts) {
//    for (int i = 0; i < Nbin * Nbin; ++i) {
//        int row = i / Nbin;
//        int col = i % Nbin;
//        std::cout << "Bin (" << row << ", " << col << "): ";
//        int end = (i == Nbin * Nbin - 1) ? num_parts : bin_id[i + 1];
//        for (int j = bin_id[i]; j < end; ++j) {
//            std::cout << part_id[j] << ", ";
//        }
//        std::cout << std::endl;
//    }
//}

// print details of parts in each bin, for debugging
// static void print_parts_in_bins(PartBinsPerProc const& bins, int Nbin_col, int Nbin_row) {
//    for (int i = 0; i < Nbin_row; ++i) {
//        for (int j = 0; j < Nbin_col; ++j) {
//            for (auto const& part : bins[i][j]) {
//                std::cout << "Particle " << part.id << ": (x, y) = (" << part.x << ", " << part.y
//                          << "), (ax, ay) = (" << part.ax << ", " << part.ay << "), (vx, vy) = ("
//                          << part.vx << ", " << part.vy << ")" << std::endl;
//            }
//        }
//    }
//}

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

__global__ void compute_forces_gpu(particle_t* particles, int num_parts) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particles[tid].ax = particles[tid].ay = 0;
    for (int j = 0; j < num_parts; j++)
        apply_force_gpu(particles[tid], particles[j]);
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
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
}

__device__ int get_bin_idx(particle_t const& particle, int Nbin, double bin_size) {
    int idx_x = (int)(particle.x / bin_size);
    int idx_y = (int)(particle.y / bin_size);
    int idx = idx_x + idx_y * Nbin;
    return idx;
}

__global__ void fill_bin_cnt(particle_t const* parts, int* bin_cnt, int num_parts, int Nbin,
                             double bin_size) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;
    int idx = get_bin_idx(parts[tid], Nbin, bin_size);
    printf("Particle %d is in bin %d\n", tid, idx);
    atomicAdd(&bin_cnt[idx], 1);
    //    printf("bin count at %d = %d\n", idx, bin_cnt[idx]);
}

__global__ void init_bin_cnt(int* bin_cnt, int Nbin) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= Nbin * Nbin)
        return;
    bin_cnt[tid] = 0;
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    Nbin = (int)sqrt(num_parts);
    bin_size = size / Nbin;

    // init array
    part_id = new int[num_parts];
    bin_id = new int[Nbin * Nbin];
    //    std::fill_n(bin_id, Nbin * Nbin, Nbin * Nbin);
    bin_cnt = new int[Nbin * Nbin]{0};

    cudaMalloc((void**)&bin_cnt_gpu, Nbin * Nbin * sizeof(int));
    cudaMalloc((void**)&part_id_gpu, num_parts * sizeof(int));
    cudaMalloc((void**)&bin_id_gpu, Nbin * Nbin * sizeof(int));

    init_bin_cnt<<<blks, NUM_THREADS>>>(bin_cnt_gpu, Nbin);
    cudaDeviceSynchronize();
    //    cudaMemcpy(bin_cnt_gpu, bin_cnt, Nbin * Nbin * sizeof(int), cudaMemcpyHostToDevice);

    //    print_part<<<blks, NUM_THREADS>>>(parts, num_parts);
    //    cudaDeviceSynchronize();

    // get #parts in each bin
    fill_bin_cnt<<<blks, NUM_THREADS>>>(parts, bin_cnt_gpu, num_parts, Nbin, bin_size);
    //    get_bin_idx<<<blks, NUM_THREADS>>>(parts, num_parts, size);
    cudaDeviceSynchronize();

    cudaMemcpy(bin_cnt, bin_cnt_gpu, Nbin * Nbin * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < Nbin * Nbin; ++i) {
        std::cout << bin_cnt[i] << " ";
    }
    std::cout << std::endl;
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
