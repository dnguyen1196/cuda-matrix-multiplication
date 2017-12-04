/*
 * CUDA matrix multiply
 * nvcc -std=c++11 -O2 -arch=compute_50 -I /usr/local/cuda/samples/common/inc mat_mult.cu ../ee193_utils.cxx
 */

#include <vector>
#include <iostream>
#include <sstream>
#include <assert.h>
#include "ee193_utils.hxx"
#include "bits.hxx"
using namespace std;

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "helper_cuda.h"

const int BS=16;	// The blocks are BS x BS.

///////////////////////////////
// Matrix is the main class that we use.
// It has methods to declare a matrix, allocate space, initialize it, do slow
// single-threaded matrix multiply, and printing support.
// It also has one fast matrix-multiply method that you'll write yourself.
///////////////////////////////
class Matrix {
    // The private members.
    vector<float> data;	// Data is stored in a 1D vector.
    int _N;		// NxN matrix (where N = 2^LOG2_N).
    int nbits_per_dim;	// This is the LOG2_N we're given.
    int index (int r, int c) const { return ((r << this->nbits_per_dim) | c); }

  public:
    Matrix (int nbits_per_dim);	// Create a matrix, allocate its storage.
    int N() const { return (this->_N); }

    // Access an element (note that operator[] can only take 1 arg, not 2).
    float &operator() (int r,int c) {return(this->data[this->index(r,c)]);}
    float operator() (int r,int c) const {return(this->data[this->index(r,c)]);}

    bool operator== (const Matrix &other) const;	// Full equality check
    void compare (const Matrix &M2) const;		// Die on first mismatch

    // Initialize a matrix; to I, to random #s in [0,1], or cyclic ints.
    void init_identity();
    void init_random(float min, float max);	
    void init_cyclic_order ();

    void mpy_dumb (const Matrix &A, const Matrix &B);	// 1 thread, unblocked
    void mpy1     (const Matrix &A, const Matrix &B);	// CUDA version

    string row_str(int row) const;	// Print one matrix row to a string.
    string str() const;			// Ditto for the entire matrix.
};

Matrix::Matrix (int nbits_per_Dim) {
    this->nbits_per_dim = nbits_per_Dim;
    this->_N = (1<<nbits_per_dim);
    unsigned int n_elements = (1 << (nbits_per_dim+nbits_per_dim));
    this->data = vector<float> (n_elements);
}

bool Matrix::operator== (const Matrix &other) const {
    return (this->data == other.data);
}

// Like ==. But: on mismatch, prints the first mismatching element and dies.
void Matrix::compare (const Matrix &M2) const {
    for (int r=0; r<_N; ++r)
	for (int c=0; c<_N; ++c)
	    if ((*this)(r,c) != M2(r,c))
		DIE ("M1["<<r<<","<<c<<"]="<<(*this)(r,c)
		     << ", M2["<<r<<","<<c<<"]="<<M2(r,c));
}

void Matrix::init_identity() {
    for (int r=0; r<_N; ++r)
	for (int c=0; c<_N; ++c)
	    this->data[index(r,c)] = ((r==c)?1.0:0.0);
}

void Matrix::init_cyclic_order() {
    for (int r=0; r<_N; ++r)
	for (int c=0; c<_N; ++c)
	    this->data[index(r,c)] = bit_get (r+c, this->nbits_per_dim-1, 0);
}

// Printing support.
string Matrix::row_str(int row) const {
    ostringstream os;
    os << "{";
    for (int c=0; c<_N; ++c)
	os << (c==0?"":", ") << (*this)(row,c);
    os << "}";
    return (os.str());
}
string Matrix::str() const {
    string s = "{";
    for (int r=0; r<_N; ++r)
	s += this->row_str(r);
    s += "}";
    return (s);
}

// Simple algorithm for multiplying two matrices.
void Matrix::mpy_dumb (const Matrix &A, const Matrix &B) {
    for (int r=0; r<_N; ++r)
	for (int c=0; c<_N; ++c) {
	    float sum=0.0;
	    for (int k=0; k<_N; ++k)
		sum += (A(r,k) * B(k,c));
	    this->data[index(r,c)] = sum;
	}
}


__global__ int map(int ri, int ci, int rb, int cb, int N){
    return N*(rb*BS+ri)+cb*BS+ci;
}

///////////////////////////////
// This is the CUDA kernel function for you to write.
//
__global__ void mat_mult (float *d_A, float *d_B, float *d_C, int N) {
    int rb = blockIdx.x;
    int cb = blockIdx.y;
    int ri = threadIdx.x;
    int ci = threadIdx.y;

    // Allocate shared memory
    __shared__ float SA[BS][BS],SB[BS][BS];

    int sum = 0; // This will store final value of C[ri,ci]
    // Copy the data to shared memory
    for (int kb = 0; kb < N; kB++) {
        SA[ri, ci] = d_A[map(ri, ci, rb, kb, N)];
        SB[ri, ci] = d_B[map(ri, ci, kb, cb, N)];
        __syncthreads();

        // Do actual computations
        for (int ki = 0; ki < BS; ki++) {
            sum += SA[ri, ki] * SB[ki, ci];
        }
        __syncthreads();
    }
    // Copy sum back to d_C
    d_C[map(ri,ci,rb,cb,N)] = sum;
}




///////////////////////////////
// This is the host function for you to write.
// It allocates memory and moves data between CPU<->GPU
//
void Matrix::mpy1 (const Matrix &A, const Matrix &B) {
    auto start = start_time();

    // Copy A from host memory to device memory.
    int numElem=_N*_N, sizeBytes = numElem*4;
    float *d_A = NULL;
    cudaError_t err = cudaMalloc((void **)&d_A, sizeBytes);
    ERR_CHK (err, "Failed to allocate device matrix A");
    cudaMemcpy(d_A, A.data(), sizeBytes, cudaMemcpyHostToDevice);

    // Allocate memory for B and copy data from host to device
    float *d_B = NULL;
    cudaError_t err = cudaMalloc((void **)&d_B, sizeBytes);
    ERR_CHK (err, "Failed to allocate device matrix B");
    cudaMemcpy(d_B, B.data(), sizeBytes, cudaMemcpyHostToDevice);

    // Alocate memory for C
    float *d_C = NULL;
    cudaError_t err = cudaMalloc((void **)&d_C, sizeBytes);
    ERR_CHK (err, "Failed to allocate device matrix C");

    int Nb = A.N()/BS;
    dim3 thBlocks(Nb, Nb), threads(BS, BS);

    // Do the computations and write on d_C
    mat_mult<<<thBlocks,threads>>>(d_A,d_B,d_C,A.N());

    // Copy the memory back from d_C to host memory
    cudaMemcpy(this->data, d_C, sizeBytes, cudaMemcpyDeviceToHost);

    long int time = delta_usec (start);
    cout<<"mpy1 took "<<(time/1000000.0)<<"sec"<<endl;
}

// This function executes the various piecese for a given matrix size.
static void run (int log2_N) {
    Matrix a(log2_N), b(log2_N), c(log2_N), d(log2_N);
    a.init_cyclic_order();
    b.init_identity();
    int N = 1<<log2_N;
    LOG ("Working on "<<N<<"x"<<N<<" matrices.");

    for (int i=0; i<1; ++i) {
	auto start = start_time();
	c.mpy_dumb (b, a);
	long int time = delta_usec (start);
	    LOG ("Dumb mpy took "<<(time/1000000.0)<<"sec");
    }

    for (int i=0; i<4; ++i) {
	d.mpy1 (b, a);
	//LOG ("Ref C="<<c.str()<<", D="<<d.str());
	c.compare (d);
    }
}

// Main() lives on the CPU.
int main() {
    run (10);	// Matrix size 1Kx1K
    run (11);	// Matrix size 2Kx2K
    run (12);	// Matrix size 4Kx4K
    return (0);
}
