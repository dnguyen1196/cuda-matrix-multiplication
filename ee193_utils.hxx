#include <chrono>

// Two functions for timing how fast your code runs. Use them as
// auto t1 = start_time ();
//	... your code...
// long int usec = delta_usec(t1);
std::chrono::time_point<std::chrono::high_resolution_clock> start_time ();
long int delta_usec
	(std::chrono::time_point<std::chrono::high_resolution_clock> start);

// Two simple macros for printing, if you want them.
#define LOG(args) cout << args << endl
#define DIE(args) { cout << args << endl; exit(0); }

// This macro is for printing in a multithreaded environment. It uses a
// lock to ensure that output from different threads doesn't interleave with
// each other. The idea is that
// 1. You declare a mutex yourself.
// 2. You pass it as the first argument to every time you call LOGM.
#define LOGM(mut, args) { mut.lock(); cout << args << endl; mut.unlock(); }

// Check the return status after a CUDA call.
#define ERR_CHK(status, args) if (status != cudaSuccess) DIE (args << " (error code "<<cudaGetErrorString(status)<<")")
