#ifndef __GSYNC_H_
#define __GSYNC_H_

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stddef.h>
#include <errno.h>
#include <stdbool.h>
#include <time.h>
#include <unistd.h>



//helper



inline void gpuAssert(cudaError_t code, const char *file, int line) {

    if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),file, line);
      exit(code);
    }
}



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }






__device__ int getGlobalIdx_2D_2D();

typedef volatile unsigned int gpulock_t;

typedef struct ticketlock {

	unsigned int queue_ticket;
	unsigned int dequeue_ticket;

}gputlock_t;


typedef volatile unsigned int gbarrier_t;

//atomic

__device__ unsigned int FETCH_AND_ADD(int * addr,unsigned int val);

__device__ void INCREMENT(int * addr);

__device__ unsigned int FETCH_AND_INCREMENT(int * addr);

__device__ void DECREMENT(int * addr);

__device__ unsigned int FETCH_AND_DECREMENT(int * addr);

__device__ void XCHG(int * addr,unsigned int val);

__device__ unsigned int CMPXCHG(int * addr,unsigned int compare,unsigned int val);




//locks

__device__ void gsleep(unsigned int clock_count);

__host__ gpulock_t * gspinlock_init(gpulock_t * lock);

__device__ void gspinlock_lock(gpulock_t * lock);

__device__ void gspinlock_unlock(gpulock_t * lock);



__host__ gputlock_t * gticketlock_init(gputlock_t * lock);

__device__ void gticketlock_lock(gputlock_t * lock);

__device__ void gticketlock_unlock(gputlock_t * lock);





//barriers

__host__ gbarrier_t * gbarrier_init(gbarrier_t * barrier,unsigned int init_val);

__device__ void gbarrier_wait(gbarrier_t * barrier);

__device__ void gbarrier_destroy(gbarrier_t * barrier);





#endif
