#include "gsync.h"

//////////////////////////////////////////////////////////////////////////////////////
/// HELPER FUNCTIONS /////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////




 __device__ int getGlobalIdx_2D_2D()
{
  int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
  int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}




//////////////////////////////////////////////////////////////////////////////////////
// ATOMIC FUNCTIONS //////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

__device__ unsigned int FETCH_AND_ADD(unsigned int * addr,unsigned int val){
     return atomicAdd((unsigned int *)addr,val); 
}


__device__ void INCREMENT(unsigned int * addr){
     atomicInc((unsigned int *)addr,1); 
}


__device__ unsigned int FETCH_AND_INCREMENT(unsigned int * addr){
     return atomicInc((unsigned int *)addr,1); 
}



__device__ void DECREMENT(unsigned int * addr){
    atomicDec((unsigned int *)addr,1); 
}

__device__ unsigned int FETCH_AND_DECREMENT(unsigned int * addr){
     return atomicDec((unsigned int *)addr,1); 
}



__device__ void XCHG(unsigned int * addr,unsigned int val){
    atomicExch((unsigned int *)addr, val); 
}

__device__ unsigned int CMPXCHG(unsigned int * addr,unsigned int compare,unsigned int val){
    return atomicCAS((unsigned int *)addr, compare,val); 
}



__device__ void gsleep(unsigned int clock_count) {

  volatile clock_t avoid_opt;


  clock_t start_clock = clock64();

  clock_t gpu_cycles = 0;

  while (gpu_cycles < clock_count) gpu_cycles = clock64() - start_clock;
  
  avoid_opt = gpu_cycles;

}


//////////////////////////////////////////////////////////////////////////////////////
///  LOCKS  //////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////


__host__ gpulock_t * gspinlock_init(gpulock_t * lock){

  gpulock_t * hostlock = (gpulock_t *)malloc( sizeof(gpulock_t) );
  
  *hostlock=0;

  gpuErrchk( cudaMalloc((void**)&lock ,sizeof(gpulock_t)));

  gpuErrchk( cudaMemcpy((void*)lock,(const void *)hostlock, sizeof(gpulock_t), cudaMemcpyHostToDevice ));
  
  return lock;
}



__device__ void gspinlock_lock(gpulock_t * lock){

  int verbose=0;

  if(verbose)printf("[D] lock %d ,%d\n",*lock,getGlobalIdx_2D_2D());

  assert((unsigned int)*lock == 0U || (unsigned int)*lock ==1U);

  while( (CMPXCHG((unsigned int*)lock,0U,1U)) ) gsleep(1000);
  

}


__device__ void gspinlock_unlock(gpulock_t * lock){

  int verbose=0;

  assert((unsigned int)*lock == 0U || (unsigned int)*lock == 1U);

  if(verbose)printf("[UN1] lock : %d,%d\n",*lock,getGlobalIdx_2D_2D());

  if((unsigned int)*lock == 1U) DECREMENT((unsigned int *)lock);

 

}



//////////////////////////////////////////////////////////////////////////////////////
///  TICKETLOCKS  //////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////




__host__ gputlock_t * gticketlock_init(gputlock_t * lock){

 
  gputlock_t * hostlock = (gputlock_t *)malloc( sizeof(gputlock_t) );
  
  hostlock->queue_ticket =0;
  
  hostlock->dequeue_ticket =0;

  gpuErrchk( cudaMalloc((void**)&lock ,sizeof(gputlock_t)));

  gpuErrchk( cudaMemcpy((void*)lock,(const void *)hostlock, sizeof(gputlock_t), cudaMemcpyHostToDevice ));
  
  return lock;

}


__device__ void gticketlock_lock(gputlock_t * lock){

  int current= FETCH_AND_INCREMENT( &lock->queue_ticket );

  while(current != lock->dequeue_ticket) gsleep(400);

}


__device__ void gticketlock_unlock(gputlock_t * lock){

  INCREMENT(& lock->dequeue_ticket);
}




//////////////////////////////////////////////////////////////////////////////////////
///  BARRIERS  ///////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////


__host__ gbarrier_t * gbarrier_init(gbarrier_t * barrier, unsigned int init_value){

    gbarrier_t * hostbarrier = (gbarrier_t *)malloc( sizeof(gbarrier_t) );

    *hostbarrier = init_value;

    gpuErrchk( cudaMalloc(( void**)& barrier ,sizeof(gbarrier_t) ) );

    gpuErrchk( cudaMemcpy((void*)barrier,(const void*)hostbarrier, sizeof(gbarrier_t), cudaMemcpyHostToDevice ));

    return barrier;
}


__device__ void gbarrier_wait(gbarrier_t * barrier){

    DECREMENT((unsigned int *)barrier);
    while((unsigned int) *barrier > 0) gsleep(4000);
    __threadfence();
    __syncthreads();
}


__device__ void gbarrier_destroy(gbarrier_t * barrier){
  cudaFree((void**)&barrier);
}

