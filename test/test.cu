#include "./gsync/gsync.h"



__device__ int gtt = 2;

int tt=1;


__global__ void mykernel(gpulock_t * lock, gputlock_t * tlock, gbarrier_t * barrier) {


    gspinlock_lock(lock);

    printf("[D] TT : %d\n",gtt );\
    // tt = 2;
    printf("[D] (cs%d), %f\n",getGlobalIdx_2D_2D(),clock64());

    for(int i=0;i<1000000000;i++);
      
    gspinlock_unlock(lock);

    

    // printf("[D] (before %d,%u)\n",getGlobalIdx_2D_2D(),*barrier);

    // gbarrier_wait(barrier);
    
    // printf("[D] (after %d,%u)\n",getGlobalIdx_2D_2D(),*barrier);

}


__host__ void usage(){

  printf(
      "\n"
      "Usage:\n"
      "    sas_mul -f input_file -n table_size -p number_of_processors [-v -h]\n" 
  );

  printf(
      "\n"
      "Options:\n"
      " -f    path    Path to input file\n"
      " -p    number  Number of Processors\n"
      " -n    number  Number of Table Size. (Assume that N is a power of P)\n"
      " -v            Verbose Printing\n"
      " -h            This help message\n"
  );

  exit(EXIT_FAILURE);
}

int P = 2;


int main(int argc, char ** argv){

  // cudaGetDeviceProperties(&deviceProp, 0);

  // freq = deviceProp.clockRate; 

  gpulock_t * lock = NULL;

  gputlock_t * tlock = NULL;

  gbarrier_t * barrier = NULL;

  int opt;

  FILE * fp = NULL;
  
  char * ff = NULL;

  int verbose = 0;

  cudaEvent_t start,end;

  cudaEventCreate(&start);

  cudaEventCreate(&end);

  while ((opt = getopt(argc, argv,"p:f:vh")) != -1) {

    switch (opt) {


      case 'p':

        P = atoi(optarg);

        if(P < 1 || P > 1024){
          printf("Invalid number of processors (P < 1 || P > 8) \nExiting..\n");
          exit(EXIT_FAILURE);
        }
        break;

      case 'f':

        fp = fopen(optarg,"r");

        ff = strdup(optarg);

        if (fp == NULL || ff == NULL){
          printf("File is null\nExiting..\n");
          exit(EXIT_FAILURE);
        }

        break;

      case 'v':
        verbose=1;
        break;

      case 'h':
        usage();
        break;

      default:
        usage();
    }
  }

  lock = gspinlock_init(lock);

  tlock = gticketlock_init(tlock);

  barrier = gbarrier_init(barrier,P);


  

  printf("[H] TT : %d\n",tt );

  cudaEventRecord(start);

  cudaMemcpyToSymbol(gtt,&tt, sizeof(int), 0, cudaMemcpyHostToDevice);

  mykernel<<<P,1>>>(lock,tlock,barrier);

  cudaEventRecord(end);

  cudaEventSynchronize(end);
  
  float elapsed = 0;

  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpyFromSymbol(&tt,gtt, sizeof(int), 0, cudaMemcpyDeviceToHost);

  printf("[H] TT : %d\n",tt );

  cudaEventElapsedTime(&elapsed,start,end);

  printf("Elapsed %f ms\n",elapsed );


  if(verbose)printf("Host exiting..\n");
  return 0;
  
}
