define(G_MALLOC, `
  #ifdef GPU
    cudaMallocManaged((void**)&$1,$2);
  #else
    malloc($1);
  #endif
    ')

define(MALLOC, ` $1 = malloc($2);')

define(EXTERN_ENV,
`
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <pthread.h>
#include <getopt.h>

#define MAX_THREADS 8

typedef struct{
 int needed;
 int called;
 pthread_mutex_t mutex;
 pthread_cond_t cond;
} barrier_t;

extern void barrier_init(barrier_t *);
extern void barrier_wait(barrier_t *, int);
extern struct timeval the_time;
extern pthread_cond_t barrier_cond;
extern pthread_mutex_t barrier_mutex;
extern int thread_count;
extern pthread_t threads[MAX_THREADS];
extern int id;
extern LOCKDEC(id_lock);
')

define(MAIN_ENV, 

#ifdef GPU 

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <pthread.h>
#include <getopt.h>

#define MAX_THREADS 8

struct timeval the_time;
pthread_cond_t barrier_cond;
pthread_mutex_t barrier_mutex;
int thread_count = 0;
pthread_t threads[MAX_THREADS];
int id = 0;
LOCKDEC(id_lock);

#else
`
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <pthread.h>
#include <getopt.h>

#define MAX_THREADS 8

typedef struct{
 int needed;
 int called;
 pthread_mutex_t mutex;
 pthread_cond_t cond;
} barrier_t;

void barrier_init(barrier_t *barrier)
{
    barrier->needed = -1;
    barrier->called = 0;
    pthread_mutex_init(&barrier->mutex,NULL);
    pthread_cond_init(&barrier->cond,NULL);
}

void barrier_wait(barrier_t *barrier, int cpu_no)
{
    pthread_mutex_lock(&barrier->mutex);
    if (barrier->needed == -1) barrier->needed = cpu_no;
    if (barrier->needed != cpu_no) {printf("Error in appl\n");}
    barrier->called++;
    if (barrier->called == barrier->needed) {
        barrier->called = 0;
        barrier->needed = -1;
        pthread_cond_broadcast(&barrier->cond);
    } else {
        pthread_cond_wait(&barrier->cond,&barrier->mutex);
    }
    pthread_mutex_unlock(&barrier->mutex);
}

struct timeval the_time;
pthread_cond_t barrier_cond;
pthread_mutex_t barrier_mutex;
int thread_count = 0;
pthread_t threads[MAX_THREADS];
int id = 0;
LOCKDEC(id_lock);

#endif
')

define(MAIN_INITENV, `
LOCKINIT(id_lock);
pthread_mutex_init(&barrier_mutex, NULL);
pthread_cond_init(&barrier_cond, NULL);
pthread_mutex_lock(&barrier_mutex);
pthread_mutex_unlock(&barrier_mutex);
')

define(MAIN_END, 
#ifdef GPU

#else
`{int ret; pthread_exit(&ret);}
#endif
')

#ifdef GPU
#define CREATE(func,p,args...) { func <<< p,1 >>> (p,args); gpuErrchk(cudaDeviceSynchronize()); }
#else

#define CREATE(...) { pthread_create(&threads[thread_count],NULL,$1,NULL); thread_count++; }

#endif


define(BARDEC, `

#ifdef MBARRIER
  mbarrier_t $1;
#elif GBARRIER
  gbarrier_t * $1 = NULL;
#else
  barrier_t $1;
#endif

')

define(BARINIT,

  `
   #ifdef MBARRIER
    mbarrier_init(&$1,$2);
    #elif GBARRIER
   $1 = gbarrier_init($1,$2);
  #else
    barrier_init(&$1);
  #endif
  ')

define(BARRIER, `

  #ifdef MBARRIER
    mbarrier_wait(&$1);
  #elif GBARRIER
    gbarrier_wait($1);
  #else
    barrier_wait(&$1,id);
  #endif

  ')

define(LOCKDEC,  `

  #ifdef SPINLOCK
    spinlock_t $1;
  #elif TICKETLOCK
    ticketlock_t $1;

  #elif GSPINLOCK
    gpulock_t * $1 = NULL;
  #elif GTICKETLOCK
    gputlock_t *$1 = NULL;

  #elif MCSLOCK

    mcslock_t $1;
    mcslock_node mynode;

  #else
    pthread_mutex_t $1;
  #endif
  ')
 
define(LOCKINIT, `

  #ifdef SPINLOCK
    spinlock_init((spinlock_t*)&($1));
  #elif TICKETLOCK
    ticketlock_init((ticketlock_t*)&($1));


   #elif GSPINLOCK
    $1 = gspinlock_init((gpulock_t*)($1));
  #elif GTICKETLOCK
    $1 = gticketlock_init((gputlock_t*)($1));


  #elif MCSLOCK
    mcslock_init((mcslock_t*)&($1),&mynode);
  #else
    pthread_mutex_init((pthread_mutex_t*)&($1), NULL);
  #endif

')

define(LOCK,     `

  #ifdef SPINLOCK
    spinlock_lock((spinlock_t*)&($1));
  #elif TICKETLOCK
    ticketlock_lock((ticketlock_t*)&($1));

  #elif GSPINLOCK
    gspinlock_lock((gpulock_t*)($1));
  #elif GTICKETLOCK
    gticketlock_lock((gputlock_t*)($1));


  #elif MCSLOCK
    mcslock_lock((mcslock_t*)&($1),&mynode);
  #else
    pthread_mutex_lock((pthread_mutex_t*)&($1));
  #endif

')
  

define(UNLOCK,   `

  #ifdef SPINLOCK
    spinlock_unlock((spinlock_t*)&($1));
  #elif TICKETLOCK
    ticketlock_unlock((ticketlock_t*)&($1));


    #elif GSPINLOCK
    gspinlock_unlock((gpulock_t*)($1));
  #elif TICKETLOCK
    gticketlock_unlock((gputlock_t*)($1));


  #elif MCSLOCK
    mcslock_unlock((mcslock_t*)&($1),&mynode);
  #else
    pthread_mutex_unlock((pthread_mutex_t*)&($1));
  #endif
')

define(ACQUIRE,  `pthread_acquire((pthread_mutex_t*)&($1));')
define(RELEASE,  `pthread_release((pthread_mutex_t*)&($1));')

define(ALOCKDEC,  `pthread_mutex_t ($1[$2]);')
define(ALOCKINIT, `{
                      int loop_j;
                      for(loop_j=0; loop_j < $2; loop_j++){
                          pthread_mutex_init((pthread_mutex_t*)&($1[loop_j]), NULL);
                      }
                   }')
define(ALOCK,      `pthread_mutex_lock((pthread_mutex_t*)&($1[$2]));')
define(AULOCK,     `pthread_mutex_unlock((pthread_mutex_t*)&($1[$2]));')

define(AACQUIRE,   `pthread_acquire((pthread_mutex_t*)&($1[$2]));')
define(ARELEASE,   `pthread_release((pthread_mutex_t*)&($1[$2]));')

define(WAIT_FOR_END, `

#ifdef GPU 
cudaDeviceSynchronize();
#else
  {
    int i;
    for(i=0;i< $1 ;i++){
      pthread_join(threads[i],NULL);
    }
  }
#endif
  ')

define(CLOCK, `{
  #ifdef GPU
  $1 = clock();
  #else
    unsigned long now;
    gettimeofday(&the_time, NULL);
    now = ((the_time.tv_sec - 879191283) * 1000) + (the_time.tv_usec / 1000);
    $1 = (unsigned int)now;
  #endif
    }')

define(RESET_STATISTICS, `')

define(GCLOCK_START, `

  cudaEvent_t start, end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);

               ')

define(GCLOCK_END, `
  ;cudaEventRecord(end);
  ;cudaEventSynchronize(end);
  ')

define(GCLOCK_DIFF, `{
 cudaEventElapsedTime(&$1,start,end);
               }')


define(GET_PID, `

#ifdef GPU
  $1 = getGlobalIdx_2D_2D();
#else
{
	LOCK(id_lock);
	$1 = id++;
	UNLOCK(id_lock);
}
#endif
')


define(AUG_ON, `')
define(AUG_OFF, `')
define(ALLOCATE_PAGE, `')
define(ALLOCATE_RANGE, `')
