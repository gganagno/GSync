
MACROS = ../../macros/c.linux.m4
 
CC = nvcc
CFLAGS = -g -G -O2 

BARFLAG = -DMBARRIER

SPIN = -DSPINLOCK
TICKET = -DTICKETLOCK
MCS = -DMCSLOCK
GPU = -DGPU

objects = test.o gsync.o

all: $(objects)
	nvcc -arch=sm_60 $(objects) -o out 

test.o: test.cu
	nvcc -x cu -arch=sm_60 -I. -dc $< -o $@

gsync.o: ./gsync/gsync.cu 
	nvcc -x cu -arch=sm_60 -I. -dc $< -o $@

clean: 
	rm -f gsync.o test.o out



