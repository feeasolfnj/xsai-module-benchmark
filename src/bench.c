#define _GNU_SOURCE
#include "bench_module.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double tm(clockid_t id){struct timespec t;clock_gettime(id,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int cmp(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return (x>y)-(x<y);}
int main(int argc,char**argv){int inverse=argc>1&&!strcmp(argv[1],"idft");unsigned seed=argc>2?strtoul(argv[2],0,0):1,rng=seed?seed:1;int16_t x[4096] __attribute__((aligned(64))),y[4096] __attribute__((aligned(64))),ref[4096] __attribute__((aligned(64)));assert(!selected_module.setup(2048,inverse));
for(int t=0;t<100;t++){for(int j=0;j<4096;j++){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;x[j]=(rng%729)-364;}reference_module.run(x,ref,1);selected_module.run(x,y,1);assert(!memcmp(ref,y,sizeof(y)));}
for(int j=0;j<10000;j++)selected_module.run(x,y,1);double samples[7],c=tm(CLOCK_PROCESS_CPUTIME_ID),w=tm(CLOCK_MONOTONIC_RAW);for(int r=0;r<7;r++){double t=tm(CLOCK_MONOTONIC_RAW);for(int j=0;j<10000;j++)selected_module.run(x,y,1);samples[r]=(tm(CLOCK_MONOTONIC_RAW)-t)*1e6/10000;}double wall=tm(CLOCK_MONOTONIC_RAW)-w,cpu=tm(CLOCK_PROCESS_CPUTIME_ID)-c;qsort(samples,7,sizeof(double),cmp);printf("[RESULT] transform=%s points=2048 median_us=%.9f cores=%.6f verified=1 seed=%u\n",inverse?"idft":"dft",samples[3],cpu/wall,seed);return 0;}
