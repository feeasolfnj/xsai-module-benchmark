#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "bench_module.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern int latency_verification_status(void);
static double tm(clockid_t id){struct timespec t;clock_gettime(id,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int cmp(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return (x>y)-(x<y);}
static int option(const char*k,int d){return getenv(k)?atoi(getenv(k)):d;}
int main(int argc,char**argv){
 int inverse=argc>1&&!strcmp(argv[1],"idft");
 unsigned seed=argc>2?strtoul(argv[2],0,0):1,rng=seed?seed:1;
 int bank=option("BENCH_INPUT_BANK",1),iterations=option("BENCH_ITERATIONS",10000),batches=option("BENCH_BATCHES",7),warmup=option("BENCH_WARMUP",10000);
 if(bank<1||bank>16||iterations<1||batches<1||batches>1000||warmup<0)return 2;
 int16_t *x,*y,*ref;assert(!posix_memalign((void**)&x,64,8192*bank));assert(!posix_memalign((void**)&y,64,8192));assert(!posix_memalign((void**)&ref,64,8192));
 const char*mode=getenv("BENCH_VERIFY");if(!mode)mode="strict";
 assert(!selected_module.setup(2048,inverse));
 int status=latency_verification_status(),skip=!strcmp(mode,"skip");
 for(int i=0;i<4096*bank;i++){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;x[i]=(rng%729)-364;}
 if(!skip){
  int matched=status!=0;
  for(int b=0;b<bank;b++){reference_module.run(x+4096*b,ref,1);selected_module.run(x+4096*b,y,1);if(memcmp(ref,y,8192))matched=0;}
  status=matched;
  if(!matched&&!strcmp(mode,"strict")){fprintf(stderr,"DFT comparison failed; --verify report permits diagnostic timing without a correctness claim.\n");return 2;}
 }else status=-1;
 for(int i=0;i<warmup;i++)selected_module.run(x+4096*(i%bank),y,1);
 double *samples=calloc(batches,sizeof(double));assert(samples);
 double c=tm(CLOCK_PROCESS_CPUTIME_ID),w=tm(CLOCK_MONOTONIC_RAW);
 for(int r=0;r<batches;r++){double t=tm(CLOCK_MONOTONIC_RAW);for(int j=0;j<iterations;j++)selected_module.run(x+4096*(j%bank),y,1);samples[r]=(tm(CLOCK_MONOTONIC_RAW)-t)*1e6/iterations;}
 double wall=tm(CLOCK_MONOTONIC_RAW)-w,cpu=tm(CLOCK_PROCESS_CPUTIME_ID)-c;
 qsort(samples,batches,sizeof(double),cmp);
 double median=batches%2?samples[batches/2]:(samples[batches/2-1]+samples[batches/2])/2;
 printf("[RESULT] transform=%s points=2048 median_batch_mean_us=%.9f mean_us=%.9f cores=%.6f verification=%s verification_scope=local_original_comparison input_bank=%d batches=%d iterations=%d seed=%u\n",inverse?"idft":"dft",median,wall*1e6/(batches*(double)iterations),cpu/wall,status<0?"skipped":status?"passed":"failed",bank,batches,iterations,seed);
 free(x);free(y);free(ref);free(samples);return 0;
}
