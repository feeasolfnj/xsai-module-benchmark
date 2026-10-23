#ifndef K3_PMU_CYCLES_H
#define K3_PMU_CYCLES_H
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
/* Task-bound inherited event. Linux enables/disables and aggregates live
 * children in kernel. User-only raw cycles; no frequency scaling/subtraction. */
typedef struct {uint64_t count,enabled,running;} pmu_read_t;
static int pmu_fd=-1,pmu_ready,pmu_threads;static pmu_read_t pmu_before;
static uint64_t pmu_calls,pmu_units,pmu_sum,pmu_en,pmu_run,pmu_empty;
static void pmu_die(const char*s){perror(s);abort();}
static pmu_read_t pmu_read(void){pmu_read_t r;if(read(pmu_fd,&r,sizeof(r))!=sizeof(r))pmu_die("PMU read");return r;}
static void pmu_on(void){
 pmu_before=pmu_read();if(ioctl(pmu_fd,PERF_EVENT_IOC_ENABLE,0))pmu_die("PMU enable");__asm__ volatile("":::"memory");
}
static uint64_t pmu_off(int accumulate){
 __asm__ volatile("":::"memory");if(ioctl(pmu_fd,PERF_EVENT_IOC_DISABLE,0))pmu_die("PMU disable");
 pmu_read_t x=pmu_read(),b=pmu_before;uint64_t cy=x.count-b.count,en=x.enabled-b.enabled,run=x.running-b.running;
 if(en!=run||(!run&&cy)){fprintf(stderr,"PMU unscheduled/multiplexed enabled=%llu running=%llu\n",(unsigned long long)en,(unsigned long long)run);abort();}
 if(accumulate){pmu_sum+=cy;pmu_en+=en;pmu_run+=run;}return cy;
}
static void pmu_report(void){
 if(!pmu_ready)return;
 printf("[PMU_PROCESS] root_tid=%ld cycles=%llu inherited=1\n",syscall(__NR_gettid),(unsigned long long)pmu_sum);
 printf("[PMU_CYCLES] windows=%llu units=%llu threads=%d cycles=%llu enabled_ns=%llu running_ns=%llu cycles_per_unit=%.6f empty_window_cycles=%.6f raw=1 user_kernel=0 exclude_kernel=1 inherited=1 scaled=0\n",(unsigned long long)pmu_calls,(unsigned long long)pmu_units,pmu_threads,(unsigned long long)pmu_sum,(unsigned long long)pmu_en,(unsigned long long)pmu_run,pmu_units?(double)pmu_sum/pmu_units:0,(double)pmu_empty/32);
}
__attribute__((constructor(101))) static void pmu_open_before_threads(void){
 DIR*d=opendir("/proc/self/task");if(!d)pmu_die("PMU tasks");struct dirent*e;int n=0;while((e=readdir(d)))if(atoi(e->d_name)>0)n++;closedir(d);
 if(n!=1){fprintf(stderr,"PMU constructor found preexisting threads=%d\n",n);abort();}
 struct perf_event_attr a={0};a.type=PERF_TYPE_HARDWARE;a.size=sizeof(a);a.config=PERF_COUNT_HW_CPU_CYCLES;a.disabled=1;a.pinned=1;a.inherit=1;a.exclude_kernel=1;a.exclude_hv=1;a.read_format=PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
 pmu_fd=syscall(__NR_perf_event_open,&a,0,-1,-1,PERF_FLAG_FD_CLOEXEC);if(pmu_fd<0)pmu_die("perf_event_open cycles");atexit(pmu_report);
}
static void pmu_begin(void){
 if(!pmu_ready){DIR*d=opendir("/proc/self/task");if(!d)pmu_die("PMU tasks");struct dirent*e;while((e=readdir(d)))if(atoi(e->d_name)>0)pmu_threads++;closedir(d);pmu_ready=1;for(int i=0;i<32;i++){pmu_on();pmu_empty+=pmu_off(0);}}
 pmu_on();
}
static void pmu_end(unsigned units){pmu_off(1);pmu_calls++;pmu_units+=units;}
#endif
