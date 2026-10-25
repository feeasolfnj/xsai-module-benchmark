#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "bench_module.h"
#include "PHY/TOOLS/tools_defs.h"
#include "common/config/config_userapi.h"
configmodule_interface_t*uniqCfg;
void exit_function(const char*f,const char*fn,int line,const char*m,int code){fprintf(stderr,"%s %d %s\n",f,line,m);abort();}
#ifdef BENCH_NATIVE_DFT
static void latency_setup(void){fprintf(stderr,"Cooperative RVV DFT is unavailable in the official profile\n");abort();}
static void latency_select(int n){(void)n;}
int latency_verification_status(void){return -1;}
#else
extern void latency_setup(void);extern void latency_select(int);
#endif
static int inverse_mode;static dftfunc_t saved_dft;static idftfunc_t saved_idft;
static int setup(size_t n,int inv){if(n!=2048)return -1;static int ready;if(!ready){load_dftslib();saved_dft=dft;saved_idft=idft;ready=1;}inverse_mode=inv;if(getenv("COOP_EIGHT")&&atoi(getenv("COOP_EIGHT"))){latency_setup();latency_select(1);}else{dft=saved_dft;idft=saved_idft;}return 0;}
static void run(int16_t*x,int16_t*y,unsigned char s){if(inverse_mode)idft(get_idft(2048),x,y,s);else dft(get_dft(2048),x,y,s);}
static void ref(int16_t*x,int16_t*y,unsigned char s){if(inverse_mode)saved_idft(get_idft(2048),x,y,s);else saved_dft(get_dft(2048),x,y,s);}
#ifdef BENCH_NATIVE_DFT
const bench_module selected_module={"local-native-2048","1 transform: local original serial implementation",setup,run,0};
#else
const bench_module selected_module={"xsai-cooperative-2048","1 transform: original serial / 8 cooperating threads; comparison status reported separately",setup,run,0};
#endif
const bench_module reference_module={"local-original","original local libdfts",setup,ref,0};
