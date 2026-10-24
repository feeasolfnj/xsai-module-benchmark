#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <sched.h>
#include <dlfcn.h>
#include <riscv_vector.h>
#include "PHY/TOOLS/tools_defs.h"
typedef struct{_Atomic uint32_t value;char pad[124];} __attribute__((aligned(128))) stamp_t;
static stamp_t stages[2][8],finished[8];
static _Atomic uint32_t sequence __attribute__((aligned(128)));
static _Atomic int closing;
static _Thread_local uint32_t job_seq;
static int32_t work[2048] __attribute__((aligned(128)));
static uint32_t reverse_table[2048] __attribute__((aligned(128)));
static const int16_t *tw;
static const int16_t *current_input;
static int16_t *current_output;
static int current_inverse,current_scale;
static pthread_t workers[7];
static dftfunc_t original_dft;
static idftfunc_t original_idft;
static int initialized,enabled;
static int verification_status=-1;
int latency_verification_status(void){return verification_status;}
static void wait_stamp(stamp_t *s,uint32_t wanted){unsigned loops=0;while(atomic_load_explicit(&s->value,memory_order_acquire)!=wanted){if(++loops==100000){sched_yield();loops=0;}}}
static void fft_barrier(int lane,int phase){atomic_store_explicit(&stages[phase][lane].value,job_seq,memory_order_release);for(int i=0;i<8;i++)wait_stamp(&stages[phase][i],job_seq);}
static void lane_fft(const int16_t *input,int16_t *output,const int16_t *twiddle,const uint32_t *bitrev,int inverse,int final_scale,int lane,int32_t *data);
static void serial_optimized_fft(const int16_t *input,int16_t *output,const int16_t *twiddle,const uint32_t *bitrev,int inverse,int final_scale);
static void *worker(void *arg){
 int lane=(int)(intptr_t)arg;uint32_t last=0;
 for(;;){uint32_t seq=atomic_load_explicit(&sequence,memory_order_acquire);
  if(seq==last){syscall(SYS_futex,&sequence,FUTEX_WAIT_PRIVATE,last,NULL,NULL,0);continue;}
  if(atomic_load_explicit(&closing,memory_order_acquire))break;
  job_seq=seq;lane_fft(current_input,current_output,tw,reverse_table,current_inverse,current_scale,lane,work);
  atomic_store_explicit(&finished[lane].value,seq,memory_order_release);last=seq;
 }
 return NULL;
}
static void parallel_fft(const int16_t *input,int16_t *output,int inverse,int scale){
 uint32_t seq=atomic_load_explicit(&sequence,memory_order_relaxed)+1;
 current_input=input;current_output=output;current_inverse=inverse;current_scale=scale;
 job_seq=seq;atomic_store_explicit(&sequence,seq,memory_order_release);syscall(SYS_futex,&sequence,FUTEX_WAKE_PRIVATE,7,NULL,NULL,0);
 lane_fft(input,output,tw,reverse_table,inverse,scale,0,work);
 for(int i=1;i<8;i++)wait_stamp(&finished[i],seq);
}
static void parallel_dft(uint8_t size,int16_t *input,int16_t *output,unsigned char scale){if(size==get_dft(2048)){if(enabled==2)serial_optimized_fft(input,output,tw,reverse_table,0,scale>0);else parallel_fft(input,output,0,scale>0);}else original_dft(size,input,output,scale);}
static void parallel_idft(uint8_t size,int16_t *input,int16_t *output,unsigned char scale){if(size==get_idft(2048)){if(enabled==2)serial_optimized_fft(input,output,tw,reverse_table,1,scale>0);else parallel_fft(input,output,1,scale>0);}else original_idft(size,input,output,scale);}
void latency_setup(void){
 if(initialized)return;initialized=1;original_dft=dft;original_idft=idft;
 Dl_info info;assert(dladdr((void*)dft,&info));void*h=dlopen(info.dli_fname,RTLD_NOW|RTLD_NOLOAD);assert(h);tw=dlsym(h,"tw2048");assert(tw);
 for(unsigned i=0;i<2048;i++){unsigned x=i,y=0;for(int b=0;b<11;b++){y=(y<<1)|(x&1);x>>=1;}reverse_table[i]=y;}
 for(int i=1;i<8;i++)assert(!pthread_create(&workers[i-1],NULL,worker,(void*)(intptr_t)i));

 const char*mode=getenv("BENCH_VERIFY");if(mode&&!strcmp(mode,"skip"))return;
 verification_status=1;
 int16_t *x,*ref,*out;assert(!posix_memalign((void**)&x,64,8192));assert(!posix_memalign((void**)&ref,64,8192));assert(!posix_memalign((void**)&out,64,8192));uint32_t seed=20261017;
 unsigned mismatches=0;
 for(int t=0;t<16;t++){
  for(int i=0;i<4096;i++){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;x[i]=t<8?(int16_t)((int)(seed%4096)-2048):(int16_t)seed;}
  for(int inverse=0;inverse<2;inverse++)for(int scale=0;scale<2;scale++){
   if(inverse)original_idft(get_idft(2048),x,ref,scale);else original_dft(get_dft(2048),x,ref,scale);
   parallel_fft(x,out,inverse,scale);
   if(memcmp(ref,out,8192))mismatches++;
  }
 }
 verification_status=mismatches==0;
 printf("[LATENCY_VERIFY] patterns=16 directions=2 scales=2 mismatch_cases=%u bit_identical=%d\n",mismatches,verification_status);
 free(x);free(ref);free(out);
 if(mismatches&&(!mode||!strcmp(mode,"strict"))){fprintf(stderr,"DFT differs from local libdfts; use --threads 1 for baseline, --verify report for diagnostic timing, or --skip-verify (unverified).\n");exit(2);}

}
void latency_select(int on){enabled=on;dft=on?parallel_dft:original_dft;idft=on?parallel_idft:original_idft;}
__attribute__((destructor))static void close_pool(void){if(!initialized)return;dft=original_dft;idft=original_idft;atomic_store(&closing,1);atomic_fetch_add_explicit(&sequence,1,memory_order_release);syscall(SYS_futex,&sequence,FUTEX_WAKE_PRIVATE,7,NULL,NULL,0);for(int i=0;i<7;i++)pthread_join(workers[i],NULL);}

static void lane_fft(const int16_t *input,int16_t *output,const int16_t *twiddle,const uint32_t *bitrev,int inverse,int final_scale,int lane,int32_t *data){
#define OAI_RVV_UNPACK_RE(name, packed, vl_)                                  \
  vint32m1_t name = __riscv_vsra_vx_i32m1(                                   \
      __riscv_vsll_vx_i32m1((packed), 16, (vl_)), 16, (vl_))
#define OAI_RVV_UNPACK_IM(name, packed, vl_)                                  \
  vint32m1_t name = __riscv_vsra_vx_i32m1((packed), 16, (vl_))
#define OAI_RVV_FFT32_PACK(value, shift_, vl_)                                \
  __riscv_vmin_vx_i32m1(                                                      \
      __riscv_vmax_vx_i32m1(                                                 \
          __riscv_vsra_vx_i32m1(                                             \
              __riscv_vadd_vx_i32m1(                                         \
                  (value), INT32_C(1) << ((shift_) - 1), (vl_)),              \
              (shift_), (vl_)), INT16_MIN, (vl_)),                            \
      INT16_MAX, (vl_))
#define OAI_RVV_PACK_COMPLEX(re_, im_, vl_)                                   \
  __riscv_vreinterpret_v_u32m1_i32m1(                                        \
      __riscv_vor_vv_u32m1(                                                  \
          __riscv_vand_vx_u32m1(                                             \
              __riscv_vreinterpret_v_i32m1_u32m1((re_)), UINT32_C(0xffff),   \
              (vl_)),                                                        \
          __riscv_vsll_vx_u32m1(                                             \
              __riscv_vreinterpret_v_i32m1_u32m1((im_)), 16, (vl_)),         \
          (vl_)))

  /* Fuse the first two scaled radix-2 passes into one radix-4 pass. */
  {
    const size_t blocks = (lane+1)*64;
    const ptrdiff_t stride = 4 * (ptrdiff_t)sizeof(int32_t);
    size_t block = lane*64;
    while (block < blocks) {
      const size_t vl = __riscv_vsetvl_e32m1(blocks - block);
      int32_t *base = data + 4 * block;
      const uint32_t *rev = bitrev + 4 * block;
      vuint32m1_t i0=__riscv_vlse32_v_u32m1(rev+0,stride,vl);
      vuint32m1_t i1=__riscv_vlse32_v_u32m1(rev+1,stride,vl);
      vuint32m1_t i2=__riscv_vlse32_v_u32m1(rev+2,stride,vl);
      vuint32m1_t i3=__riscv_vlse32_v_u32m1(rev+3,stride,vl);
      i0=__riscv_vsll_vx_u32m1(i0,2,vl); i1=__riscv_vsll_vx_u32m1(i1,2,vl);
      i2=__riscv_vsll_vx_u32m1(i2,2,vl); i3=__riscv_vsll_vx_u32m1(i3,2,vl);
      vint32m1_t x0p=__riscv_vluxei32_v_i32m1((const int32_t *)input,i0,vl);
      vint32m1_t x1p=__riscv_vluxei32_v_i32m1((const int32_t *)input,i1,vl);
      vint32m1_t x2p=__riscv_vluxei32_v_i32m1((const int32_t *)input,i2,vl);
      vint32m1_t x3p=__riscv_vluxei32_v_i32m1((const int32_t *)input,i3,vl);
      OAI_RVV_UNPACK_RE(x0r,x0p,vl); OAI_RVV_UNPACK_IM(x0i,x0p,vl);
      OAI_RVV_UNPACK_RE(x1r,x1p,vl); OAI_RVV_UNPACK_IM(x1i,x1p,vl);
      OAI_RVV_UNPACK_RE(x2r,x2p,vl); OAI_RVV_UNPACK_IM(x2i,x2p,vl);
      OAI_RVV_UNPACK_RE(x3r,x3p,vl); OAI_RVV_UNPACK_IM(x3i,x3p,vl);
#define OAI_RVV_RAD2_FIRST(out_, lhs_, rhs_, op_)                             \
      vint32m1_t out_ = OAI_RVV_FFT32_PACK(                                   \
          __riscv_##op_##_vv_i32m1(                                           \
              __riscv_vmul_vx_i32m1((lhs_),INT16_MAX,vl),                    \
              __riscv_vmul_vx_i32m1((rhs_),INT16_MAX,vl),vl),16,vl)
      OAI_RVV_RAD2_FIRST(p0r,x0r,x1r,vadd); OAI_RVV_RAD2_FIRST(p0i,x0i,x1i,vadd);
      OAI_RVV_RAD2_FIRST(p1r,x0r,x1r,vsub); OAI_RVV_RAD2_FIRST(p1i,x0i,x1i,vsub);
      OAI_RVV_RAD2_FIRST(p2r,x2r,x3r,vadd); OAI_RVV_RAD2_FIRST(p2i,x2i,x3i,vadd);
      OAI_RVV_RAD2_FIRST(p3r,x2r,x3r,vsub); OAI_RVV_RAD2_FIRST(p3i,x2i,x3i,vsub);
#undef OAI_RVV_RAD2_FIRST
      vint32m1_t p0qr=__riscv_vmul_vx_i32m1(p0r,INT16_MAX,vl);
      vint32m1_t p0qi=__riscv_vmul_vx_i32m1(p0i,INT16_MAX,vl);
      vint32m1_t p1qr=__riscv_vmul_vx_i32m1(p1r,INT16_MAX,vl);
      vint32m1_t p1qi=__riscv_vmul_vx_i32m1(p1i,INT16_MAX,vl);
      vint32m1_t p2qr=__riscv_vmul_vx_i32m1(p2r,INT16_MAX,vl);
      vint32m1_t p2qi=__riscv_vmul_vx_i32m1(p2i,INT16_MAX,vl);
      vint32m1_t jp3r=__riscv_vmul_vx_i32m1(p3i,inverse ? -INT16_MAX : INT16_MAX,vl);
      vint32m1_t jp3i=__riscv_vmul_vx_i32m1(p3r,inverse ? INT16_MAX : -INT16_MAX,vl);
      vint32m1_t y0r=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(p0qr,p2qr,vl),16,vl);
      vint32m1_t y0i=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(p0qi,p2qi,vl),16,vl);
      vint32m1_t y2r=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(p0qr,p2qr,vl),16,vl);
      vint32m1_t y2i=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(p0qi,p2qi,vl),16,vl);
      vint32m1_t y1r=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(p1qr,jp3r,vl),16,vl);
      vint32m1_t y1i=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(p1qi,jp3i,vl),16,vl);
      vint32m1_t y3r=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(p1qr,jp3r,vl),16,vl);
      vint32m1_t y3i=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(p1qi,jp3i,vl),16,vl);
      __riscv_vsse32_v_i32m1(base+0,stride,OAI_RVV_PACK_COMPLEX(y0r,y0i,vl),vl);
      __riscv_vsse32_v_i32m1(base+1,stride,OAI_RVV_PACK_COMPLEX(y1r,y1i,vl),vl);
      __riscv_vsse32_v_i32m1(base+2,stride,OAI_RVV_PACK_COMPLEX(y2r,y2i,vl),vl);
      __riscv_vsse32_v_i32m1(base+3,stride,OAI_RVV_PACK_COMPLEX(y3r,y3i,vl),vl);
      block += vl;
    }
  }

  /* Fuse stages three and four.  Each lane handles one independent
   * 16-point group while j selects one of four radix-4 columns. */
  {
    const size_t blocks = (lane+1)*16;
    const ptrdiff_t stride = 16 * (ptrdiff_t)sizeof(int32_t);
    for (unsigned int j = 0; j < 4; ++j) {
      const int32_t w8r = twiddle[2 * j * 256];
      const int32_t w8i0 = twiddle[2 * j * 256 + 1];
      const int32_t w8i = inverse ? -w8i0 : w8i0;
      const int32_t w16ar = twiddle[2 * j * 128];
      const int32_t w16ai0 = twiddle[2 * j * 128 + 1];
      const int32_t w16ai = inverse ? -w16ai0 : w16ai0;
      const int32_t w16br = twiddle[2 * (j + 4) * 128];
      const int32_t w16bi0 = twiddle[2 * (j + 4) * 128 + 1];
      const int32_t w16bi = inverse ? -w16bi0 : w16bi0;
      size_t block = lane*16;
      while (block < blocks) {
        const size_t vl = __riscv_vsetvl_e32m1(blocks - block);
        int32_t *base = data + 16 * block + j;
        vint32m1_t x0p=__riscv_vlse32_v_i32m1(base+0,stride,vl);
        vint32m1_t x1p=__riscv_vlse32_v_i32m1(base+4,stride,vl);
        vint32m1_t x2p=__riscv_vlse32_v_i32m1(base+8,stride,vl);
        vint32m1_t x3p=__riscv_vlse32_v_i32m1(base+12,stride,vl);
        OAI_RVV_UNPACK_RE(x0r,x0p,vl); OAI_RVV_UNPACK_IM(x0i,x0p,vl);
        OAI_RVV_UNPACK_RE(x1r,x1p,vl); OAI_RVV_UNPACK_IM(x1i,x1p,vl);
        OAI_RVV_UNPACK_RE(x2r,x2p,vl); OAI_RVV_UNPACK_IM(x2i,x2p,vl);
        OAI_RVV_UNPACK_RE(x3r,x3p,vl); OAI_RVV_UNPACK_IM(x3i,x3p,vl);
#define OAI_RVV_CMUL_SCALAR(prefix_, xr_, xi_, wr_, wi_)                      \
        vint32m1_t prefix_##r = __riscv_vsub_vv_i32m1(                       \
            __riscv_vmul_vx_i32m1((xr_),(wr_),vl),                           \
            __riscv_vmul_vx_i32m1((xi_),(wi_),vl),vl);                       \
        vint32m1_t prefix_##i = __riscv_vadd_vv_i32m1(                       \
            __riscv_vmul_vx_i32m1((xr_),(wi_),vl),                           \
            __riscv_vmul_vx_i32m1((xi_),(wr_),vl),vl)
#define OAI_RVV_RAD2_Q15(prefix0_,prefix1_,ar_,ai_,tr_,ti_)                   \
        vint32m1_t prefix0_##r=OAI_RVV_FFT32_PACK(                            \
            __riscv_vadd_vv_i32m1(                                           \
                __riscv_vmul_vx_i32m1((ar_),INT16_MAX,vl),(tr_),vl),16,vl);  \
        vint32m1_t prefix0_##i=OAI_RVV_FFT32_PACK(                            \
            __riscv_vadd_vv_i32m1(                                           \
                __riscv_vmul_vx_i32m1((ai_),INT16_MAX,vl),(ti_),vl),16,vl);  \
        vint32m1_t prefix1_##r=OAI_RVV_FFT32_PACK(                            \
            __riscv_vsub_vv_i32m1(                                           \
                __riscv_vmul_vx_i32m1((ar_),INT16_MAX,vl),(tr_),vl),16,vl);  \
        vint32m1_t prefix1_##i=OAI_RVV_FFT32_PACK(                            \
            __riscv_vsub_vv_i32m1(                                           \
                __riscv_vmul_vx_i32m1((ai_),INT16_MAX,vl),(ti_),vl),16,vl)
        OAI_RVV_CMUL_SCALAR(t1,x1r,x1i,w8r,w8i);
        OAI_RVV_CMUL_SCALAR(t3,x3r,x3i,w8r,w8i);
        OAI_RVV_RAD2_Q15(p0,p1,x0r,x0i,t1r,t1i);
        OAI_RVV_RAD2_Q15(p2,p3,x2r,x2i,t3r,t3i);
        OAI_RVV_CMUL_SCALAR(q2,p2r,p2i,w16ar,w16ai);
        OAI_RVV_CMUL_SCALAR(q3,p3r,p3i,w16br,w16bi);
        OAI_RVV_RAD2_Q15(y0,y2,p0r,p0i,q2r,q2i);
        OAI_RVV_RAD2_Q15(y1,y3,p1r,p1i,q3r,q3i);
#undef OAI_RVV_RAD2_Q15
#undef OAI_RVV_CMUL_SCALAR
        __riscv_vsse32_v_i32m1(base+0,stride,OAI_RVV_PACK_COMPLEX(y0r,y0i,vl),vl);
        __riscv_vsse32_v_i32m1(base+4,stride,OAI_RVV_PACK_COMPLEX(y1r,y1i,vl),vl);
        __riscv_vsse32_v_i32m1(base+8,stride,OAI_RVV_PACK_COMPLEX(y2r,y2i,vl),vl);
        __riscv_vsse32_v_i32m1(base+12,stride,OAI_RVV_PACK_COMPLEX(y3r,y3i,vl),vl);
        block += vl;
      }
    }
  }

  /* Stages 5..8 remain inside each lane's independent 256-point block. */
  for(unsigned width=32;width<=256;width*=2){
    unsigned half=width/2,step=2048/width;int shift=width==32?16:15;
    for(unsigned base=lane*256;base<(lane+1)*256;base+=width){
      size_t j=0;
      while(j<half){
        size_t vl=__riscv_vsetvl_e32m1(half-j);int32_t *a=data+base+j,*b=a+half;
        vint32m1_t ap=__riscv_vle32_v_i32m1(a,vl),bp=__riscv_vle32_v_i32m1(b,vl);
        vint32m1_t wp=__riscv_vlse32_v_i32m1((const int32_t*)twiddle+j*step,step*sizeof(int32_t),vl);
        OAI_RVV_UNPACK_RE(ar,ap,vl);OAI_RVV_UNPACK_IM(ai,ap,vl);
        OAI_RVV_UNPACK_RE(br,bp,vl);OAI_RVV_UNPACK_IM(bi,bp,vl);
        OAI_RVV_UNPACK_RE(wr,wp,vl);OAI_RVV_UNPACK_IM(wi0,wp,vl);
        vint32m1_t wi=inverse?__riscv_vneg_v_i32m1(wi0,vl):wi0;
        vint32m1_t tr=__riscv_vsub_vv_i32m1(__riscv_vmul_vv_i32m1(br,wr,vl),__riscv_vmul_vv_i32m1(bi,wi,vl),vl);
        vint32m1_t ti=__riscv_vadd_vv_i32m1(__riscv_vmul_vv_i32m1(br,wi,vl),__riscv_vmul_vv_i32m1(bi,wr,vl),vl);
        vint32m1_t qr=__riscv_vmul_vx_i32m1(ar,INT16_MAX,vl),qi=__riscv_vmul_vx_i32m1(ai,INT16_MAX,vl);
        vint32m1_t o0r=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(qr,tr,vl),shift,vl);
        vint32m1_t o0i=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(qi,ti,vl),shift,vl);
        vint32m1_t o1r=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(qr,tr,vl),shift,vl);
        vint32m1_t o1i=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(qi,ti,vl),shift,vl);
        __riscv_vse32_v_i32m1(a,OAI_RVV_PACK_COMPLEX(o0r,o0i,vl),vl);
        __riscv_vse32_v_i32m1(b,OAI_RVV_PACK_COMPLEX(o1r,o1i,vl),vl);j+=vl;
      }
    }
  }
  fft_barrier(lane,0);
  /* Fuse stages nine and ten.  Each pass consumes four contiguous 256-point
   * quarters and writes the complete 1024-point result only once. */
  for (unsigned int base = 0; base < 2048; base += 1024) {
    size_t j = lane*32;
    while (j < (lane+1)*32) {
      const size_t vl = __riscv_vsetvl_e32m1((lane+1)*32 - j);
      int32_t *p = data + base + j;
      vint32m1_t x0p = __riscv_vle32_v_i32m1(p + 0, vl);
      vint32m1_t x1p = __riscv_vle32_v_i32m1(p + 256, vl);
      vint32m1_t x2p = __riscv_vle32_v_i32m1(p + 512, vl);
      vint32m1_t x3p = __riscv_vle32_v_i32m1(p + 768, vl);
      const int32_t *tw32 = (const int32_t *)twiddle;
      vint32m1_t w9p = __riscv_vlse32_v_i32m1(
          tw32 + 4 * j, 4 * (ptrdiff_t)sizeof(int32_t), vl);
      vint32m1_t w10ap = __riscv_vlse32_v_i32m1(
          tw32 + 2 * j, 2 * (ptrdiff_t)sizeof(int32_t), vl);
      vint32m1_t w10bp = __riscv_vlse32_v_i32m1(
          tw32 + 2 * (j + 256), 2 * (ptrdiff_t)sizeof(int32_t), vl);
      OAI_RVV_UNPACK_RE(x0r,x0p,vl); OAI_RVV_UNPACK_IM(x0i,x0p,vl);
      OAI_RVV_UNPACK_RE(x1r,x1p,vl); OAI_RVV_UNPACK_IM(x1i,x1p,vl);
      OAI_RVV_UNPACK_RE(x2r,x2p,vl); OAI_RVV_UNPACK_IM(x2i,x2p,vl);
      OAI_RVV_UNPACK_RE(x3r,x3p,vl); OAI_RVV_UNPACK_IM(x3i,x3p,vl);
      OAI_RVV_UNPACK_RE(w9r,w9p,vl); OAI_RVV_UNPACK_IM(w9i0,w9p,vl);
      OAI_RVV_UNPACK_RE(w10ar,w10ap,vl); OAI_RVV_UNPACK_IM(w10ai0,w10ap,vl);
      OAI_RVV_UNPACK_RE(w10br,w10bp,vl); OAI_RVV_UNPACK_IM(w10bi0,w10bp,vl);
      vint32m1_t w9i=inverse?__riscv_vneg_v_i32m1(w9i0,vl):w9i0;
      vint32m1_t w10ai=inverse?__riscv_vneg_v_i32m1(w10ai0,vl):w10ai0;
      vint32m1_t w10bi=inverse?__riscv_vneg_v_i32m1(w10bi0,vl):w10bi0;
#define OAI_RVV_CMUL_V(prefix_,xr_,xi_,wr_,wi_)                              \
      vint32m1_t prefix_##r=__riscv_vsub_vv_i32m1(                           \
          __riscv_vmul_vv_i32m1((xr_),(wr_),vl),                             \
          __riscv_vmul_vv_i32m1((xi_),(wi_),vl),vl);                         \
      vint32m1_t prefix_##i=__riscv_vadd_vv_i32m1(                           \
          __riscv_vmul_vv_i32m1((xr_),(wi_),vl),                             \
          __riscv_vmul_vv_i32m1((xi_),(wr_),vl),vl)
#define OAI_RVV_RAD2_V(prefix0_,prefix1_,ar_,ai_,tr_,ti_)                    \
      vint32m1_t prefix0_##r=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(      \
          __riscv_vmul_vx_i32m1((ar_),INT16_MAX,vl),(tr_),vl),15,vl);       \
      vint32m1_t prefix0_##i=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(      \
          __riscv_vmul_vx_i32m1((ai_),INT16_MAX,vl),(ti_),vl),15,vl);       \
      vint32m1_t prefix1_##r=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(      \
          __riscv_vmul_vx_i32m1((ar_),INT16_MAX,vl),(tr_),vl),15,vl);       \
      vint32m1_t prefix1_##i=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(      \
          __riscv_vmul_vx_i32m1((ai_),INT16_MAX,vl),(ti_),vl),15,vl)
      OAI_RVV_CMUL_V(t1,x1r,x1i,w9r,w9i);
      OAI_RVV_CMUL_V(t3,x3r,x3i,w9r,w9i);
      OAI_RVV_RAD2_V(a0,a1,x0r,x0i,t1r,t1i);
      OAI_RVV_RAD2_V(a2,a3,x2r,x2i,t3r,t3i);
      OAI_RVV_CMUL_V(u2,a2r,a2i,w10ar,w10ai);
      OAI_RVV_CMUL_V(u3,a3r,a3i,w10br,w10bi);
      OAI_RVV_RAD2_V(y0,y2,a0r,a0i,u2r,u2i);
      OAI_RVV_RAD2_V(y1,y3,a1r,a1i,u3r,u3i);
#undef OAI_RVV_RAD2_V
#undef OAI_RVV_CMUL_V
      __riscv_vse32_v_i32m1(p+0,OAI_RVV_PACK_COMPLEX(y0r,y0i,vl),vl);
      __riscv_vse32_v_i32m1(p+256,OAI_RVV_PACK_COMPLEX(y1r,y1i,vl),vl);
      __riscv_vse32_v_i32m1(p+512,OAI_RVV_PACK_COMPLEX(y2r,y2i,vl),vl);
      __riscv_vse32_v_i32m1(p+768,OAI_RVV_PACK_COMPLEX(y3r,y3i,vl),vl);
      j += vl;
    }
  }

  fft_barrier(lane,1);
  /* Final radix-2 stage and optional 1/sqrt(2) scaling write directly to the
   * caller's output, avoiding a full temporary-array store/load round trip. */
  {
    size_t j = lane*128;
    while (j < (lane+1)*128) {
      const size_t vl = __riscv_vsetvl_e32m1((lane+1)*128 - j);
      vint32m1_t ap=__riscv_vle32_v_i32m1(data+j,vl);
      vint32m1_t bp=__riscv_vle32_v_i32m1(data+1024+j,vl);
      vint32m1_t tp=__riscv_vle32_v_i32m1((const int32_t *)twiddle+j,vl);
      OAI_RVV_UNPACK_RE(ar,ap,vl); OAI_RVV_UNPACK_IM(ai,ap,vl);
      OAI_RVV_UNPACK_RE(br,bp,vl); OAI_RVV_UNPACK_IM(bi,bp,vl);
      OAI_RVV_UNPACK_RE(wr,tp,vl); OAI_RVV_UNPACK_IM(wi0,tp,vl);
      vint32m1_t wi=inverse?__riscv_vneg_v_i32m1(wi0,vl):wi0;
      vint32m1_t tr=__riscv_vsub_vv_i32m1(
          __riscv_vmul_vv_i32m1(br,wr,vl),__riscv_vmul_vv_i32m1(bi,wi,vl),vl);
      vint32m1_t ti=__riscv_vadd_vv_i32m1(
          __riscv_vmul_vv_i32m1(br,wi,vl),__riscv_vmul_vv_i32m1(bi,wr,vl),vl);
      vint32m1_t aqr=__riscv_vmul_vx_i32m1(ar,INT16_MAX,vl);
      vint32m1_t aqi=__riscv_vmul_vx_i32m1(ai,INT16_MAX,vl);
      vint32m1_t o0r=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(aqr,tr,vl),15,vl);
      vint32m1_t o0i=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(aqi,ti,vl),15,vl);
      vint32m1_t o1r=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(aqr,tr,vl),15,vl);
      vint32m1_t o1i=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(aqi,ti,vl),15,vl);
      if (final_scale) {
#define OAI_RVV_FINAL_SCALE(v_)                                               \
        __riscv_vmin_vx_i32m1(__riscv_vmax_vx_i32m1(                         \
            __riscv_vsra_vx_i32m1(__riscv_vadd_vx_i32m1(                    \
                __riscv_vmul_vx_i32m1((v_),23170,vl),16384,vl),15,vl),      \
            INT16_MIN,vl),INT16_MAX,vl)
        o0r=OAI_RVV_FINAL_SCALE(o0r); o0i=OAI_RVV_FINAL_SCALE(o0i);
        o1r=OAI_RVV_FINAL_SCALE(o1r); o1i=OAI_RVV_FINAL_SCALE(o1i);
#undef OAI_RVV_FINAL_SCALE
      }
      __riscv_vse32_v_i32m1((int32_t *)output+j,
          OAI_RVV_PACK_COMPLEX(o0r,o0i,vl),vl);
      __riscv_vse32_v_i32m1((int32_t *)output+1024+j,
          OAI_RVV_PACK_COMPLEX(o1r,o1i,vl),vl);
      j += vl;
    }
  }
#undef OAI_RVV_UNPACK_RE
#undef OAI_RVV_UNPACK_IM
#undef OAI_RVV_FFT32_PACK
#undef OAI_RVV_PACK_COMPLEX
}
static void serial_optimized_fft(const int16_t *input,int16_t *output,const int16_t *twiddle,const uint32_t *bitrev,int inverse,int final_scale){
int32_t data[2048] __attribute__((aligned(128)));
#define OAI_RVV_UNPACK_RE(name, packed, vl_)                                  \
  vint32m1_t name = __riscv_vsra_vx_i32m1(                                   \
      __riscv_vsll_vx_i32m1((packed), 16, (vl_)), 16, (vl_))
#define OAI_RVV_UNPACK_IM(name, packed, vl_)                                  \
  vint32m1_t name = __riscv_vsra_vx_i32m1((packed), 16, (vl_))
#define OAI_RVV_FFT32_PACK(value, shift_, vl_)                                \
  __riscv_vmin_vx_i32m1(                                                      \
      __riscv_vmax_vx_i32m1(                                                 \
          __riscv_vsra_vx_i32m1(                                             \
              __riscv_vadd_vx_i32m1(                                         \
                  (value), INT32_C(1) << ((shift_) - 1), (vl_)),              \
              (shift_), (vl_)), INT16_MIN, (vl_)),                            \
      INT16_MAX, (vl_))
#define OAI_RVV_PACK_COMPLEX(re_, im_, vl_)                                   \
  __riscv_vreinterpret_v_u32m1_i32m1(                                        \
      __riscv_vor_vv_u32m1(                                                  \
          __riscv_vand_vx_u32m1(                                             \
              __riscv_vreinterpret_v_i32m1_u32m1((re_)), UINT32_C(0xffff),   \
              (vl_)),                                                        \
          __riscv_vsll_vx_u32m1(                                             \
              __riscv_vreinterpret_v_i32m1_u32m1((im_)), 16, (vl_)),         \
          (vl_)))

  /* Fuse the first two scaled radix-2 passes into one radix-4 pass. */
  {
    const size_t blocks = 512;
    const ptrdiff_t stride = 4 * (ptrdiff_t)sizeof(int32_t);
    size_t block = 0;
    while (block < blocks) {
      const size_t vl = __riscv_vsetvl_e32m1(blocks - block);
      int32_t *base = data + 4 * block;
      const uint32_t *rev = bitrev + 4 * block;
      vuint32m1_t i0=__riscv_vlse32_v_u32m1(rev+0,stride,vl);
      vuint32m1_t i1=__riscv_vlse32_v_u32m1(rev+1,stride,vl);
      vuint32m1_t i2=__riscv_vlse32_v_u32m1(rev+2,stride,vl);
      vuint32m1_t i3=__riscv_vlse32_v_u32m1(rev+3,stride,vl);
      i0=__riscv_vsll_vx_u32m1(i0,2,vl); i1=__riscv_vsll_vx_u32m1(i1,2,vl);
      i2=__riscv_vsll_vx_u32m1(i2,2,vl); i3=__riscv_vsll_vx_u32m1(i3,2,vl);
      vint32m1_t x0p=__riscv_vluxei32_v_i32m1((const int32_t *)input,i0,vl);
      vint32m1_t x1p=__riscv_vluxei32_v_i32m1((const int32_t *)input,i1,vl);
      vint32m1_t x2p=__riscv_vluxei32_v_i32m1((const int32_t *)input,i2,vl);
      vint32m1_t x3p=__riscv_vluxei32_v_i32m1((const int32_t *)input,i3,vl);
      OAI_RVV_UNPACK_RE(x0r,x0p,vl); OAI_RVV_UNPACK_IM(x0i,x0p,vl);
      OAI_RVV_UNPACK_RE(x1r,x1p,vl); OAI_RVV_UNPACK_IM(x1i,x1p,vl);
      OAI_RVV_UNPACK_RE(x2r,x2p,vl); OAI_RVV_UNPACK_IM(x2i,x2p,vl);
      OAI_RVV_UNPACK_RE(x3r,x3p,vl); OAI_RVV_UNPACK_IM(x3i,x3p,vl);
#define OAI_RVV_RAD2_FIRST(out_, lhs_, rhs_, op_)                             \
      vint32m1_t out_ = OAI_RVV_FFT32_PACK(                                   \
          __riscv_##op_##_vv_i32m1(                                           \
              __riscv_vmul_vx_i32m1((lhs_),INT16_MAX,vl),                    \
              __riscv_vmul_vx_i32m1((rhs_),INT16_MAX,vl),vl),16,vl)
      OAI_RVV_RAD2_FIRST(p0r,x0r,x1r,vadd); OAI_RVV_RAD2_FIRST(p0i,x0i,x1i,vadd);
      OAI_RVV_RAD2_FIRST(p1r,x0r,x1r,vsub); OAI_RVV_RAD2_FIRST(p1i,x0i,x1i,vsub);
      OAI_RVV_RAD2_FIRST(p2r,x2r,x3r,vadd); OAI_RVV_RAD2_FIRST(p2i,x2i,x3i,vadd);
      OAI_RVV_RAD2_FIRST(p3r,x2r,x3r,vsub); OAI_RVV_RAD2_FIRST(p3i,x2i,x3i,vsub);
#undef OAI_RVV_RAD2_FIRST
      vint32m1_t p0qr=__riscv_vmul_vx_i32m1(p0r,INT16_MAX,vl);
      vint32m1_t p0qi=__riscv_vmul_vx_i32m1(p0i,INT16_MAX,vl);
      vint32m1_t p1qr=__riscv_vmul_vx_i32m1(p1r,INT16_MAX,vl);
      vint32m1_t p1qi=__riscv_vmul_vx_i32m1(p1i,INT16_MAX,vl);
      vint32m1_t p2qr=__riscv_vmul_vx_i32m1(p2r,INT16_MAX,vl);
      vint32m1_t p2qi=__riscv_vmul_vx_i32m1(p2i,INT16_MAX,vl);
      vint32m1_t jp3r=__riscv_vmul_vx_i32m1(p3i,inverse ? -INT16_MAX : INT16_MAX,vl);
      vint32m1_t jp3i=__riscv_vmul_vx_i32m1(p3r,inverse ? INT16_MAX : -INT16_MAX,vl);
      vint32m1_t y0r=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(p0qr,p2qr,vl),16,vl);
      vint32m1_t y0i=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(p0qi,p2qi,vl),16,vl);
      vint32m1_t y2r=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(p0qr,p2qr,vl),16,vl);
      vint32m1_t y2i=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(p0qi,p2qi,vl),16,vl);
      vint32m1_t y1r=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(p1qr,jp3r,vl),16,vl);
      vint32m1_t y1i=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(p1qi,jp3i,vl),16,vl);
      vint32m1_t y3r=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(p1qr,jp3r,vl),16,vl);
      vint32m1_t y3i=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(p1qi,jp3i,vl),16,vl);
      __riscv_vsse32_v_i32m1(base+0,stride,OAI_RVV_PACK_COMPLEX(y0r,y0i,vl),vl);
      __riscv_vsse32_v_i32m1(base+1,stride,OAI_RVV_PACK_COMPLEX(y1r,y1i,vl),vl);
      __riscv_vsse32_v_i32m1(base+2,stride,OAI_RVV_PACK_COMPLEX(y2r,y2i,vl),vl);
      __riscv_vsse32_v_i32m1(base+3,stride,OAI_RVV_PACK_COMPLEX(y3r,y3i,vl),vl);
      block += vl;
    }
  }

  /* Fuse stages three and four.  Each lane handles one independent
   * 16-point group while j selects one of four radix-4 columns. */
  {
    const size_t blocks = 128;
    const ptrdiff_t stride = 16 * (ptrdiff_t)sizeof(int32_t);
    for (unsigned int j = 0; j < 4; ++j) {
      const int32_t w8r = twiddle[2 * j * 256];
      const int32_t w8i0 = twiddle[2 * j * 256 + 1];
      const int32_t w8i = inverse ? -w8i0 : w8i0;
      const int32_t w16ar = twiddle[2 * j * 128];
      const int32_t w16ai0 = twiddle[2 * j * 128 + 1];
      const int32_t w16ai = inverse ? -w16ai0 : w16ai0;
      const int32_t w16br = twiddle[2 * (j + 4) * 128];
      const int32_t w16bi0 = twiddle[2 * (j + 4) * 128 + 1];
      const int32_t w16bi = inverse ? -w16bi0 : w16bi0;
      size_t block = 0;
      while (block < blocks) {
        const size_t vl = __riscv_vsetvl_e32m1(blocks - block);
        int32_t *base = data + 16 * block + j;
        vint32m1_t x0p=__riscv_vlse32_v_i32m1(base+0,stride,vl);
        vint32m1_t x1p=__riscv_vlse32_v_i32m1(base+4,stride,vl);
        vint32m1_t x2p=__riscv_vlse32_v_i32m1(base+8,stride,vl);
        vint32m1_t x3p=__riscv_vlse32_v_i32m1(base+12,stride,vl);
        OAI_RVV_UNPACK_RE(x0r,x0p,vl); OAI_RVV_UNPACK_IM(x0i,x0p,vl);
        OAI_RVV_UNPACK_RE(x1r,x1p,vl); OAI_RVV_UNPACK_IM(x1i,x1p,vl);
        OAI_RVV_UNPACK_RE(x2r,x2p,vl); OAI_RVV_UNPACK_IM(x2i,x2p,vl);
        OAI_RVV_UNPACK_RE(x3r,x3p,vl); OAI_RVV_UNPACK_IM(x3i,x3p,vl);
#define OAI_RVV_CMUL_SCALAR(prefix_, xr_, xi_, wr_, wi_)                      \
        vint32m1_t prefix_##r = __riscv_vsub_vv_i32m1(                       \
            __riscv_vmul_vx_i32m1((xr_),(wr_),vl),                           \
            __riscv_vmul_vx_i32m1((xi_),(wi_),vl),vl);                       \
        vint32m1_t prefix_##i = __riscv_vadd_vv_i32m1(                       \
            __riscv_vmul_vx_i32m1((xr_),(wi_),vl),                           \
            __riscv_vmul_vx_i32m1((xi_),(wr_),vl),vl)
#define OAI_RVV_RAD2_Q15(prefix0_,prefix1_,ar_,ai_,tr_,ti_)                   \
        vint32m1_t prefix0_##r=OAI_RVV_FFT32_PACK(                            \
            __riscv_vadd_vv_i32m1(                                           \
                __riscv_vmul_vx_i32m1((ar_),INT16_MAX,vl),(tr_),vl),16,vl);  \
        vint32m1_t prefix0_##i=OAI_RVV_FFT32_PACK(                            \
            __riscv_vadd_vv_i32m1(                                           \
                __riscv_vmul_vx_i32m1((ai_),INT16_MAX,vl),(ti_),vl),16,vl);  \
        vint32m1_t prefix1_##r=OAI_RVV_FFT32_PACK(                            \
            __riscv_vsub_vv_i32m1(                                           \
                __riscv_vmul_vx_i32m1((ar_),INT16_MAX,vl),(tr_),vl),16,vl);  \
        vint32m1_t prefix1_##i=OAI_RVV_FFT32_PACK(                            \
            __riscv_vsub_vv_i32m1(                                           \
                __riscv_vmul_vx_i32m1((ai_),INT16_MAX,vl),(ti_),vl),16,vl)
        OAI_RVV_CMUL_SCALAR(t1,x1r,x1i,w8r,w8i);
        OAI_RVV_CMUL_SCALAR(t3,x3r,x3i,w8r,w8i);
        OAI_RVV_RAD2_Q15(p0,p1,x0r,x0i,t1r,t1i);
        OAI_RVV_RAD2_Q15(p2,p3,x2r,x2i,t3r,t3i);
        OAI_RVV_CMUL_SCALAR(q2,p2r,p2i,w16ar,w16ai);
        OAI_RVV_CMUL_SCALAR(q3,p3r,p3i,w16br,w16bi);
        OAI_RVV_RAD2_Q15(y0,y2,p0r,p0i,q2r,q2i);
        OAI_RVV_RAD2_Q15(y1,y3,p1r,p1i,q3r,q3i);
#undef OAI_RVV_RAD2_Q15
#undef OAI_RVV_CMUL_SCALAR
        __riscv_vsse32_v_i32m1(base+0,stride,OAI_RVV_PACK_COMPLEX(y0r,y0i,vl),vl);
        __riscv_vsse32_v_i32m1(base+4,stride,OAI_RVV_PACK_COMPLEX(y1r,y1i,vl),vl);
        __riscv_vsse32_v_i32m1(base+8,stride,OAI_RVV_PACK_COMPLEX(y2r,y2i,vl),vl);
        __riscv_vsse32_v_i32m1(base+12,stride,OAI_RVV_PACK_COMPLEX(y3r,y3i,vl),vl);
        block += vl;
      }
    }
  }

  /* Stages 5..8 remain inside each lane's independent 256-point block. */
  for(unsigned width=32;width<=256;width*=2){
    unsigned half=width/2,step=2048/width;int shift=width==32?16:15;
    for(unsigned base=0;base<2048;base+=width){
      size_t j=0;
      while(j<half){
        size_t vl=__riscv_vsetvl_e32m1(half-j);int32_t *a=data+base+j,*b=a+half;
        vint32m1_t ap=__riscv_vle32_v_i32m1(a,vl),bp=__riscv_vle32_v_i32m1(b,vl);
        vint32m1_t wp=__riscv_vlse32_v_i32m1((const int32_t*)twiddle+j*step,step*sizeof(int32_t),vl);
        OAI_RVV_UNPACK_RE(ar,ap,vl);OAI_RVV_UNPACK_IM(ai,ap,vl);
        OAI_RVV_UNPACK_RE(br,bp,vl);OAI_RVV_UNPACK_IM(bi,bp,vl);
        OAI_RVV_UNPACK_RE(wr,wp,vl);OAI_RVV_UNPACK_IM(wi0,wp,vl);
        vint32m1_t wi=inverse?__riscv_vneg_v_i32m1(wi0,vl):wi0;
        vint32m1_t tr=__riscv_vsub_vv_i32m1(__riscv_vmul_vv_i32m1(br,wr,vl),__riscv_vmul_vv_i32m1(bi,wi,vl),vl);
        vint32m1_t ti=__riscv_vadd_vv_i32m1(__riscv_vmul_vv_i32m1(br,wi,vl),__riscv_vmul_vv_i32m1(bi,wr,vl),vl);
        vint32m1_t qr=__riscv_vmul_vx_i32m1(ar,INT16_MAX,vl),qi=__riscv_vmul_vx_i32m1(ai,INT16_MAX,vl);
        vint32m1_t o0r=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(qr,tr,vl),shift,vl);
        vint32m1_t o0i=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(qi,ti,vl),shift,vl);
        vint32m1_t o1r=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(qr,tr,vl),shift,vl);
        vint32m1_t o1i=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(qi,ti,vl),shift,vl);
        __riscv_vse32_v_i32m1(a,OAI_RVV_PACK_COMPLEX(o0r,o0i,vl),vl);
        __riscv_vse32_v_i32m1(b,OAI_RVV_PACK_COMPLEX(o1r,o1i,vl),vl);j+=vl;
      }
    }
  }

  /* Fuse stages nine and ten.  Each pass consumes four contiguous 256-point
   * quarters and writes the complete 1024-point result only once. */
  for (unsigned int base = 0; base < 2048; base += 1024) {
    size_t j = 0;
    while (j < 256) {
      const size_t vl = __riscv_vsetvl_e32m1(256 - j);
      int32_t *p = data + base + j;
      vint32m1_t x0p = __riscv_vle32_v_i32m1(p + 0, vl);
      vint32m1_t x1p = __riscv_vle32_v_i32m1(p + 256, vl);
      vint32m1_t x2p = __riscv_vle32_v_i32m1(p + 512, vl);
      vint32m1_t x3p = __riscv_vle32_v_i32m1(p + 768, vl);
      const int32_t *tw32 = (const int32_t *)twiddle;
      vint32m1_t w9p = __riscv_vlse32_v_i32m1(
          tw32 + 4 * j, 4 * (ptrdiff_t)sizeof(int32_t), vl);
      vint32m1_t w10ap = __riscv_vlse32_v_i32m1(
          tw32 + 2 * j, 2 * (ptrdiff_t)sizeof(int32_t), vl);
      vint32m1_t w10bp = __riscv_vlse32_v_i32m1(
          tw32 + 2 * (j + 256), 2 * (ptrdiff_t)sizeof(int32_t), vl);
      OAI_RVV_UNPACK_RE(x0r,x0p,vl); OAI_RVV_UNPACK_IM(x0i,x0p,vl);
      OAI_RVV_UNPACK_RE(x1r,x1p,vl); OAI_RVV_UNPACK_IM(x1i,x1p,vl);
      OAI_RVV_UNPACK_RE(x2r,x2p,vl); OAI_RVV_UNPACK_IM(x2i,x2p,vl);
      OAI_RVV_UNPACK_RE(x3r,x3p,vl); OAI_RVV_UNPACK_IM(x3i,x3p,vl);
      OAI_RVV_UNPACK_RE(w9r,w9p,vl); OAI_RVV_UNPACK_IM(w9i0,w9p,vl);
      OAI_RVV_UNPACK_RE(w10ar,w10ap,vl); OAI_RVV_UNPACK_IM(w10ai0,w10ap,vl);
      OAI_RVV_UNPACK_RE(w10br,w10bp,vl); OAI_RVV_UNPACK_IM(w10bi0,w10bp,vl);
      vint32m1_t w9i=inverse?__riscv_vneg_v_i32m1(w9i0,vl):w9i0;
      vint32m1_t w10ai=inverse?__riscv_vneg_v_i32m1(w10ai0,vl):w10ai0;
      vint32m1_t w10bi=inverse?__riscv_vneg_v_i32m1(w10bi0,vl):w10bi0;
#define OAI_RVV_CMUL_V(prefix_,xr_,xi_,wr_,wi_)                              \
      vint32m1_t prefix_##r=__riscv_vsub_vv_i32m1(                           \
          __riscv_vmul_vv_i32m1((xr_),(wr_),vl),                             \
          __riscv_vmul_vv_i32m1((xi_),(wi_),vl),vl);                         \
      vint32m1_t prefix_##i=__riscv_vadd_vv_i32m1(                           \
          __riscv_vmul_vv_i32m1((xr_),(wi_),vl),                             \
          __riscv_vmul_vv_i32m1((xi_),(wr_),vl),vl)
#define OAI_RVV_RAD2_V(prefix0_,prefix1_,ar_,ai_,tr_,ti_)                    \
      vint32m1_t prefix0_##r=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(      \
          __riscv_vmul_vx_i32m1((ar_),INT16_MAX,vl),(tr_),vl),15,vl);       \
      vint32m1_t prefix0_##i=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(      \
          __riscv_vmul_vx_i32m1((ai_),INT16_MAX,vl),(ti_),vl),15,vl);       \
      vint32m1_t prefix1_##r=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(      \
          __riscv_vmul_vx_i32m1((ar_),INT16_MAX,vl),(tr_),vl),15,vl);       \
      vint32m1_t prefix1_##i=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(      \
          __riscv_vmul_vx_i32m1((ai_),INT16_MAX,vl),(ti_),vl),15,vl)
      OAI_RVV_CMUL_V(t1,x1r,x1i,w9r,w9i);
      OAI_RVV_CMUL_V(t3,x3r,x3i,w9r,w9i);
      OAI_RVV_RAD2_V(a0,a1,x0r,x0i,t1r,t1i);
      OAI_RVV_RAD2_V(a2,a3,x2r,x2i,t3r,t3i);
      OAI_RVV_CMUL_V(u2,a2r,a2i,w10ar,w10ai);
      OAI_RVV_CMUL_V(u3,a3r,a3i,w10br,w10bi);
      OAI_RVV_RAD2_V(y0,y2,a0r,a0i,u2r,u2i);
      OAI_RVV_RAD2_V(y1,y3,a1r,a1i,u3r,u3i);
#undef OAI_RVV_RAD2_V
#undef OAI_RVV_CMUL_V
      __riscv_vse32_v_i32m1(p+0,OAI_RVV_PACK_COMPLEX(y0r,y0i,vl),vl);
      __riscv_vse32_v_i32m1(p+256,OAI_RVV_PACK_COMPLEX(y1r,y1i,vl),vl);
      __riscv_vse32_v_i32m1(p+512,OAI_RVV_PACK_COMPLEX(y2r,y2i,vl),vl);
      __riscv_vse32_v_i32m1(p+768,OAI_RVV_PACK_COMPLEX(y3r,y3i,vl),vl);
      j += vl;
    }
  }

  /* Final radix-2 stage and optional 1/sqrt(2) scaling write directly to the
   * caller's output, avoiding a full temporary-array store/load round trip. */
  {
    size_t j = 0;
    while (j < 1024) {
      const size_t vl = __riscv_vsetvl_e32m1(1024 - j);
      vint32m1_t ap=__riscv_vle32_v_i32m1(data+j,vl);
      vint32m1_t bp=__riscv_vle32_v_i32m1(data+1024+j,vl);
      vint32m1_t tp=__riscv_vle32_v_i32m1((const int32_t *)twiddle+j,vl);
      OAI_RVV_UNPACK_RE(ar,ap,vl); OAI_RVV_UNPACK_IM(ai,ap,vl);
      OAI_RVV_UNPACK_RE(br,bp,vl); OAI_RVV_UNPACK_IM(bi,bp,vl);
      OAI_RVV_UNPACK_RE(wr,tp,vl); OAI_RVV_UNPACK_IM(wi0,tp,vl);
      vint32m1_t wi=inverse?__riscv_vneg_v_i32m1(wi0,vl):wi0;
      vint32m1_t tr=__riscv_vsub_vv_i32m1(
          __riscv_vmul_vv_i32m1(br,wr,vl),__riscv_vmul_vv_i32m1(bi,wi,vl),vl);
      vint32m1_t ti=__riscv_vadd_vv_i32m1(
          __riscv_vmul_vv_i32m1(br,wi,vl),__riscv_vmul_vv_i32m1(bi,wr,vl),vl);
      vint32m1_t aqr=__riscv_vmul_vx_i32m1(ar,INT16_MAX,vl);
      vint32m1_t aqi=__riscv_vmul_vx_i32m1(ai,INT16_MAX,vl);
      vint32m1_t o0r=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(aqr,tr,vl),15,vl);
      vint32m1_t o0i=OAI_RVV_FFT32_PACK(__riscv_vadd_vv_i32m1(aqi,ti,vl),15,vl);
      vint32m1_t o1r=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(aqr,tr,vl),15,vl);
      vint32m1_t o1i=OAI_RVV_FFT32_PACK(__riscv_vsub_vv_i32m1(aqi,ti,vl),15,vl);
      if (final_scale) {
#define OAI_RVV_FINAL_SCALE(v_)                                               \
        __riscv_vmin_vx_i32m1(__riscv_vmax_vx_i32m1(                         \
            __riscv_vsra_vx_i32m1(__riscv_vadd_vx_i32m1(                    \
                __riscv_vmul_vx_i32m1((v_),23170,vl),16384,vl),15,vl),      \
            INT16_MIN,vl),INT16_MAX,vl)
        o0r=OAI_RVV_FINAL_SCALE(o0r); o0i=OAI_RVV_FINAL_SCALE(o0i);
        o1r=OAI_RVV_FINAL_SCALE(o1r); o1i=OAI_RVV_FINAL_SCALE(o1i);
#undef OAI_RVV_FINAL_SCALE
      }
      __riscv_vse32_v_i32m1((int32_t *)output+j,
          OAI_RVV_PACK_COMPLEX(o0r,o0i,vl),vl);
      __riscv_vse32_v_i32m1((int32_t *)output+1024+j,
          OAI_RVV_PACK_COMPLEX(o1r,o1i,vl),vl);
      j += vl;
    }
  }
#undef OAI_RVV_UNPACK_RE
#undef OAI_RVV_UNPACK_IM
#undef OAI_RVV_FFT32_PACK
#undef OAI_RVV_PACK_COMPLEX
}