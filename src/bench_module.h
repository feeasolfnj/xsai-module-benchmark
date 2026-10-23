#include <stdint.h>
#include <stddef.h>
typedef struct{const char*name,*contract;int(*setup)(size_t,int);void(*run)(int16_t*,int16_t*,unsigned char);int approximate;}bench_module;
extern const bench_module selected_module,reference_module;
