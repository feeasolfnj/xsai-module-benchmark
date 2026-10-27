/* Standalone codec simulations may free a gNB without loading SRK plugins. */
#include <stdbool.h>
#include "PHY/defs_nr_common.h"

static bool plugins_initialized;
void __real_init_plugins(const NR_DL_FRAME_PARMS *fp);
void __real_free_plugins(void);

void __wrap_init_plugins(const NR_DL_FRAME_PARMS *fp)
{
  __real_init_plugins(fp);
  plugins_initialized = true;
}

void __wrap_free_plugins(void)
{
  if (plugins_initialized) {
    plugins_initialized = false;
    __real_free_plugins();
  }
}
