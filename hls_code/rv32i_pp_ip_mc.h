#pragma once
#include "rv32i_pp_ip.h"

#ifdef __cplusplus
extern "C" {
#endif

void rv32i_pp_ip_mc(
  unsigned int  ip_num,                              // core index (optional to write into a0/x10)
  unsigned int  start_pc,
  unsigned int  ip_code_ram[CODE_RAM_SIZE],          // per-core IMEM (BRAM)
  int           ip_data_ram[DATA_RAM_SIZE],          // per-core DMEM (BRAM)
  unsigned int *nb_instruction,
  unsigned int *nb_cycle);

#ifdef __cplusplus
}
#endif
