// tb_mc_gmem.cpp (example)
#include "rv32i_pp_ip.h"

// Prototype from the new top:
extern "C" void rv32i_pp_ip_mc_gmem(
  unsigned int  ip_num,
  unsigned int  start_pc,            // word index
  volatile unsigned int *gmem,       // unified AXI mem
  unsigned int  code_base_words,     // word base for code
  unsigned int  data_base_words,     // word base for data
  unsigned int *nb_instruction,
  unsigned int *nb_cycle);

int main(){
  // build a unified memory big enough for code + data
  static unsigned int gmem[CODE_RAM_SIZE + DATA_RAM_SIZE + 1024];

  // Choose a simple layout: code first, then data (both in WORDS):
  const unsigned code_base_words = 0;
  const unsigned data_base_words = CODE_RAM_SIZE;

  // Fill code image into gmem[code_base_words + i]
  // Fill input data into  (int*)&gmem[data_base_words]

  unsigned nbi=0, nbc=0;
  unsigned start_pc = 0; // word index from the code base

  rv32i_pp_ip_mc_gmem(/*ip_num*/0,
                       start_pc,
                       gmem,
                       code_base_words,
                       data_base_words,
                       &nbi, &nbc);

  // Check results in (int*)&gmem[data_base_words]
  return 0;
}
