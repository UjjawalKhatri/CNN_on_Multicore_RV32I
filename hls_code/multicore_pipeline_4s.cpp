// rv32pp_multicore_ip.cpp
#include "multicore_pipeline_4s.h"

static inline bool is_local_addr(u32 byte_addr) {
  return byte_addr < IP_DATA_RAM_SIZE;
}
static inline u32  local_index(u32 byte_addr) { return (byte_addr >> 2); } // word index

// HLS TOP
extern "C" {
void multicore_pipeline_4s(
    u32                 ip_num,                        // AXI-Lite
    volatile u32        code_ram[IP_CODE_RAM_WORDS],   // BRAM (IMEM)
    volatile u32        ip_data_ram[IP_DATA_RAM_SIZE/4],// BRAM (DMEM local)
    volatile u32       *data_ram,                      // M_AXI (DMEM global base)
    u32                *nb_instruction,                // AXI-Lite (optional counters)
    u32                *nb_cycle                       // AXI-Lite (optional counters)
){
#pragma HLS INTERFACE s_axilite port=ip_num        bundle=CTRL
#pragma HLS INTERFACE bram      port=code_ram
#pragma HLS INTERFACE bram      port=ip_data_ram
#pragma HLS INTERFACE m_axi     port=data_ram      offset=slave bundle=DATA depth=1024
#pragma HLS INTERFACE s_axilite port=nb_instruction bundle=CTRL
#pragma HLS INTERFACE s_axilite port=nb_cycle       bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return         bundle=CTRL

  // ----------------- Helpers used by MEM stage -----------------
  auto ld_word = [&](u32 byte_addr)->u32 {
#pragma HLS INLINE
    if (is_local_addr(byte_addr)) {
      return ip_data_ram[local_index(byte_addr)];
    } else {
      // global window layout: bank i at base i*IP_DATA_RAM_SIZE
      // (PS must set M_AXI base to start of DMEM region, e.g., CORE0_DMEM_BASE)
      u32 gword = byte_addr >> 2;
      return data_ram[gword];
    }
  };
  auto st_word = [&](u32 byte_addr, u32 wdata) {
#pragma HLS INLINE
    if (is_local_addr(byte_addr)) {
      ip_data_ram[local_index(byte_addr)] = wdata;
    } else {
      u32 gword = (byte_addr >> 2);
      data_ram[gword] = wdata;
    }
  };

  // ----------------- OPTIONAL counters -----------------
  u32 instret = 0;
  u32 cycles  = 0;

  // ----------------- RESET/INIT -----------------
  // If you want: clear some DMEM header (mailbox) here or let host do it.

  // ----------------- PIPELINE CORE EXECUTION LOOP -----------------
  // Hook your 4-stage rv32_pp core here.
  // Expected minimal contracts:
  //  - fetch from code_ram[pc>>2]
  //  - MEM stage uses ld_word()/st_word() above
  //  - PC starts at 0 (linker -Ttext=0x0)
  //  - on program termination: write minstret/mcycle to mailbox and return
  //
  // NOTE: If your existing rv32_pp core is already a function, call it and
  // give it lambdas or function pointers for memory access.

  // *** PSEUDO-SKELETON (replace with your implementation) ***
  u32 pc = 0;
  bool running = true;
  while (running) {
#pragma HLS PIPELINE II=1
    // FETCH
    u32 inst = code_ram[(pc >> 2)];
    // DECODE/EXEC/WRITEBACK ... (your existing rv32_pp stages & state)
    // Use ld_word()/st_word() in MEM stage.
    // Update 'pc' as per instruction semantics.
    // Update instret++ per retired instruction; cycles++ each loop if you want.
    // Set 'running=false' on your exit convention (e.g., write to mailbox and break)
    running = false; // placeholder to allow C-synth; remove when integrating core
  }

  if (nb_instruction) *nb_instruction = instret;
  if (nb_cycle)       *nb_cycle       = cycles;
}
} // extern "C"
