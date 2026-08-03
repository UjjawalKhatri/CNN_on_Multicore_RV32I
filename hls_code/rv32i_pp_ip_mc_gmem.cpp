// rv32i_pp_ip_mc_gmem.cpp
#include "rv32i_pp_ip.h"         // brings CODE_RAM_SIZE / DATA_RAM_SIZE, types, structs
#include "fetch_decode.h"
#include "execute.h"
#include "mem.h"
#include "wb.h"
#include "rv32i_pp_ip_mc.h"
// Depth for AXI memory model (words). You can oversize it; safe choice:
enum { GMEM_DEPTH = (CODE_RAM_SIZE + DATA_RAM_SIZE) };

// ---------------------- TOP ----------------------
//  - Instruction fetch: via m_axi_gmem (IMEM sits behind AXI Interconnect -> AXI BRAM Ctrl -> BMG)
//  - Data memory: native BRAM port ip_data_ram (connect to BMG directly)
//  - Optional: PS can access DMEM by wiring the BMG second port to an AXI BRAM Ctrl
extern "C" void rv32i_pp_ip_mc_gmem(
  unsigned int  ip_num,                   // optional: we store it in x10/a0 (see init below)
  unsigned int  start_pc,                 // word index (like your current design)

  // AXI master used for IMEM (connect to interconnect)
  volatile unsigned int *gmem,
  unsigned int  code_base_words,          // IMEM base (WORDS) in the BRAM mapped behind the AXI BRAM Ctrl

  // Native BRAM DMEM (connect directly to Block Memory)
  int           ip_data_ram[DATA_RAM_SIZE],

  unsigned int *nb_instruction,
  unsigned int *nb_cycle)
{
  // ---------- Interfaces ----------
  #pragma HLS INTERFACE s_axilite port=ip_num
  #pragma HLS INTERFACE s_axilite port=start_pc

  #pragma HLS INTERFACE m_axi     port=gmem offset=slave bundle=gmem depth=GMEM_DEPTH
  #pragma HLS INTERFACE s_axilite port=code_base_words

  #pragma HLS INTERFACE bram      port=ip_data_ram

  #pragma HLS INTERFACE s_axilite port=nb_instruction
  #pragma HLS INTERFACE s_axilite port=nb_cycle
  #pragma HLS INTERFACE s_axilite port=return
  #pragma HLS INLINE recursive

  // ---------- Local state (same as your single-core top) ----------
  int            reg_file[NB_REGISTER];
  #pragma HLS ARRAY_PARTITION variable=reg_file dim=1 complete
  from_f_to_f_t  f_to_f, f_from_f;
  from_f_to_e_t  f_to_e, e_from_f;
  from_e_to_f_t  e_to_f, f_from_e;
  from_e_to_e_t  e_to_e, e_from_e;
  from_e_to_m_t  e_to_m, m_from_e;
  from_m_to_w_t  m_to_w, w_from_m;
  bit_t          is_running;
  unsigned int   nbi = 0, nbc = 0;

  // Init regs (x10/a0 = ip_num like in Ch.12)
  for (reg_num_p1_t i=0; i<NB_REGISTER; i++) reg_file[i] = 0;
  if (NB_REGISTER > 10) reg_file[10] = (int)ip_num;

  // Start PC
  e_to_f.target_pc = start_pc;
  e_to_f.set_pc    = 1;
  e_to_e.cancel    = 1;
  e_to_m.cancel    = 1;
  m_to_w.cancel    = 1;

  // View of IMEM inside the unified AXI space
  volatile unsigned int *ip_code_ram = gmem + code_base_words;

  do{
    #pragma HLS PIPELINE II=3

    // Latch forwarding (same as your rv32i_pp_ip.cpp)
    f_from_f = f_to_f; f_from_e = e_to_f; e_from_f = f_to_e;
    e_from_e = e_to_e; m_from_e = e_to_m; w_from_m = m_to_w;

    // IF+ID: fetch from AXI-mapped IMEM
    fetch_decode(f_from_f, f_from_e,
                 (unsigned int*)ip_code_ram,
                 &f_to_f, &f_to_e);

    // EX
    execute(f_to_e, e_from_f, e_from_e.cancel,
            m_from_e.cancel, m_from_e.d_i.has_no_dest,
            m_from_e.d_i.rd, m_from_e.result,
            w_from_m.cancel, w_from_m.has_no_dest,
            w_from_m.rd, w_from_m.result, reg_file,
            &e_to_f, &e_to_e, &e_to_m);

    // MEM: use native BRAM DMEM
    mem_access(m_from_e, ip_data_ram, &m_to_w);

    // WB
    wb(w_from_m, reg_file);

    // Stats / stop
    nbi += (unsigned)(!w_from_m.cancel);
    nbc += 1;
    is_running = (w_from_m.cancel || !w_from_m.is_ret || w_from_m.result != 0);
  } while (is_running);

  *nb_cycle       = nbc;
  *nb_instruction = nbi;
}
