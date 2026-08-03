// tb_mc_hybrid.cpp
#include <cstdio>
#include "rv32i_pp_ip.h"

// HLS top under test (hybrid: m_axi_gmem for IMEM, BRAM for DMEM)
extern "C" void rv32i_pp_ip_mc_gmem(
  unsigned int  ip_num,
  unsigned int  start_pc,                 // WORD index
  volatile unsigned int *gmem,            // unified AXI space (IMEM lives here)
  unsigned int  code_base_words,          // WORD base of IMEM window
  int           ip_data_ram[DATA_RAM_SIZE], // native BRAM DMEM
  unsigned int *nb_instruction,
  unsigned int *nb_cycle);

// ----------------------------------------------------------------------
// Small helpers to build a minimal program (all encodings are RV32I)
static inline unsigned ENCOD_I(int imm12, int rs1, int funct3, int rd, unsigned opcode) {
  return ((imm12 & 0xFFF) << 20) | ((rs1 & 0x1F) << 15) | ((funct3 & 7) << 12)
         | ((rd & 0x1F) << 7) | (opcode & 0x7F);
}
static inline unsigned ENCOD_S(int imm12, int rs2, int rs1, int funct3, unsigned opcode) {
  unsigned imm_11_5 = (imm12 >> 5) & 0x7F;
  unsigned imm_4_0  = imm12 & 0x1F;
  return (imm_11_5 << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15)
         | ((funct3 & 7) << 12) | (imm_4_0 << 7) | (opcode & 0x7F);
}

// mnemonics we’ll use
static const unsigned OPC_OP_IMM = 0x13; // addi
static const unsigned OPC_LOAD   = 0x03; // lw
static const unsigned OPC_STORE  = 0x23; // sw
static const unsigned RET32      = 0x00008067u; // jalr x0,x1,0 (x1=RA=0 at reset)

// Program:
//  0: addi x5,x0,7
//  1: sw   x5,0(x0)         ; DMEM[0] = 7
//  2: lw   x6,0(x0)         ; x6 = DMEM[0]
//  3: ret                   ; stop (is_ret=1, ra/result=0)
static void build_program(unsigned *code) {
  code[0] = ENCOD_I(/*imm*/7, /*rs1*/0, /*funct3*/0, /*rd*/5, OPC_OP_IMM);
  code[1] = ENCOD_S(/*imm*/0, /*rs2*/5, /*rs1*/0, /*funct3*/2, OPC_STORE);
  code[2] = ENCOD_I(/*imm*/0, /*rs1*/0, /*funct3*/2, /*rd*/6, OPC_LOAD);
  code[3] = RET32;
}

int main() {
  // unified AXI space for IMEM (big enough for comfort)
  static unsigned int gmem[CODE_RAM_SIZE + DATA_RAM_SIZE + 64];
  // native BRAM DMEM
  static int          dmem[DATA_RAM_SIZE];

  // zero memories
  for (unsigned i=0; i<CODE_RAM_SIZE + DATA_RAM_SIZE + 64; ++i) gmem[i] = 0x00000013u; // NOPs
  for (unsigned i=0; i<DATA_RAM_SIZE; ++i) dmem[i] = 0;

  // lay out IMEM at the start of gmem (word addressing)
  const unsigned code_base_words = 0;
  unsigned *code = &gmem[code_base_words];

  build_program(code);

  // run
  unsigned nbi=0, nbc=0;
  unsigned start_pc = 0;          // first instruction index
  unsigned ip_num   = 0;          // a0/x10 will see this if you use it

  std::printf("[TB] start\n");
  rv32i_pp_ip_mc_gmem(ip_num, start_pc, gmem, code_base_words, dmem, &nbi, &nbc);
  std::printf("[TB] done: nbi=%u nbc=%u, DMEM[0]=%d\n", nbi, nbc, dmem[0]);

  // simple checks
  if (dmem[0] != 7) {
    std::printf("[TB] ERROR: DMEM[0] != 7 (got %d)\n", dmem[0]);
    return 1;
  }
  if (nbi == 0) {
    std::printf("[TB] ERROR: nbi==0\n");
    return 1;
  }
  return 0;
}
