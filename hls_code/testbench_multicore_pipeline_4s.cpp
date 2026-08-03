#include <cstdio>
#include <cstring>
#include "rv32i_pp_ip.h"

// avoid name clashes with macros from the header
#undef ADDI
#undef LW
#undef SW

static inline unsigned enc_ADDI(unsigned rd, unsigned rs1, int imm12) {
  unsigned uimm = (unsigned)(imm12 & 0xFFF);
  return (uimm << 20) | (rs1 << 15) | (0b000 << 12) | (rd << 7) | 0b0010011;
}
static inline unsigned enc_LW(unsigned rd, unsigned rs1, int imm12) {
  unsigned uimm = (unsigned)(imm12 & 0xFFF);
  return (uimm << 20) | (rs1 << 15) | (0b010 << 12) | (rd << 7) | 0b0000011;
}
static inline unsigned enc_SW(unsigned rs2, unsigned rs1, int imm12) {
  unsigned uimm = (unsigned)(imm12 & 0xFFF);
  unsigned imm11_5 = (uimm >> 5) & 0x7F;
  unsigned imm4_0  = uimm & 0x1F;
  return (imm11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (0b010 << 12)
       | (imm4_0 << 7) | 0b0100011;
}
static inline unsigned enc_RET() {
  unsigned imm12 = 0, rd = 0, rs1 = 1;
  return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (0b000 << 12) | (rd << 7) | 0b1100111;
}

int main() {
  static unsigned code_ram[CODE_RAM_SIZE];
  static int      ip_local[IP_DATA_RAM_SIZE];
  static int      data_ram[NB_IP * IP_DATA_RAM_SIZE];

  std::memset(code_ram, 0, sizeof(code_ram));
  std::memset(ip_local, 0, sizeof(ip_local));
  std::memset(data_ram, 0, sizeof(data_ram));

  // Simple program: x5 = 123; store to [0]; load to x6; ret
  code_ram[0] = enc_ADDI(1, 0, 0);    // nop-ish
  code_ram[1] = enc_ADDI(5, 0, 123);
  code_ram[2] = enc_SW  (5, 0, 0);
  code_ram[3] = enc_LW  (6, 0, 0);
  code_ram[4] = enc_RET();

  unsigned start_pc = 0;
  unsigned ip_num   = 0;
  unsigned nbi = 0, nbc = 0;

  rv32i_pp_ip(start_pc, ip_num, code_ram, data_ram, &nbi, &nbc, ip_local);

  std::printf("nb_instruction=%u nb_cycle=%u\n", nbi, nbc);
  std::printf("data_ram[0]=%d (expect 123)\n", data_ram[0]);
  return (data_ram[0] == 123) ? 0 : 1;
}
