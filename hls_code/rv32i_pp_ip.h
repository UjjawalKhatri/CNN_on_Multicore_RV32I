#ifndef __RV32I_PP_IP
#define __RV32I_PP_IP

#include "ap_int.h"
#include "debug_rv32i_pp_ip.h"
#include <stdint.h>

/* ====== sizes ====== */
#define LOG_CODE_RAM_SIZE 15
#define CODE_RAM_SIZE     (1u << LOG_CODE_RAM_SIZE)   // 64K instr words
#define LOG_DATA_RAM_SIZE 15
#define DATA_RAM_SIZE     (1u << LOG_DATA_RAM_SIZE)   // 64K data words
#define LOG_REG_FILE_SIZE 5
#define NB_REGISTER       (1u << LOG_REG_FILE_SIZE)

/* ====== ISA fields / opcodes ====== */
typedef unsigned int               instruction_t;
typedef ap_uint<LOG_CODE_RAM_SIZE> code_address_t;  // word index
typedef ap_uint<LOG_DATA_RAM_SIZE> w_data_address_t;
typedef ap_uint<LOG_DATA_RAM_SIZE+1> h_data_address_t;
typedef ap_uint<LOG_DATA_RAM_SIZE+2> b_data_address_t;

typedef ap_uint<3>  type_t;
typedef ap_int<20>  immediate_t;
typedef ap_int<12>  i_immediate_t;
typedef ap_int<12>  s_immediate_t;
typedef ap_int<12>  b_immediate_t;
typedef ap_int<20>  u_immediate_t;
typedef ap_int<20>  j_immediate_t;
typedef ap_uint<5>  opcode_t;       // instr[6:2]
typedef ap_uint<3>  func3_t;
typedef ap_uint<7>  func7_t;
typedef ap_uint<1>  bit_t;
typedef ap_uint<LOG_REG_FILE_SIZE+1> reg_num_p1_t;
typedef ap_uint<LOG_REG_FILE_SIZE>   reg_num_t;

#define UNDEFINED_TYPE 0
#define R_TYPE 1
#define I_TYPE 2
#define S_TYPE 3
#define B_TYPE 4
#define U_TYPE 5
#define J_TYPE 6
#define OTHER_TYPE 7

/* opcodes (instr[6:2]) */
#define LOAD     0b00000
#define OP_IMM   0b00100
#define AUIPC    0b00101
#define STORE    0b01000
#define OP       0b01100
#define LUI      0b01101
#define BRANCH   0b11000
#define JALR     0b11001
#define JAL      0b11011
#define SYSTEM   0b11100

/* branch funct3 */
#define BEQ  0
#define BNE  1
#define BLT  4
#define BGE  5
#define BLTU 6
#define BGEU 7

/* ALU funct3 */
#define ADD  0
#define SLL  1
#define SLT  2
#define SLTU 3
#define XOR  4
#define SRL  5  // SRA when func7[5]=1
#define OR   6
#define AND  7

// OP-IMM funct3 aliases (needed by print_op_imm in print.cpp)
#define ADDI  0
#define SLLI  1
#define SLTI  2
#define SLTIU 3
#define XORI  4
#define SRLI  5
#define ORI   6
#define ANDI  7

/* load/store funct3 */
#define LB  0
#define LH  1
#define LW  2
#define LBU 4
#define LHU 5

#define SB 0
#define SH 1
#define SW 2

/* registers */
#define RA 1  // x1 (return addr)

/* “RET” sentinel (ECALL via jalr x0,x1,0) used by the book design */
#define RET 0x00008067u
#define NOP 0x00000013u

/* ====== decoded instruction ====== */
typedef struct {
  opcode_t    opcode;
  reg_num_t   rd;
  func3_t     func3;
  reg_num_t   rs1;
  reg_num_t   rs2;
  func7_t     func7;
  type_t      type;
  immediate_t imm;
  bit_t       is_load;
  bit_t       is_store;
  bit_t       is_branch;
  bit_t       is_jalr;
  bit_t       is_jal;
  bit_t       is_ret;
  bit_t       is_lui;
  bit_t       is_op_imm;
  bit_t       has_no_dest;
  bit_t       is_r_type;
} decoded_instruction_t;

typedef struct {
  bit_t      inst_31;
  ap_uint<6> inst_30_25;
  ap_uint<4> inst_24_21;
  bit_t      inst_20;
  ap_uint<8> inst_19_12;
  ap_uint<4> inst_11_8;
  bit_t      inst_7;
} decoded_immediate_t;

/* ====== pipeline regs ====== */
typedef struct { code_address_t next_pc; } from_f_to_f_t;

typedef struct {
  code_address_t        pc;
  decoded_instruction_t d_i;
#ifndef __SYNTHESIS__
#ifdef DEBUG_DISASSEMBLE
  instruction_t         instruction;
#endif
#endif
} from_f_to_e_t;

typedef struct {
  code_address_t target_pc;
  bit_t          set_pc;
} from_e_to_f_t;

typedef struct { bit_t cancel; } from_e_to_e_t;

typedef struct {
  bit_t                 cancel;
  int                   result;
  int                   rv2;
  decoded_instruction_t d_i;
#ifndef __SYNTHESIS__
#ifdef DEBUG_DISASSEMBLE
  code_address_t        pc;
  instruction_t         instruction;
#endif
#ifdef DEBUG_EMULATE
  code_address_t        next_pc;
#endif
#endif
} from_e_to_m_t;

typedef struct {
  bit_t     cancel;
  int       result;
  reg_num_t rd;
  bit_t     is_ret;
  bit_t     has_no_dest;
#ifndef __SYNTHESIS__
#ifdef DEBUG_DISASSEMBLE
  instruction_t         instruction;
  decoded_instruction_t d_i;
  code_address_t        pc;
#endif
#ifdef DEBUG_EMULATE
#ifndef DEBUG_DISASSEMBLE
  decoded_instruction_t d_i;
#endif
  code_address_t        next_pc;
#endif
#endif
} from_m_to_w_t;

/* ====== stage prototypes ====== */
void fetch(code_address_t pc, unsigned int *code_ram,
           code_address_t *next_pc, instruction_t *instruction);

void decode(instruction_t instruction, decoded_instruction_t *d_i);

void fetch_decode(from_f_to_f_t f_from_f, from_e_to_f_t f_from_e,
                  unsigned int *code_ram, from_f_to_f_t *f_to_f, from_f_to_e_t *f_to_e);

int   read_reg(int *reg_file, reg_num_t rs);
bit_t compute_branch_result(int rv1, int rv2, func3_t func3);
int   compute_op_result(decoded_instruction_t d_i, int rv1, int rv2);
int   compute_result(code_address_t pc, decoded_instruction_t d_i, int rv1);
code_address_t compute_next_pc(code_address_t pc, decoded_instruction_t d_i, int rv1, bit_t cond);

void execute(from_f_to_e_t  f_to_e, from_f_to_e_t  e_from_f, bit_t e_cancel,
             bit_t m_cancel, bit_t m_has_no_dest, reg_num_t m_rd, int m_result,
             bit_t w_cancel, bit_t w_has_no_dest, reg_num_t w_rd, int w_result,
             int *reg_file,
             from_e_to_f_t *e_to_f, from_e_to_e_t *e_to_e, from_e_to_m_t *e_to_m);

void mem_access(from_e_to_m_t m_from_e, int *data_ram, from_m_to_w_t *m_to_w);

void wb(from_m_to_w_t w_from_m, int *reg_file);

/* ====== top ====== */
void rv32i_pp_ip(
  unsigned int  start_pc,
  unsigned int  code_ram[CODE_RAM_SIZE],
  int           data_ram[DATA_RAM_SIZE],
  unsigned int *nb_instruction,
  unsigned int *nb_cycle,
  unsigned int  max_cycles  // watchdog; 0 = disabled
);

#endif
