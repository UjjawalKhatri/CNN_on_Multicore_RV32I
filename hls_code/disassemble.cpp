#ifndef __SYNTHESIS__
#include <stdio.h>

// Make sure instruction/macro defs (ECALL/EBREAK, opcodes, types) are visible first.
#include "rv32i_pp_ip.h"
#include "print.h"

// Fallbacks in case another TU compiled without rv32i_pp_ip.h first.
#ifndef ECALL
#define ECALL  0
#endif
#ifndef EBREAK
#define EBREAK 1
#endif

void disassemble(
  code_address_t        pc,
  instruction_t         instruction,
  decoded_instruction_t d_i)
{
  switch (d_i.type) {
    case R_TYPE: // OP
      print_op(d_i.func3, d_i.func7);
      printf(" ");
      print_reg_name(d_i.rd);
      printf(", ");
      print_reg_name(d_i.rs1);
      printf(", ");
      print_reg_name(d_i.rs2);
      break;

    case I_TYPE: // JALR || OP_IMM || LOAD || SYSTEM
      if (d_i.opcode == JALR) {
        if (d_i.rd == 0 && d_i.rs1 == RA) {
          printf("ret");
        } else {
          if (d_i.rd == 0) {
            printf("jr ");
          } else {
            printf("jalr ");
            if (d_i.rd != RA) {
              print_reg_name(d_i.rd);
              printf(", ");
            }
          }
          if (d_i.imm == 0) {
            print_reg_name(d_i.rs1);
          } else {
            printf("%d(", (int)d_i.imm);
            print_reg_name(d_i.rs1);
            printf(")");
          }
        }
      } else if (d_i.is_op_imm) {
        if (instruction == NOP) {
          printf("nop");
        } else {
          if (d_i.func3 == ADDI && d_i.rs1 == 0)
            printf("li");
          else
            print_op_imm(d_i.func3, d_i.func7);
          printf(" ");
          print_reg_name(d_i.rd);
          printf(", ");
          if (d_i.func3 != ADDI || d_i.rs1 != 0) {
            print_reg_name(d_i.rs1);
            printf(", ");
          }
          if (d_i.func3 != SLLI && d_i.func3 != SRLI)
            printf("%d", (int)d_i.imm);
          else
            printf("%u", (unsigned)d_i.rs2);
        }
      } else if (d_i.is_load) {
        printf("l");
        print_msize(d_i.func3);
        printf(" ");
        print_reg_name(d_i.rd);
        printf(", ");
        printf("%d(", (int)d_i.imm);
        print_reg_name(d_i.rs1);
        printf(")");
      } else if (d_i.opcode == SYSTEM) {
        // ECALL/EBREAK encode in imm[0]
        if ((d_i.imm & 1) == ECALL) printf("ecall");
        else                         printf("ebreak");
        printf(" %d", d_i.func3);
      }
      break;

    case S_TYPE: // STORE
      printf("s");
      print_msize(d_i.func3);
      printf(" ");
      print_reg_name(d_i.rs2);
      printf(", ");
      printf("%d(", (int)d_i.imm);
      print_reg_name(d_i.rs1);
      printf(")");
      break;

    case B_TYPE: { // BRANCH
      print_branch(d_i.func3);
      printf(" ");
      print_reg_name(d_i.rs1);
      printf(", ");
      print_reg_name(d_i.rs2);
      printf(", ");
      // Target = (pc + imm) in bytes; your pc is word-indexed, imm is in halfwords
      int target_bytes = ((int)pc + (int)(d_i.imm >> 1)) * (int)sizeof(instruction_t);
      printf("%d", target_bytes);
      break;
    }

    case U_TYPE: // LUI || AUIPC
      printf(d_i.is_lui ? "lui " : "auipc ");
      print_reg_name(d_i.rd);
      printf(", %d", (int)(d_i.imm << 12));
      break;

    case J_TYPE: { // JAL
      if (d_i.rd == 0) printf("j ");
      else {
        printf("jal ");
        print_reg_name(d_i.rd);
        printf(", ");
      }
      int target_bytes = ((int)pc + (int)(d_i.imm >> 1)) * (int)sizeof(instruction_t);
      printf("%d", target_bytes);
      break;
    }

    default: // UNDEFINED_TYPE, OTHER_TYPE
      break;
  }
  printf("\n");
}
#endif
