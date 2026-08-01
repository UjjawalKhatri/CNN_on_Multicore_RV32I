    .section .text
    .globl _start
_start:
    /* Set stack top — pick one that matches your per-core BRAM */
    /* 0x20000 = 128 KB, 0x10000 = 64 KB, 0x8000 = 32 KB */
    li   sp, 0x10000
    addi sp, sp, -16

    /* Jump to main() */
    li   a0, 0
    li   a1, 0
    jal  ra, main

1:  j    1b    /* if main returns, spin */
