# Hardware Architecture and FPGA Platform

This directory contains hardware platform configuration files and architectural documentation for the RV32I multicore system deployed on the PYNQ-Z2 (Zynq-7020) FPGA board.

## Subdirectories and Components

- `platform/platform.tcl`: TCL automation script to recreate the Vitis hardware platform (`4core_4stage_pipe`) from the Vivado exported hardware `.xsa` file.
- `rtl/`: (Place for Vivado SystemVerilog/Verilog core files)
- `hls/`: (Place for Vitis HLS cores and IP block synthesis definitions)

## System Architecture

The overall hardware organization comprises:
1. **Host Processing System (PS)**: ARM Cortex-A9 managing workload distribution, peripheral communication, UART console, and memory initialization.
2. **AXI Interconnect**: AXI4/AXI4-Lite interconnect bridging the Zynq PS and worker cores.
3. **RV32I Worker Cores**: 4-stage in-order pipelined RV32I soft-processors (scaled across 2, 4, and 8 core variants) and comparative 6-stage multicycle cores.
4. **Per-Core Private Memory Windows**: Equal BRAM partitioning:
   - 2-Core Configuration: 128 KiB BRAM per core
   - 4-Core Configuration: 64 KiB BRAM per core
   - 8-Core Configuration: 32 KiB BRAM per core

## Memory Address Mapping

Base Address: `CORE_BASE(i) = 0x40000000 + i * WINDOW_SIZE`

In-Window Offset Layout:
- `0x0000 - 0x1FFF`: Private Core Stack Space
- `0x2000 - 0x20FF`: Mailbox Control & Synchronization Registers (MB[0]: Done flag, MB[5]: Job flag, MB[1..4]: CSR Cycle/Instret Counters)
- `0x2100+`: Job descriptors, input activation buffers, weight matrices, and output arrays.
