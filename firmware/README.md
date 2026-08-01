# Bare-Metal RISC-V Worker Firmware

This directory contains freestanding bare-metal C firmware for the RV32I soft cores, executable without libc or operating system overhead.

## Directory Organization

- `common/`: Shared startup assembly (`crt0_minimal.s`), linker script (`link.ld`), and format conversion script (`bin2carray.py`).
- `cnn_conv_core/`: Firmware servicing 2D Convolution layers with int8 quantized MAC arithmetic (`cnn_conv_core_service.c`).
- `cnn_fc_core/`: Firmware servicing Fully-Connected layers (`cnn_fc_core_service.c`).
- `fc_int8_core/`: Firmware for single-layer int8 FC kernel (`fc_int8_core.c`).
- `fc_mnist_core/`: Firmware for MNIST single-layer classification with 64-bit CSR cycle and instruction counters (`fc_mnist_core.c`).
- `fc_mlp_core/`: Firmware servicing 2-layer MLP inference with selectable int8 and int32 output modes (`fc_mlp_core_service.c`).

## Build Workflow

To compile any firmware service for the RV32I cores, use the standard RISC-V GNU Toolchain (`riscv64-unknown-elf-gcc` or `riscv32-unknown-elf-gcc`):

```bash
# Compile bare-metal C source with CRT0 startup and linker script
riscv64-unknown-elf-gcc -march=rv32i -mabi=ilp32 -O2 -nostdlib -nostartfiles \
  -T firmware/common/link.ld \
  firmware/common/crt0_minimal.s \
  firmware/cnn_conv_core/cnn_conv_core_service.c \
  -o cnn_conv_core.elf -lgcc

# Extract raw binary
riscv64-unknown-elf-objcopy -O binary cnn_conv_core.elf cnn_conv_core.bin

# Convert binary to C hex array for embedding into ARM host header files
python firmware/common/bin2carray.py cnn_conv_core.bin cnn_conv_core.hex
```
