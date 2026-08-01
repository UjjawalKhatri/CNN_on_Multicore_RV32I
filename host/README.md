# ARM Processing System Host Applications

This directory contains standalone C applications executed on the Zynq PS ARM Cortex-A9 host controller using Xilinx Standalone BSP functions.

## Applications Overview

1. `fruit_cnn_inference/`: Complete end-to-end Fruit CNN inference controller (`main.c`). Orchestrates Conv1 on RV32I workers, MaxPool1 on ARM host, Conv2 on RV32I workers, MaxPool2 on ARM host, FC1 on ARM, FC2 on ARM, and outputs predicted fruit classes via UART.
2. `mnist_single_layer/`: Host driver for 1-image (`host_1_image.c`) and 10-image batch (`host_10_images.c`) MNIST CNN inference.
3. `mnist_fc_single_layer/`: Continuous service-mode host driver (`host_10_images_single_layer.c`) for single-layer int8 FC classification.
4. `mnist_mlp_two_layer/`: Host controller (`host_1image_mlp.c`) orchestrating 2-layer MLP (784 -> 64 -> 10).
5. `int8_fc_test/`: Verification test bench (`host_singlelayer_test.c`) comparing hardware results against golden PS reference calculation.

## Host Pipeline Operation

1. Initialises AXI memory controllers (`XRv32i_pp_ip_mc_gmem`).
2. Flushes instruction memory and copies firmware hex images to worker BRAMs.
3. Packs input data, weights, bias parameters, and calibration scale factors into designated BRAM mailbox windows.
4. Asserts START flag (`MB[5] = 1`).
5. Polls for worker completion (`MB[0] == 1`).
6. Gathers worker outputs, performs pooling/soft-float dequantization, and prints predicted labels over UART interface at 115200 baud.
