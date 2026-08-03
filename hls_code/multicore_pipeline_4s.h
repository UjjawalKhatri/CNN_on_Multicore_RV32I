
#pragma once
#include "ap_int.h"

#ifndef LOG_NB_IP
#define LOG_NB_IP          1            // 2^LOG_NB_IP cores  (1 => 2 cores; 2 => 4 cores)
#endif

#ifndef LOG_DATA_RAM_SIZE
#define LOG_DATA_RAM_SIZE  17           // 2^17 = 128KB per core DMEM
#endif

#ifndef LOG_CODE_RAM_SIZE
#define LOG_CODE_RAM_SIZE  14           // 2^14 = 16KB per core IMEM
#endif

#define NB_IP               (1<<LOG_NB_IP)
#define IP_DATA_RAM_SIZE    (1u<<LOG_DATA_RAM_SIZE)     // bytes
#define IP_CODE_RAM_WORDS   (1u<<LOG_CODE_RAM_SIZE)     // 32b words

typedef ap_uint<LOG_NB_IP>    ip_num_t;
typedef ap_uint<LOG_NB_IP+1>  ip_num_p1_t;

// AXI beats are 32-bit words for RV32
typedef unsigned               u32;
