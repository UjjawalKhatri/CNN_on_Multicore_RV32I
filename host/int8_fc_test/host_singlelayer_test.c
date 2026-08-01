/* host_quadcore_fc_int8.c — INT8 FC on 4-stage hybrid, 4 cores
 *
 * Firmware per core:
 *   fc_int8_core0.hex .. fc_int8_core3.hex
 *
 * Worker mailbox ABI:
 *   MB[0]  = done
 *   MB[1]  = instret low
 *   MB[2]  = instret high
 *   MB[3]  = cycle low
 *   MB[4]  = cycle high
 *   MB[8]  = input_dim
 *   MB[9]  = output_dim total
 *   MB[10] = out_start
 *   MB[11] = out_count
 *   MB[12] = requant_shift
 *   MB[13] = relu enable
 *
 * Data at ARR_OFS:
 *   [input vector]                    input_dim * int8
 *   [weight slice]                    (input_dim * out_count) * int8
 *   [bias slice]                      out_count * int32
 *   [output slice]                    out_count * int8
 */

#include "platform.h"
#include "xil_printf.h"
#include "xil_cache.h"
#include "xil_types.h"
#include "xparameters.h"
#include "sleep.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "xrv32i_pp_ip_mc_gmem.h"
#include "xrv32i_pp_ip_mc_gmem_hw.h"

typedef u32 word_type;

/* ===== design/map ===== */
#define NB_CORES 4

/* If your Address Editor uses 64KB windows per core: */
static const u32 IMEM_BASE[NB_CORES] = {
  0x40000000U, 0x40010000U, 0x40020000U, 0x40030000U
};
#define BRAM_SIZE (64U*1024U)

/* If instead you have 128KB windows, switch to:
static const u32 IMEM_BASE[NB_CORES] = {
  0x40000000U, 0x40020000U, 0x40040000U, 0x40060000U
};
#define BRAM_SIZE (128U*1024U)
*/

#define MB_OFS   0x2000U
#define ARR_OFS  0x2100U
#define MB_ADDR(c)   (IMEM_BASE[(c)] + MB_OFS)
#define ARR_ADDR(c)  (IMEM_BASE[(c)] + ARR_OFS)

static inline u32 align_up(u32 x, u32 a){
  u32 m = a - 1U;
  return (x + m) & ~m;
}

/* ===== firmware images ===== */
word_type code_ram_core0[] = {
  #include "fc_int8_core0.hex"
};
word_type code_ram_core1[] = {
  #include "fc_int8_core1.hex"
};
word_type code_ram_core2[] = {
  #include "fc_int8_core2.hex"
};
word_type code_ram_core3[] = {
  #include "fc_int8_core3.hex"
};

static word_type* IMG[NB_CORES] = {
  code_ram_core0, code_ram_core1, code_ram_core2, code_ram_core3
};

static int IMG_WORDS[NB_CORES] = {
  (int)(sizeof(code_ram_core0)/4),
  (int)(sizeof(code_ram_core1)/4),
  (int)(sizeof(code_ram_core2)/4),
  (int)(sizeof(code_ram_core3)/4)
};

/* ===== driver instances ===== */
static XRv32i_pp_ip_mc_gmem g[NB_CORES];

static void write_imem_words(int cid, const word_type* img, int words){
  u32 base = IMEM_BASE[cid];
  for(int i = 0; i < words; i++) Xil_Out32(base + (i << 2), img[i]);
}

static int init_core(int cid){
  u16 did = (cid==0)? XPAR_XRV32I_PP_IP_MC_GMEM_0_DEVICE_ID :
            (cid==1)? XPAR_XRV32I_PP_IP_MC_GMEM_1_DEVICE_ID :
            (cid==2)? XPAR_XRV32I_PP_IP_MC_GMEM_2_DEVICE_ID :
                      XPAR_XRV32I_PP_IP_MC_GMEM_3_DEVICE_ID;

  XRv32i_pp_ip_mc_gmem_Config* cfg = XRv32i_pp_ip_mc_gmem_LookupConfig(did);
  if(!cfg){
    xil_printf("LookupConfig fail %d\r\n", cid);
    return -1;
  }

  int st = XRv32i_pp_ip_mc_gmem_CfgInitialize(&g[cid], cfg);
  if(st != XST_SUCCESS){
    xil_printf("CfgInit fail %d (%d)\r\n", cid, st);
    return -1;
  }
  return 0;
}

static void start_core(int cid){
  xil_printf("Load Core%d code: %d words @0x%08lx\r\n",
             cid, IMG_WORDS[cid], (unsigned long)IMEM_BASE[cid]);

  write_imem_words(cid, IMG[cid], IMG_WORDS[cid]);
  Xil_DCacheFlushRange(IMEM_BASE[cid], BRAM_SIZE);

  XRv32i_pp_ip_mc_gmem_Set_code_base_words(&g[cid], (u32)(IMEM_BASE[cid] >> 2));
  XRv32i_pp_ip_mc_gmem_Set_ip_num(&g[cid], (u32)cid);
  XRv32i_pp_ip_mc_gmem_Set_start_pc(&g[cid], 0);
  XRv32i_pp_ip_mc_gmem_Set_gmem_offset(&g[cid], 0);
  XRv32i_pp_ip_mc_gmem_Start(&g[cid]);
}

/* ===== first safe test sizes ===== */
#define INPUT_DIM      8
#define OUTPUT_DIM     8
#define REQUANT_SHIFT  0
#define RELU_EN        0

/* ===== PS-side tensors ===== */
static int8_t  g_input[INPUT_DIM];
static int8_t  g_weights[OUTPUT_DIM][INPUT_DIM];
static int32_t g_bias[OUTPUT_DIM];
static int8_t  g_output_ref[OUTPUT_DIM];
static int8_t  g_output_hw[OUTPUT_DIM];

static inline int8_t sat_i8_host(int32_t x){
  if(x > 127) return 127;
  if(x < -128) return -128;
  return (int8_t)x;
}

static void build_test_vectors(void){
  int i, j;

  for(i = 0; i < INPUT_DIM; ++i){
    g_input[i] = (int8_t)(i - 3);
  }

  for(j = 0; j < OUTPUT_DIM; ++j){
    for(i = 0; i < INPUT_DIM; ++i){
      g_weights[j][i] = (int8_t)((j + 1) - i);
    }
  }

  for(j = 0; j < OUTPUT_DIM; ++j){
    g_bias[j] = (int32_t)(j * 3 - 2);
  }
}

static void compute_reference(void){
  int i, j;

  for(j = 0; j < OUTPUT_DIM; ++j){
    int32_t acc = g_bias[j];

    for(i = 0; i < INPUT_DIM; ++i){
      acc += (int32_t)g_input[i] * (int32_t)g_weights[j][i];
    }

    if(REQUANT_SHIFT){
      int32_t rnd = 1 << (REQUANT_SHIFT - 1);
      acc = (acc + rnd) >> REQUANT_SHIFT;
    }

    if(RELU_EN && acc < 0) acc = 0;
    g_output_ref[j] = sat_i8_host(acc);
  }
}

typedef struct {
  u32 off_in, off_w, off_b, off_out;
  u32 out_start, out_count;
} CoreLayout;

/* ===== IPC printers ===== */
static void print_ipc_scaled(const char* tag, unsigned inst, unsigned cyc){
  if(!cyc){
    xil_printf("%s: %u inst / %u cyc (IPC=NA)\r\n", tag, inst, cyc);
    return;
  }
  unsigned long long num = (unsigned long long)inst * 1000ull;
  unsigned ipc = (unsigned)((num + (cyc/2)) / cyc);
  xil_printf("%s: %u inst / %u cyc (IPC=%u.%03u)\r\n",
             tag, inst, cyc, ipc/1000, ipc%1000);
}

static void print_parallel_ipc4(u32 inst[NB_CORES], u32 cyc[NB_CORES]){
  u32 wall = 0;
  unsigned long long tot = 0;

  for(int c = 0; c < NB_CORES; c++){
    if(cyc[c] > wall) wall = cyc[c];
    tot += inst[c];
  }

  if(!wall){
    xil_printf("Parallel IPC: NA (wall=0)\r\n");
    return;
  }

  unsigned m = (unsigned)((tot * 1000ull + wall/2) / wall);
  xil_printf("Parallel IPC ~= %u.%03u  (tot_inst=%u, wall_cyc=%u)\r\n",
             m/1000, m%1000, (u32)tot, wall);
}

/* ===== pack one core ===== */
static void pack_core_fc(int cid, int out_start, int out_count, CoreLayout* L){
  volatile u8* base = (u8*)ARR_ADDR(cid);

  u32 off_in  = 0;
  u32 off_w   = align_up(off_in + INPUT_DIM, 64);
  u32 off_b   = align_up(off_w + (u32)INPUT_DIM * (u32)out_count, 64);
  u32 off_out = align_up(off_b + (u32)out_count * 4u, 64);
  u32 need    = off_out + (u32)out_count;

  if(ARR_OFS + need > BRAM_SIZE){
    xil_printf("ERROR: BRAM overflow on core %d (need %u B)\r\n", cid, (unsigned)need);
    while(1);
  }

  /* input vector */
  {
    volatile int8_t* in = (volatile int8_t*)(base + off_in);
    for(int i = 0; i < INPUT_DIM; ++i) in[i] = g_input[i];
  }

  /* weight slice: local-output-major layout */
  {
    volatile int8_t* w = (volatile int8_t*)(base + off_w);
    for(int j = 0; j < out_count; ++j){
      int gj = out_start + j;
      for(int i = 0; i < INPUT_DIM; ++i){
        w[j * INPUT_DIM + i] = g_weights[gj][i];
      }
    }
  }

  /* bias slice */
  {
    volatile int32_t* b = (volatile int32_t*)(base + off_b);
    for(int j = 0; j < out_count; ++j){
      b[j] = g_bias[out_start + j];
    }
  }

  /* clear output slice */
  {
    volatile int8_t* out = (volatile int8_t*)(base + off_out);
    for(int j = 0; j < out_count; ++j) out[j] = 0;
  }

  /* mailbox */
  {
    volatile u32* mb = (u32*)MB_ADDR(cid);
    for(int t = 0; t < 16; ++t) mb[t] = 0;

    mb[8]  = INPUT_DIM;
    mb[9]  = OUTPUT_DIM;
    mb[10] = (u32)out_start;
    mb[11] = (u32)out_count;
    mb[12] = REQUANT_SHIFT;
    mb[13] = RELU_EN;
  }

  Xil_DCacheFlushRange(IMEM_BASE[cid], BRAM_SIZE);

  L->off_in = off_in;
  L->off_w = off_w;
  L->off_b = off_b;
  L->off_out = off_out;
  L->out_start = (u32)out_start;
  L->out_count = (u32)out_count;

  xil_printf("Core%d pack: out_start=%d out_count=%d off_in=%lu off_w=%lu off_b=%lu off_out=%lu\r\n",
             cid, out_start, out_count,
             (unsigned long)off_in, (unsigned long)off_w,
             (unsigned long)off_b, (unsigned long)off_out);
}

/* ===== print/gather one core ===== */
static void dump_outputs(int cid, const CoreLayout* L){
  Xil_DCacheInvalidateRange(IMEM_BASE[cid], BRAM_SIZE);
  volatile u8* base = (u8*)ARR_ADDR(cid);
  const int8_t* out = (const int8_t*)(base + L->off_out);

  xil_printf("Core%d output slice: ", cid);
  for(u32 j = 0; j < L->out_count; ++j) xil_printf("%d ", (int)out[j]);
  xil_printf("\r\n");
}

static void gather_core_output(int cid, const CoreLayout* L){
  Xil_DCacheInvalidateRange(IMEM_BASE[cid], BRAM_SIZE);
  volatile u8* base = (u8*)ARR_ADDR(cid);
  const int8_t* out = (const int8_t*)(base + L->off_out);

  for(u32 j = 0; j < L->out_count; ++j){
    g_output_hw[L->out_start + j] = out[j];
  }
}

static int verify_outputs(void){
  int errors = 0;

  xil_printf("HW output : ");
  for(int j = 0; j < OUTPUT_DIM; ++j) xil_printf("%d ", (int)g_output_hw[j]);
  xil_printf("\r\n");

  xil_printf("REF output: ");
  for(int j = 0; j < OUTPUT_DIM; ++j) xil_printf("%d ", (int)g_output_ref[j]);
  xil_printf("\r\n");

  for(int j = 0; j < OUTPUT_DIM; ++j){
    if(g_output_hw[j] != g_output_ref[j]) errors++;
  }
  return errors;
}

/* ===== main ===== */
int main(void){
  init_platform();
  Xil_DCacheDisable();
  Xil_ICacheDisable();

  xil_printf("\r\n=== INT8 FC (quad-core, 4-stage hybrid) ===\r\n");

  build_test_vectors();
  compute_reference();
  memset(g_output_hw, 0, sizeof(g_output_hw));

  for(int c = 0; c < NB_CORES; c++){
    if(init_core(c) != 0){
      xil_printf("IP init fail c=%d\r\n", c);
      while(1);
    }
    volatile u32* mb = (u32*)MB_ADDR(c);
    for(int t = 0; t < 16; t++) mb[t] = 0;
    Xil_DCacheFlushRange(IMEM_BASE[c], BRAM_SIZE);
  }

  for(int c = 0; c < NB_CORES; c++) start_core(c);

  /* split output neurons across 4 cores */
  int Oc[NB_CORES];
  int baseO = OUTPUT_DIM / NB_CORES;
  int remO  = OUTPUT_DIM % NB_CORES;
  int prefix = 0;

  for(int c = 0; c < NB_CORES; c++){
    Oc[c] = baseO + (c < remO);
  }

  CoreLayout L[NB_CORES];
  for(int c = 0; c < NB_CORES; c++){
    pack_core_fc(c, prefix, Oc[c], &L[c]);
    prefix += Oc[c];
  }
  for(int c = 0; c < NB_CORES; c++){
    volatile u32* mb = (u32*)MB_ADDR(c);
    xil_printf("Waiting core%d...\r\n", c);

    while(mb[0] != 1U){
      Xil_DCacheInvalidateRange(IMEM_BASE[c], BRAM_SIZE);
      usleep(1000);
    }

    xil_printf("core%d done\r\n", c);
  }

  int errors = 0;
  for(int c = 0; c < NB_CORES; c++){
    dump_outputs(c, &L[c]);
    gather_core_output(c, &L[c]);
  }

  errors = verify_outputs();
  xil_printf(errors ? "VERIFY: FAIL (%d mismatches)\r\n"
                    : "VERIFY: PASS\r\n", errors);

  xil_printf("FC run complete.\r\n");

  while(1);
  return 0;
}
