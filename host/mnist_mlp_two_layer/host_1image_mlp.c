/* host_quadcore_mnist_mlp1.c
 *
 * One-image 2-layer MLP on 4-stage hybrid, 4 cores
 *
 * Pass 1: 784 -> 64, ReLU + requant -> int8 hidden
 * Pass 2: 64 -> 10, raw int32 logits
 *
 * Worker mailbox ABI:
 *   MB[0]  = done
 *   MB[5]  = job/start flag
 *   MB[6]  = output_mode   (0=int32 out, 1=int8 out)
 *   MB[8]  = input_dim
 *   MB[9]  = output_dim total
 *   MB[10] = out_start
 *   MB[11] = out_count
 *   MB[12] = requant_shift
 *   MB[13] = relu enable
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

#include "mnist_mlp1_data.h"

typedef u32 word_type;

/* ===== design/map ===== */
#define NB_CORES 4

static const u32 IMEM_BASE[NB_CORES] = {
  0x40000000U, 0x40010000U, 0x40020000U, 0x40030000U
};
#define BRAM_SIZE (64U*1024U)

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
  #include "fc_mnist_core0.hex"
};
word_type code_ram_core1[] = {
  #include "fc_mnist_core1.hex"
};
word_type code_ram_core2[] = {
  #include "fc_mnist_core2.hex"
};
word_type code_ram_core3[] = {
  #include "fc_mnist_core3.hex"
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

/* ===== dimensions ===== */
#define INPUT_DIM   MLP1_INPUT_DIM
#define HIDDEN_DIM  MLP1_HIDDEN_DIM
#define OUTPUT_DIM  MLP1_OUTPUT_DIM

static int8_t  g_hidden_hw[HIDDEN_DIM];
static int32_t g_logits_hw[OUTPUT_DIM];

typedef struct {
  u32 off_in, off_w, off_b, off_out;
  u32 out_start, out_count;
  u32 output_mode; /* 0=int32, 1=int8 */
} CoreLayout;

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

  {
    int st = XRv32i_pp_ip_mc_gmem_CfgInitialize(&g[cid], cfg);
    if(st != XST_SUCCESS){
      xil_printf("CfgInit fail %d (%d)\r\n", cid, st);
      return -1;
    }
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

static int argmax10(const int32_t *a){
  int idx = 0;
  for(int i = 1; i < OUTPUT_DIM; ++i){
    if(a[i] > a[idx]) idx = i;
  }
  return idx;
}

static void build_core_layout(int input_dim, int out_start, int out_count, int output_mode, CoreLayout* L){
  u32 off_in  = 0;
  u32 off_w   = align_up(off_in + (u32)input_dim, 64);
  u32 off_b   = align_up(off_w + (u32)input_dim * (u32)out_count, 64);
  u32 off_out = align_up(off_b + (u32)out_count * 4u, 64);
  u32 out_bytes = (output_mode == 0) ? ((u32)out_count * 4u) : (u32)out_count;
  u32 need    = off_out + out_bytes;

  if(ARR_OFS + need > BRAM_SIZE){
    xil_printf("ERROR: BRAM overflow (need %lu B)\r\n", (unsigned long)need);
    while(1);
  }

  L->off_in = off_in;
  L->off_w = off_w;
  L->off_b = off_b;
  L->off_out = off_out;
  L->out_start = (u32)out_start;
  L->out_count = (u32)out_count;
  L->output_mode = (u32)output_mode;
}

static void pack_static_pass1(int cid, const CoreLayout* L){
  volatile u8* base = (u8*)ARR_ADDR(cid);

  /* W1 slice: [local_out][784] */
  {
    volatile int8_t* w = (volatile int8_t*)(base + L->off_w);
    for(u32 j = 0; j < L->out_count; ++j){
      u32 gj = L->out_start + j;
      for(int i = 0; i < INPUT_DIM; ++i){
        w[j * INPUT_DIM + i] = mlp1_w1[gj][i];
      }
    }
  }

  /* b1 slice */
  {
    volatile int32_t* b = (volatile int32_t*)(base + L->off_b);
    for(u32 j = 0; j < L->out_count; ++j){
      b[j] = mlp1_b1[L->out_start + j];
    }
  }

  /* mailbox config for pass1 */
  {
    volatile u32* mb = (u32*)MB_ADDR(cid);
    mb[6]  = 1;                 /* int8 output */
    mb[8]  = INPUT_DIM;
    mb[9]  = HIDDEN_DIM;
    mb[10] = L->out_start;
    mb[11] = L->out_count;
    mb[12] = MLP1_SHIFT1;
    mb[13] = 1;                 /* ReLU on */
  }

  Xil_DCacheFlushRange(IMEM_BASE[cid], BRAM_SIZE);
}

static void pack_dynamic_pass1_input(int cid, const CoreLayout* L){
  volatile u8* base = (u8*)ARR_ADDR(cid);

  /* input image */
  {
    volatile int8_t* in = (volatile int8_t*)(base + L->off_in);
    for(int i = 0; i < INPUT_DIM; ++i){
      in[i] = mlp1_input[i];
    }
  }

  /* clear int8 output */
  {
    volatile int8_t* out = (volatile int8_t*)(base + L->off_out);
    for(u32 j = 0; j < L->out_count; ++j){
      out[j] = 0;
    }
  }

  Xil_DCacheFlushRange(IMEM_BASE[cid], BRAM_SIZE);
}

static void gather_pass1_hidden(int cid, const CoreLayout* L){
  Xil_DCacheInvalidateRange(IMEM_BASE[cid], BRAM_SIZE);
  {
    volatile u8* base = (u8*)ARR_ADDR(cid);
    const int8_t* out = (const int8_t*)(base + L->off_out);
    for(u32 j = 0; j < L->out_count; ++j){
      g_hidden_hw[L->out_start + j] = out[j];
    }
  }
}

static void pack_static_pass2(int cid, const CoreLayout* L){
  volatile u8* base = (u8*)ARR_ADDR(cid);

  /* W2 slice: [local_out][64] */
  {
    volatile int8_t* w = (volatile int8_t*)(base + L->off_w);
    for(u32 j = 0; j < L->out_count; ++j){
      u32 gj = L->out_start + j;
      for(int i = 0; i < HIDDEN_DIM; ++i){
        w[j * HIDDEN_DIM + i] = mlp1_w2[gj][i];
      }
    }
  }

  /* b2 slice */
  {
    volatile int32_t* b = (volatile int32_t*)(base + L->off_b);
    for(u32 j = 0; j < L->out_count; ++j){
      b[j] = mlp1_b2[L->out_start + j];
    }
  }

  /* mailbox config for pass2 */
  {
    volatile u32* mb = (u32*)MB_ADDR(cid);
    mb[6]  = 0;              /* int32 output */
    mb[8]  = HIDDEN_DIM;
    mb[9]  = OUTPUT_DIM;
    mb[10] = L->out_start;
    mb[11] = L->out_count;
    mb[12] = 0;
    mb[13] = 0;
  }

  Xil_DCacheFlushRange(IMEM_BASE[cid], BRAM_SIZE);
}

static void pack_dynamic_pass2_input(int cid, const CoreLayout* L){
  volatile u8* base = (u8*)ARR_ADDR(cid);

  /* input hidden[64] */
  {
    volatile int8_t* in = (volatile int8_t*)(base + L->off_in);
    for(int i = 0; i < HIDDEN_DIM; ++i){
      in[i] = g_hidden_hw[i];
    }
  }

  /* clear int32 output */
  {
    volatile int32_t* out = (volatile int32_t*)(base + L->off_out);
    for(u32 j = 0; j < L->out_count; ++j){
      out[j] = 0;
    }
  }

  Xil_DCacheFlushRange(IMEM_BASE[cid], BRAM_SIZE);
}

static void gather_pass2_logits(int cid, const CoreLayout* L){
  Xil_DCacheInvalidateRange(IMEM_BASE[cid], BRAM_SIZE);
  {
    volatile u8* base = (u8*)ARR_ADDR(cid);
    const int32_t* out = (const int32_t*)(base + L->off_out);
    for(u32 j = 0; j < L->out_count; ++j){
      g_logits_hw[L->out_start + j] = out[j];
    }
  }
}

static void run_job_all_cores(void){
  /* start job */
  for(int c = 0; c < NB_CORES; ++c){
    volatile u32* mb = (u32*)MB_ADDR(c);
    mb[0] = 0;
    mb[5] = 0;
    Xil_DCacheFlushRange(IMEM_BASE[c], BRAM_SIZE);
  }

  for(int c = 0; c < NB_CORES; ++c){
    volatile u32* mb = (u32*)MB_ADDR(c);
    mb[5] = 1;
    Xil_DCacheFlushRange(IMEM_BASE[c], BRAM_SIZE);
  }

  /* wait all done */
  for(int c = 0; c < NB_CORES; ++c){
    volatile u32* mb = (u32*)MB_ADDR(c);
    while(mb[0] != 1U){
      Xil_DCacheInvalidateRange(IMEM_BASE[c], BRAM_SIZE);
      usleep(1000);
    }
  }

  /* acknowledge */
  for(int c = 0; c < NB_CORES; ++c){
    volatile u32* mb = (u32*)MB_ADDR(c);
    mb[5] = 0;
    Xil_DCacheFlushRange(IMEM_BASE[c], BRAM_SIZE);
  }
}

int main(void){
  CoreLayout L1[NB_CORES];
  CoreLayout L2[NB_CORES];
  int prefix;

  init_platform();
  Xil_DCacheDisable();
  Xil_ICacheDisable();

  xil_printf("\r\n=== MNIST 2-layer MLP, one image ===\r\n");
  xil_printf("Test index : %d\r\n", MLP1_TEST_INDEX);
  xil_printf("True label : %d\r\n", MLP1_TRUE_LABEL);
  xil_printf("Ref pred   : %d\r\n", MLP1_REF_PRED);
  xil_printf("SHIFT1     : %d\r\n", MLP1_SHIFT1);

  memset(g_hidden_hw, 0, sizeof(g_hidden_hw));
  memset(g_logits_hw, 0, sizeof(g_logits_hw));

  /* init + start cores once */
  for(int c = 0; c < NB_CORES; ++c){
    if(init_core(c) != 0){
      xil_printf("IP init fail c=%d\r\n", c);
      while(1);
    }
    {
      volatile u32* mb = (u32*)MB_ADDR(c);
      for(int t = 0; t < 16; ++t) mb[t] = 0;
    }
  }

  for(int c = 0; c < NB_CORES; ++c){
    start_core(c);
  }

  xil_printf("All cores started once.\r\n");

  /* ---------- PASS 1: 784 -> 64 ---------- */
  prefix = 0;
  for(int c = 0; c < NB_CORES; ++c){
    build_core_layout(INPUT_DIM, prefix, 16, 1, &L1[c]); /* 16 each */
    prefix += 16;
  }

  for(int c = 0; c < NB_CORES; ++c){
    pack_static_pass1(c, &L1[c]);
    pack_dynamic_pass1_input(c, &L1[c]);
  }

  run_job_all_cores();

  for(int c = 0; c < NB_CORES; ++c){
    gather_pass1_hidden(c, &L1[c]);
  }

  xil_printf("Hidden first 16: ");
  for(int i = 0; i < 16; ++i){
    xil_printf("%d ", (int)g_hidden_hw[i]);
  }
  xil_printf("\r\n");

  /* ---------- PASS 2: 64 -> 10 ---------- */
  {
    int Oc[NB_CORES];
    int baseO = OUTPUT_DIM / NB_CORES;
    int remO  = OUTPUT_DIM % NB_CORES;

    prefix = 0;
    for(int c = 0; c < NB_CORES; ++c){
      Oc[c] = baseO + (c < remO);   /* 3,3,2,2 */
    }

    for(int c = 0; c < NB_CORES; ++c){
      build_core_layout(HIDDEN_DIM, prefix, Oc[c], 0, &L2[c]);
      prefix += Oc[c];
    }

    for(int c = 0; c < NB_CORES; ++c){
      pack_static_pass2(c, &L2[c]);
      pack_dynamic_pass2_input(c, &L2[c]);
    }

    run_job_all_cores();

    for(int c = 0; c < NB_CORES; ++c){
      gather_pass2_logits(c, &L2[c]);
    }
  }

  /* ---------- verify ---------- */
  {
    int hw_pred = argmax10(g_logits_hw);
    int hidden_ok = 1;
    int logits_ok = 1;

    for(int i = 0; i < HIDDEN_DIM; ++i){
      if(g_hidden_hw[i] != mlp1_hidden_ref[i]){
        hidden_ok = 0;
        break;
      }
    }

    for(int j = 0; j < OUTPUT_DIM; ++j){
      if(g_logits_hw[j] != mlp1_logits_ref[j]){
        logits_ok = 0;
        break;
      }
    }

    xil_printf("Hidden match : %s\r\n", hidden_ok ? "OK" : "BAD");

    xil_printf("HW logits  : ");
    for(int j = 0; j < OUTPUT_DIM; ++j) xil_printf("%ld ", (long)g_logits_hw[j]);
    xil_printf("\r\n");

    xil_printf("REF logits : ");
    for(int j = 0; j < OUTPUT_DIM; ++j) xil_printf("%ld ", (long)mlp1_logits_ref[j]);
    xil_printf("\r\n");

    xil_printf("HW pred    : %d\r\n", hw_pred);
    xil_printf("Logits match: %s\r\n", logits_ok ? "OK" : "BAD");
  }

  while(1);
  return 0;
}
