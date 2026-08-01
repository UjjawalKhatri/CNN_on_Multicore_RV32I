/* host_quadcore_mnist_fc_int8_10_service.c
 *
 * MNIST single-layer FC on 4-stage hybrid, 4 cores
 * Continuous service mode: cores started once, many images processed
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

#include "mnist_fc_data_10.h"

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

#define INPUT_DIM      MNIST_INPUT_DIM
#define OUTPUT_DIM     MNIST_OUTPUT_DIM
#define REQUANT_SHIFT  0
#define RELU_EN        0

static int32_t g_output_hw[OUTPUT_DIM];

typedef struct {
  u32 off_in, off_w, off_b, off_out;
  u32 out_start, out_count;
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

static void build_core_layout(int out_start, int out_count, CoreLayout* L){
  u32 off_in  = 0;
  u32 off_w   = align_up(off_in + INPUT_DIM, 64);
  u32 off_b   = align_up(off_w + (u32)INPUT_DIM * (u32)out_count, 64);
  u32 off_out = align_up(off_b + (u32)out_count * 4u, 64);
  u32 need    = off_out + (u32)out_count * 4u;

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
}

static void pack_static_core_fc(int cid, const CoreLayout* L){
  volatile u8* base = (u8*)ARR_ADDR(cid);

  /* weight slice */
  {
    volatile int8_t* w = (volatile int8_t*)(base + L->off_w);
    for(u32 j = 0; j < L->out_count; ++j){
      u32 gj = L->out_start + j;
      for(int i = 0; i < INPUT_DIM; ++i){
        w[j * INPUT_DIM + i] = mnist_weights[gj][i];
      }
    }
  }

  /* bias slice */
  {
    volatile int32_t* b = (volatile int32_t*)(base + L->off_b);
    for(u32 j = 0; j < L->out_count; ++j){
      b[j] = mnist_bias[L->out_start + j];
    }
  }

  /* mailbox config */
  {
    volatile u32* mb = (u32*)MB_ADDR(cid);

    mb[8]  = INPUT_DIM;
    mb[9]  = OUTPUT_DIM;
    mb[10] = L->out_start;
    mb[11] = L->out_count;
    mb[12] = REQUANT_SHIFT;
    mb[13] = RELU_EN;
  }

  Xil_DCacheFlushRange(IMEM_BASE[cid], BRAM_SIZE);
}

static void pack_dynamic_input_output(int cid, int img_idx, const CoreLayout* L){
  volatile u8* base = (u8*)ARR_ADDR(cid);

  /* input vector */
  {
    volatile int8_t* in = (volatile int8_t*)(base + L->off_in);
    for(int i = 0; i < INPUT_DIM; ++i){
      in[i] = mnist_inputs[img_idx][i];
    }
  }

  /* clear output slice */
  {
    volatile int32_t* out = (volatile int32_t*)(base + L->off_out);
    for(u32 j = 0; j < L->out_count; ++j){
      out[j] = 0;
    }
  }

  Xil_DCacheFlushRange(IMEM_BASE[cid], BRAM_SIZE);
}

static void gather_core_output(int cid, const CoreLayout* L){
  Xil_DCacheInvalidateRange(IMEM_BASE[cid], BRAM_SIZE);
  {
    volatile u8* base = (u8*)ARR_ADDR(cid);
    const int32_t* out = (const int32_t*)(base + L->off_out);

    for(u32 j = 0; j < L->out_count; ++j){
      g_output_hw[L->out_start + j] = out[j];
    }
  }
}

int main(void){
  int total_ok = 0;
  int Oc[NB_CORES];
  int baseO = OUTPUT_DIM / NB_CORES;
  int remO  = OUTPUT_DIM % NB_CORES;
  int prefix = 0;
  CoreLayout L[NB_CORES];

  init_platform();
  Xil_DCacheDisable();
  Xil_ICacheDisable();

  xil_printf("\r\n=== MNIST INT8 FC service mode, %d images ===\r\n", NUM_TEST_IMAGES);

  for(int c = 0; c < NB_CORES; ++c){
    Oc[c] = baseO + (c < remO);
  }

  for(int c = 0; c < NB_CORES; ++c){
    build_core_layout(prefix, Oc[c], &L[c]);
    prefix += Oc[c];
  }

  for(int c = 0; c < NB_CORES; c++){
    if(init_core(c) != 0){
      xil_printf("IP init fail c=%d\r\n", c);
      while(1);
    }

    {
      volatile u32* mb = (u32*)MB_ADDR(c);
      for(int t = 0; t < 16; ++t) mb[t] = 0;
    }

    pack_static_core_fc(c, &L[c]);
  }

  for(int c = 0; c < NB_CORES; c++){
    start_core(c);
  }

  xil_printf("All cores started once.\r\n");

  for(int img = 0; img < NUM_TEST_IMAGES; ++img){
    xil_printf("Starting img %d\r\n", img);
    memset(g_output_hw, 0, sizeof(g_output_hw));

    /* prepare fresh input and clear outputs */
    for(int c = 0; c < NB_CORES; ++c){
      volatile u32* mb = (u32*)MB_ADDR(c);
      mb[0] = 0;   /* done = 0 */
      mb[5] = 0;   /* job flag = 0 */
      pack_dynamic_input_output(c, img, &L[c]);
    }

    /* launch this image on all cores */
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

    /* gather */
    for(int c = 0; c < NB_CORES; ++c){
      gather_core_output(c, &L[c]);
    }

    /* acknowledge / return job flag to 0 */
    for(int c = 0; c < NB_CORES; ++c){
      volatile u32* mb = (u32*)MB_ADDR(c);
      mb[5] = 0;
      Xil_DCacheFlushRange(IMEM_BASE[c], BRAM_SIZE);
    }

    {
      int hw_pred = argmax10(g_output_hw);
      int ok_pred = (hw_pred == mnist_true_labels[img]);
      int ok_logits = 1;

      for(int j = 0; j < OUTPUT_DIM; ++j){
        if(g_output_hw[j] != mnist_ref_logits[img][j]){
          ok_logits = 0;
          break;
        }
      }

      xil_printf("img %d: true=%d ref=%d hw=%d  pred=%s logits=%s\r\n",
                 img,
                 (int)mnist_true_labels[img],
                 (int)mnist_ref_preds[img],
                 hw_pred,
                 ok_pred ? "OK" : "BAD",
                 ok_logits ? "OK" : "BAD");

      if(ok_pred && ok_logits) total_ok++;
    }
  }

  xil_printf("FINAL: %d/%d images matched exactly\r\n", total_ok, NUM_TEST_IMAGES);

  while(1);
  return 0;
}
t