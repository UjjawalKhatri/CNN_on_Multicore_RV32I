#include "platform.h"
#include "xil_printf.h"
#include "xil_cache.h"
#include "xil_types.h"
#include "xparameters.h"
#include "sleep.h"
#include <stdint.h>
#include <string.h>

#include "xrv32i_pp_ip_mc_gmem.h"
#include "xrv32i_pp_ip_mc_gmem_hw.h"

#include "mnist_cnn1_tflite_data.h"

typedef u32 word_type;

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

/* conv worker images */
word_type conv_code_ram_core0[] = {
		#include "cnn_conv_core0.hex"
		};
word_type conv_code_ram_core1[] = {
		#include "cnn_conv_core1.hex"
		};
word_type conv_code_ram_core2[] = {
		#include "cnn_conv_core2.hex"
		};
word_type conv_code_ram_core3[] = {
		#include "cnn_conv_core3.hex"
		};

/* fc worker images */
word_type fc_code_ram_core0[] = {
		#include "cnn_fc_core0.hex"
		};
word_type fc_code_ram_core1[] = {
		#include "cnn_fc_core1.hex"
		};
word_type fc_code_ram_core2[] = {
		#include "cnn_fc_core2.hex"
		};
word_type fc_code_ram_core3[] = {
		#include "cnn_fc_core3.hex"
		};

static XRv32i_pp_ip_mc_gmem g[NB_CORES];
static int8_t g_conv_flat[CNN1_FLAT_DIM];
static int8_t g_out_hw[10];

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
  if(!cfg) return -1;
  return XRv32i_pp_ip_mc_gmem_CfgInitialize(&g[cid], cfg);
}

static void start_core_with_image(int cid, word_type *img, int words){
  write_imem_words(cid, img, words);
  Xil_DCacheFlushRange(IMEM_BASE[cid], BRAM_SIZE);
  XRv32i_pp_ip_mc_gmem_Set_code_base_words(&g[cid], (u32)(IMEM_BASE[cid] >> 2));
  XRv32i_pp_ip_mc_gmem_Set_ip_num(&g[cid], (u32)cid);
  XRv32i_pp_ip_mc_gmem_Set_start_pc(&g[cid], 0);
  XRv32i_pp_ip_mc_gmem_Set_gmem_offset(&g[cid], 0);
  XRv32i_pp_ip_mc_gmem_Start(&g[cid]);
}

/* ---------------- conv pass ---------------- */

static void pack_conv_core(int cid, int filt){
  volatile u8* base = (u8*)ARR_ADDR(cid);

  u32 off_in  = 0;
  u32 off_w   = align_up(off_in + CNN1_IN_H * CNN1_IN_W, 64);
  u32 off_b   = align_up(off_w + CNN1_KERNEL * CNN1_KERNEL, 64);
  u32 off_out = align_up(off_b + 4u, 64);
  u32 off_ws  = align_up(off_out + CNN1_OUT_H * CNN1_OUT_W, 64);
  u32 off_os  = align_up(off_ws + 4u, 64);

  volatile int8_t *in = (volatile int8_t*)(base + off_in);
  volatile int8_t *w  = (volatile int8_t*)(base + off_w);
  volatile int32_t *b = (volatile int32_t*)(base + off_b);
  volatile int8_t *out = (volatile int8_t*)(base + off_out);
  volatile float *p_ws = (volatile float*)(base + off_ws);
  volatile float *p_os = (volatile float*)(base + off_os);

  for(int y = 0; y < CNN1_IN_H; ++y){
    for(int x = 0; x < CNN1_IN_W; ++x){
      in[y * CNN1_IN_W + x] = cnn1_input[y][x];
    }
  }

  for(int ky = 0; ky < CNN1_KERNEL; ++ky){
    for(int kx = 0; kx < CNN1_KERNEL; ++kx){
      w[ky * CNN1_KERNEL + kx] = cnn1_conv_w[filt][ky][kx];
    }
  }

  b[0] = cnn1_conv_b[filt];

  for(int i = 0; i < CNN1_OUT_H * CNN1_OUT_W; ++i) out[i] = 0;

  p_ws[0] = cnn1_conv_w_scales[filt];
  p_os[0] = CNN1_CONV_OUT_SCALE;

  {
    volatile u32 *mb = (u32*)MB_ADDR(cid);
    for(int t = 0; t < 16; ++t) mb[t] = 0;
    mb[8]  = CNN1_IN_H;
    mb[9]  = CNN1_IN_W;
    mb[10] = CNN1_OUT_H;
    mb[11] = CNN1_OUT_W;
    mb[12] = CNN1_KERNEL;
    mb[13] = (u32)filt;
    mb[14] = (u32)CNN1_INPUT_ZERO;
    mb[15] = (u32)CNN1_CONV_OUT_ZERO;
  }

  Xil_DCacheFlushRange(IMEM_BASE[cid], BRAM_SIZE);
}

static void run_conv_all(void){
  for(int c = 0; c < NB_CORES; ++c){
    volatile u32 *mb = (u32*)MB_ADDR(c);
    mb[0] = 0;
    mb[5] = 0;
    Xil_DCacheFlushRange(IMEM_BASE[c], BRAM_SIZE);
  }
  for(int c = 0; c < NB_CORES; ++c){
    volatile u32 *mb = (u32*)MB_ADDR(c);
    mb[5] = 1;
    Xil_DCacheFlushRange(IMEM_BASE[c], BRAM_SIZE);
  }
  for(int c = 0; c < NB_CORES; ++c){
    volatile u32 *mb = (u32*)MB_ADDR(c);
    while(mb[0] != 1U){
      Xil_DCacheInvalidateRange(IMEM_BASE[c], BRAM_SIZE);
      usleep(1000);
    }
  }
  for(int c = 0; c < NB_CORES; ++c){
    volatile u32 *mb = (u32*)MB_ADDR(c);
    mb[5] = 0;
    Xil_DCacheFlushRange(IMEM_BASE[c], BRAM_SIZE);
  }
}

static int gather_conv_and_check(void){
  int errors = 0;

  for(int c = 0; c < NB_CORES; ++c){
    volatile u8* base = (u8*)ARR_ADDR(c);
    u32 off_in  = 0;
    u32 off_w   = align_up(off_in + CNN1_IN_H * CNN1_IN_W, 64);
    u32 off_b   = align_up(off_w + CNN1_KERNEL * CNN1_KERNEL, 64);
    u32 off_out = align_up(off_b + 4u, 64);

    Xil_DCacheInvalidateRange(IMEM_BASE[c], BRAM_SIZE);

    {
      const int8_t *out = (const int8_t*)(base + off_out);

      for(int oy = 0; oy < CNN1_OUT_H; ++oy){
        for(int ox = 0; ox < CNN1_OUT_W; ++ox){
          int idx_local = oy * CNN1_OUT_W + ox;

          /* keep conv correctness check */
          if(out[idx_local] != cnn1_conv_out_ref[c][oy][ox]) errors++;

          /* NHWC flatten: y, x, channel */
          g_conv_flat[(oy * CNN1_OUT_W + ox) * CNN1_NUM_FILTERS + c] = out[idx_local];
        }
      }
    }
  }

  return errors;
}

static void pack_fc_core(int cid, int out_start, int out_count){
  volatile u8* base = (u8*)ARR_ADDR(cid);

  u32 off_in       = 0;
  u32 off_w        = align_up(off_in + CNN1_FLAT_DIM, 64);
  u32 off_b        = align_up(off_w + CNN1_FLAT_DIM * out_count, 64);
  u32 off_out      = align_up(off_b + out_count * 4u, 64);
  u32 off_scales   = align_up(off_out + out_count, 64);
  u32 off_wzeros   = align_up(off_scales + out_count * 4u, 64);
  u32 off_inscale  = align_up(off_wzeros + out_count * 4u, 64);
  u32 off_outscale = align_up(off_inscale + 4u, 64);

  volatile int8_t *in = (volatile int8_t*)(base + off_in);
  volatile int8_t *w  = (volatile int8_t*)(base + off_w);
  volatile int32_t *b = (volatile int32_t*)(base + off_b);
  volatile int8_t *out = (volatile int8_t*)(base + off_out);
  volatile float *scales = (volatile float*)(base + off_scales);
  volatile int32_t *wzeros = (volatile int32_t*)(base + off_wzeros);
  volatile float *in_scale = (volatile float*)(base + off_inscale);
  volatile float *out_scale = (volatile float*)(base + off_outscale);

  for(int i = 0; i < CNN1_FLAT_DIM; ++i) in[i] = g_conv_flat[i];

  for(int j = 0; j < out_count; ++j){
    int gj = out_start + j;
    for(int i = 0; i < CNN1_FLAT_DIM; ++i){
      w[j * CNN1_FLAT_DIM + i] = cnn1_fc_w[gj][i];
    }
    b[j] = cnn1_fc_b[gj];
    scales[j] = cnn1_fc_w_scales[gj];
    wzeros[j] = cnn1_fc_w_zero_points[gj];
    out[j] = 0;
  }

  in_scale[0] = CNN1_CONV_OUT_SCALE;
  out_scale[0] = CNN1_OUTPUT_SCALE;

  {
    volatile u32 *mb = (u32*)MB_ADDR(cid);
    for(int t = 0; t < 16; ++t) mb[t] = 0;
    mb[8]  = CNN1_FLAT_DIM;
    mb[9]  = (u32)out_start;
    mb[10] = (u32)out_count;
    mb[11] = (u32)CNN1_CONV_OUT_ZERO;
    mb[12] = (u32)CNN1_OUTPUT_ZERO;
  }

  Xil_DCacheFlushRange(IMEM_BASE[cid], BRAM_SIZE);
}
static void run_fc_all(void){
  for(int c = 0; c < NB_CORES; ++c){
    volatile u32 *mb = (u32*)MB_ADDR(c);
    mb[0] = 0;
    mb[5] = 0;
    Xil_DCacheFlushRange(IMEM_BASE[c], BRAM_SIZE);
  }
  for(int c = 0; c < NB_CORES; ++c){
    volatile u32 *mb = (u32*)MB_ADDR(c);
    mb[5] = 1;
    Xil_DCacheFlushRange(IMEM_BASE[c], BRAM_SIZE);
  }
  for(int c = 0; c < NB_CORES; ++c){
    volatile u32 *mb = (u32*)MB_ADDR(c);
    while(mb[0] != 1U){
      Xil_DCacheInvalidateRange(IMEM_BASE[c], BRAM_SIZE);
      usleep(1000);
    }
  }
  for(int c = 0; c < NB_CORES; ++c){
    volatile u32 *mb = (u32*)MB_ADDR(c);
    mb[5] = 0;
    Xil_DCacheFlushRange(IMEM_BASE[c], BRAM_SIZE);
  }
}

static int gather_fc_and_check(void){
  int errors = 0;
  int Oc[NB_CORES];
  int baseO = 10 / NB_CORES;
  int remO  = 10 % NB_CORES;
  int prefix = 0;

  for(int c = 0; c < NB_CORES; ++c){
    Oc[c] = baseO + (c < remO);
  }

  for(int c = 0; c < NB_CORES; ++c){
    volatile u8* base = (u8*)ARR_ADDR(c);
    u32 off_in       = 0;
    u32 off_w        = align_up(off_in + CNN1_FLAT_DIM, 64);
    u32 off_b        = align_up(off_w + CNN1_FLAT_DIM * Oc[c], 64);
    u32 off_out      = align_up(off_b + Oc[c] * 4u, 64);

    Xil_DCacheInvalidateRange(IMEM_BASE[c], BRAM_SIZE);
    {
      const int8_t *out = (const int8_t*)(base + off_out);
      for(int j = 0; j < Oc[c]; ++j){
        g_out_hw[prefix + j] = out[j];
      }
    }
    prefix += Oc[c];
  }

  for(int i = 0; i < 10; ++i){
    if(g_out_hw[i] != cnn1_output_ref[i]) errors++;
  }

  return errors;
}

static int argmax10(const int8_t *a){
  int idx = 0;
  for(int i = 1; i < 10; ++i){
    if(a[i] > a[idx]) idx = i;
  }
  return idx;
}

int main(void){
  int conv_errors, fc_errors;
  int Oc[NB_CORES];
  int baseO = 10 / NB_CORES;
  int remO  = 10 % NB_CORES;
  int prefix = 0;

  init_platform();
  Xil_DCacheDisable();
  Xil_ICacheDisable();

  xil_printf("\r\n=== CNN one-image TFLite int8 run ===\r\n");
  xil_printf("True label : %d\r\n", CNN1_TRUE_LABEL);
  xil_printf("Ref pred   : %d\r\n", CNN1_REF_PRED);

  memset(g_conv_flat, 0, sizeof(g_conv_flat));
  memset(g_out_hw, 0, sizeof(g_out_hw));

  for(int c = 0; c < NB_CORES; ++c){
    if(init_core(c) != 0){
      xil_printf("IP init fail c=%d\r\n", c);
      while(1);
    }
  }

  /* ---------- CONV PASS ---------- */
  for(int c = 0; c < NB_CORES; ++c){
    start_core_with_image(c,
      (c==0)? conv_code_ram_core0 :
      (c==1)? conv_code_ram_core1 :
      (c==2)? conv_code_ram_core2 :
              conv_code_ram_core3,
      (c==0)? (int)(sizeof(conv_code_ram_core0)/4) :
      (c==1)? (int)(sizeof(conv_code_ram_core1)/4) :
      (c==2)? (int)(sizeof(conv_code_ram_core2)/4) :
              (int)(sizeof(conv_code_ram_core3)/4)
    );
    pack_conv_core(c, c);
  }

  run_conv_all();
  conv_errors = gather_conv_and_check();

  xil_printf("Conv compare: %s (%d mismatches)\r\n",
             conv_errors ? "BAD" : "OK", conv_errors);

  /* ---------- FC PASS ---------- */
  for(int c = 0; c < NB_CORES; ++c){
    Oc[c] = baseO + (c < remO);
  }

  for(int c = 0; c < NB_CORES; ++c){
    start_core_with_image(c,
      (c==0)? fc_code_ram_core0 :
      (c==1)? fc_code_ram_core1 :
      (c==2)? fc_code_ram_core2 :
              fc_code_ram_core3,
      (c==0)? (int)(sizeof(fc_code_ram_core0)/4) :
      (c==1)? (int)(sizeof(fc_code_ram_core1)/4) :
      (c==2)? (int)(sizeof(fc_code_ram_core2)/4) :
              (int)(sizeof(fc_code_ram_core3)/4)
    );
    pack_fc_core(c, prefix, Oc[c]);
    prefix += Oc[c];
  }

  run_fc_all();
  fc_errors = gather_fc_and_check();

  xil_printf("FC output compare: %s (%d mismatches)\r\n",
             fc_errors ? "BAD" : "OK", fc_errors);

  xil_printf("HW out  : ");
  for(int i = 0; i < 10; ++i) xil_printf("%d ", (int)g_out_hw[i]);
  xil_printf("\r\n");

  xil_printf("REF out : ");
  for(int i = 0; i < 10; ++i) xil_printf("%d ", (int)cnn1_output_ref[i]);
  xil_printf("\r\n");

  xil_printf("HW pred : %d\r\n", argmax10(g_out_hw));

  while(1);
  return 0;
}
