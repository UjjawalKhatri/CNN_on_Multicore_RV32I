#include <stdint.h>

#define MB_OFS   0x2000u
#define ARR_OFS  0x2100u

#define MB32     ((volatile uint32_t*)MB_OFS)
#define ARR_BASE ((volatile uint8_t*)ARR_OFS)

static inline uint32_t align_up(uint32_t x, uint32_t a){
  uint32_t m = a - 1u;
  return (x + m) & ~m;
}

/* 64-bit CSR reads on RV32 */
static inline uint64_t read_mcycle(void){
  uint32_t hi0, lo, hi1;
  do {
    __asm__ volatile("rdcycleh %0" : "=r"(hi0));
    __asm__ volatile("rdcycle  %0" : "=r"(lo));
    __asm__ volatile("rdcycleh %0" : "=r"(hi1));
  } while (hi0 != hi1);
  return ((uint64_t)hi1 << 32) | lo;
}

static inline uint64_t read_minstret(void){
  uint32_t hi0, lo, hi1;
  do {
    __asm__ volatile("rdinstreth %0" : "=r"(hi0));
    __asm__ volatile("rdinstret  %0" : "=r"(lo));
    __asm__ volatile("rdinstreth %0" : "=r"(hi1));
  } while (hi0 != hi1);
  return ((uint64_t)hi1 << 32) | lo;
}

int main(void){
  uint32_t input_dim, output_dim, out_start, out_count;
  uint32_t requant_shift, relu_en;

  MB32[15] = 0x1111;   /* entered main */
  MB32[0]  = 0;        /* done */
  MB32[5]  = 0;        /* job flag */

  /* Wait once for valid configuration */
  do {
    input_dim      = MB32[8];
    output_dim     = MB32[9];
    out_start      = MB32[10];
    out_count      = MB32[11];
    requant_shift  = MB32[12];
    relu_en        = MB32[13];
  } while (!(input_dim && output_dim && out_count));

  MB32[14] = 0x2222;   /* mailbox config seen */

  (void)output_dim;
  (void)out_start;

  {
    uint32_t off_in   = 0;
    uint32_t off_w    = align_up(off_in + input_dim, 64);
    uint32_t off_b    = align_up(off_w + input_dim * out_count, 64);
    uint32_t off_out  = align_up(off_b + out_count * 4u, 64);

    const int8_t  *in   = (const int8_t *)(ARR_BASE + off_in);
    const int8_t  *w    = (const int8_t *)(ARR_BASE + off_w);
    const int32_t *bias = (const int32_t*)(ARR_BASE + off_b);
    int32_t       *out  = (int32_t *)(ARR_BASE + off_out);

    while(1){
      uint64_t c0, c1, i0, i1;

      /* wait for host to post a job */
      while(MB32[5] != 1u){
      }

      MB32[0]  = 0;
      MB32[14] = 0x3001;   /* started job */

      c0 = read_mcycle();
      i0 = read_minstret();

      for (uint32_t j = 0; j < out_count; ++j){
        int32_t acc = bias[j];
        const int8_t *wj = w + j * input_dim;

        for (uint32_t i = 0; i < input_dim; ++i){
          acc += (int32_t)in[i] * (int32_t)wj[i];
        }

        if (requant_shift){
          int32_t rnd = 1 << (requant_shift - 1);
          acc = (acc + rnd) >> requant_shift;
        }

        if (relu_en && acc < 0) acc = 0;

        out[j] = acc;
      }

      c1 = read_mcycle();
      i1 = read_minstret();

      MB32[1] = (uint32_t)((i1 - i0) & 0xffffffffu);
      MB32[2] = (uint32_t)((i1 - i0) >> 32);
      MB32[3] = (uint32_t)((c1 - c0) & 0xffffffffu);
      MB32[4] = (uint32_t)((c1 - c0) >> 32);

      MB32[14] = 0x3333;   /* compute done */
      MB32[0]  = 1;        /* done */

      /* wait for host to acknowledge / clear job flag */
      while(MB32[5] != 0u){
      }

      MB32[14] = 0x3000;   /* idle again */
    }
  }

  return 0;
}