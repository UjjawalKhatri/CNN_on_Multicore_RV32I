#include <stdint.h>

#define MB_OFS   0x2000u
#define ARR_OFS  0x2100u

#define MB32     ((volatile uint32_t*)MB_OFS)
#define ARR_BASE ((volatile uint8_t*)ARR_OFS)

static inline uint32_t align_up(uint32_t x, uint32_t a){
  uint32_t m = a - 1u;
  return (x + m) & ~m;
}

static inline int8_t sat_i8(int32_t x){
  if (x > 127) return 127;
  if (x < -128) return -128;
  return (int8_t)x;
}

static inline int32_t round_nearest_float_to_int(float x){
  return (x >= 0.0f) ? (int32_t)(x + 0.5f) : (int32_t)(x - 0.5f);
}

int main(void){
  MB32[0] = 0;
  MB32[5] = 0;

  while(1){
    uint32_t input_dim, out_start, out_count;
    int32_t input_zero, output_zero;
    uint32_t off_in, off_w, off_b, off_out, off_scales, off_wzeros, off_inscale, off_outscale;
    float conv_out_scale, output_scale;

    while(MB32[5] != 1u){
    }

    input_dim   = MB32[8];
    out_start   = MB32[9];
    out_count   = MB32[10];
    input_zero  = (int32_t)MB32[11];
    output_zero = (int32_t)MB32[12];

    (void)out_start;

    off_in       = 0;
    off_w        = align_up(off_in + input_dim, 64);
    off_b        = align_up(off_w + input_dim * out_count, 64);
    off_out      = align_up(off_b + out_count * 4u, 64);
    off_scales   = align_up(off_out + out_count, 64);
    off_wzeros   = align_up(off_scales + out_count * 4u, 64);
    off_inscale  = align_up(off_wzeros + out_count * 4u, 64);
    off_outscale = align_up(off_inscale + 4u, 64);

    {
      const int8_t  *in   = (const int8_t *)(ARR_BASE + off_in);
      const int8_t  *w    = (const int8_t *)(ARR_BASE + off_w);
      const int32_t *bias = (const int32_t*)(ARR_BASE + off_b);
      int8_t        *out  = (int8_t *)(ARR_BASE + off_out);
      const float   *w_scales = (const float *)(ARR_BASE + off_scales);
      const int32_t *w_zeros  = (const int32_t *)(ARR_BASE + off_wzeros);
      const float   *p_in_scale = (const float *)(ARR_BASE + off_inscale);
      const float   *p_out_scale = (const float *)(ARR_BASE + off_outscale);

      conv_out_scale = p_in_scale[0];
      output_scale   = p_out_scale[0];

      for(uint32_t j = 0; j < out_count; ++j){
        int32_t acc = bias[j];
        const int8_t *wj = w + j * input_dim;
        int32_t w_zero = w_zeros[j];
        float multiplier = (conv_out_scale * w_scales[j]) / output_scale;

        for(uint32_t i = 0; i < input_dim; ++i){
          int32_t in_q = (int32_t)in[i];
          int32_t w_q  = (int32_t)wj[i];
          acc += (in_q - input_zero) * (w_q - w_zero);
        }

        {
          int32_t q = round_nearest_float_to_int(((float)acc) * multiplier) + output_zero;
          out[j] = sat_i8(q);
        }
      }
    }

    MB32[0] = 1;

    while(MB32[5] != 0u){
    }
  }

  return 0;
}