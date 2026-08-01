#include <stdint.h>

#define MB_OFS   0x2000u
#define ARR_OFS  0x2100u

#define MB32     ((volatile uint32_t*)MB_OFS)
#define ARR_BASE ((volatile uint8_t*)ARR_OFS)

/*
Mailbox
MB[0]  = done
MB[5]  = start/job flag

Conv-specific:
MB[8]  = in_h
MB[9]  = in_w
MB[10] = out_h
MB[11] = out_w
MB[12] = kernel
MB[13] = filter_idx (for debug only)
MB[14] = input_zero
MB[15] = conv_out_zero

Data layout at ARR_OFS:
[input image int8: in_h * in_w]
[conv weights int8: kernel * kernel]
[conv bias int32: 1]
[conv output int8: out_h * out_w]

Float params after that:
[conv_w_scale float: 1]
[conv_out_scale float: 1]
*/

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
    uint32_t in_h, in_w, out_h, out_w, kernel;
    int32_t input_zero, conv_out_zero;
    uint32_t off_in, off_w, off_b, off_out, off_ws, off_os;
    float conv_w_scale, conv_out_scale, multiplier;

    while(MB32[5] != 1u){
    }

    in_h  = MB32[8];
    in_w  = MB32[9];
    out_h = MB32[10];
    out_w = MB32[11];
    kernel = MB32[12];

    input_zero    = (int32_t)MB32[14];
    conv_out_zero = (int32_t)MB32[15];

    off_in  = 0;
    off_w   = align_up(off_in + in_h * in_w, 64);
    off_b   = align_up(off_w + kernel * kernel, 64);
    off_out = align_up(off_b + 4u, 64);
    off_ws  = align_up(off_out + out_h * out_w, 64);
    off_os  = align_up(off_ws + 4u, 64);

    {
      const int8_t  *in  = (const int8_t *)(ARR_BASE + off_in);
      const int8_t  *w   = (const int8_t *)(ARR_BASE + off_w);
      const int32_t *b   = (const int32_t*)(ARR_BASE + off_b);
      int8_t        *out = (int8_t *)(ARR_BASE + off_out);
      const float   *p_ws = (const float *)(ARR_BASE + off_ws);
      const float   *p_os = (const float *)(ARR_BASE + off_os);

      conv_w_scale  = p_ws[0];
      conv_out_scale = p_os[0];
      multiplier = conv_w_scale / conv_out_scale; /* input scale is 1.0 */

      for(uint32_t oy = 0; oy < out_h; ++oy){
        for(uint32_t ox = 0; ox < out_w; ++ox){
          int32_t acc = b[0];

          for(uint32_t ky = 0; ky < kernel; ++ky){
            for(uint32_t kx = 0; kx < kernel; ++kx){
              int32_t in_q = (int32_t)in[(oy + ky) * in_w + (ox + kx)];
              int32_t w_q  = (int32_t)w[ky * kernel + kx]; /* weight zero assumed 0 */
              acc += (in_q - input_zero) * w_q;
            }
          }

          /* TFLite conv output has ReLU already due to model activation */
          int32_t q = round_nearest_float_to_int(((float)acc) * multiplier) + conv_out_zero;
          if(q < conv_out_zero) q = conv_out_zero; /* ReLU in quantized domain because zero real value maps to zero point */
          out[oy * out_w + ox] = sat_i8(q);
        }
      }
    }

    MB32[0] = 1;

    while(MB32[5] != 0u){
    }
  }

  return 0;
}