#define SINGLE_FILE
/**
  ******************************************************************************
  * @file    number.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    2 february 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef __NUMBER_H__
#define __NUMBER_H__

#include <stdint.h>

#define FIXED_POINT	9	// Fixed point scaling factor, set to 0 when using floating point
#define NUMBER_MIN	-32768	// Max value for this numeric type
#define NUMBER_MAX	32767	// Min value for this numeric type
typedef int16_t number_t;		// Standard size numeric type used for weights and activations
typedef int32_t long_number_t;	// Long numeric type used for intermediate results

#ifndef min
static inline long_number_t min(long_number_t a, long_number_t b) {
	if (a <= b)
		return a;
	return b;
}
#endif

#ifndef max
static inline long_number_t max(long_number_t a, long_number_t b) {
	if (a >= b)
		return a;
	return b;
}
#endif

#if FIXED_POINT > 0 // Scaling/clamping for fixed-point representation
static inline long_number_t scale_number_t(long_number_t number) {
	return number >> FIXED_POINT;
}
static inline number_t clamp_to_number_t(long_number_t number) {
	return (number_t) max(NUMBER_MIN, min(NUMBER_MAX, number));
}
#else // No scaling/clamping required for floating-point
static inline long_number_t scale_number_t(long_number_t number) {
	return number;
}
static inline number_t clamp_to_number_t(long_number_t number) {
	return (number_t) number;
}
#endif


#endif //__NUMBER_H__
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  1
#define INPUT_SAMPLES   16000
#define POOL_SIZE       20
#define POOL_STRIDE     20
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       800
#define CONV_FILTERS        8
#define CONV_KERNEL_SIZE    40
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    1
#define CONV_FILTERS      8
#define CONV_KERNEL_SIZE  40


const int16_t conv1d_bias[CONV_FILTERS] = {-11, 0, 71, 64, 63, 65, 85, 3}
;

const int16_t conv1d_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{0, -50, 26, -43, 16, 48, 43, 39, -59, 29, 10, 53, 0, -70, -70, -48, -18, 45, 0, -42, 58, 41, 45, -53, -54, -44, 33, 30, 11, -21, 11, -22, 23, 18, 17, 25, 4, -50, 63, 8}
}
, {{31, -54, -14, 4, 51, 32, 47, 24, 16, 13, 1, 50, 0, 24, -19, -28, -17, 12, -71, -34, -60, -65, -50, 26, 14, -45, 17, 17, -30, -72, -23, 33, 13, -23, -27, 27, -13, -53, -37, 7}
}
, {{-40, 29, 15, 40, -52, -61, 1, -59, 42, -9, 33, -39, -48, 0, -60, -19, -35, 4, -3, 36, -17, -2, 16, -56, -70, -5, -22, -42, -50, 43, 29, 19, -55, 60, 49, 0, 45, -46, 54, 51}
}
, {{45, 48, 51, 53, -26, 66, 49, 65, -28, -53, 14, -50, 64, -38, -21, -14, -33, -46, 50, 20, -28, -9, 22, -13, -28, -19, 45, -7, 4, 62, 12, -15, -31, 59, -18, 58, -47, -52, 1, -44}
}
, {{39, -31, 43, -8, 27, -63, -59, 28, 39, 60, -64, 38, 19, -34, -64, -21, -63, 33, 21, -29, 26, 35, 4, 49, 16, -37, -57, -37, 32, 20, -64, 17, -35, -11, 16, 21, 14, -34, -13, -54}
}
, {{39, -49, -7, 16, 6, -48, 22, -9, 4, 42, 40, -40, 58, -57, -19, 48, 3, 29, 44, -36, 3, 0, -26, -2, 46, -25, 38, 12, 49, 15, 47, 6, -40, -8, -10, 48, -51, -49, 44, 68}
}
, {{-42, 30, -40, 31, -59, 32, 58, 33, 39, 10, -1, -20, -37, -12, 12, 55, 12, 36, 14, -32, 18, -57, -9, -52, -53, -47, 55, -20, -8, 22, -61, 22, 7, -66, 9, 7, -42, 2, 56, -18}
}
, {{26, 66, -24, 62, -26, 51, 59, -37, -42, -52, 32, -45, -30, -9, -6, 17, 47, -19, 38, -24, -21, 46, -12, -43, 56, 31, -41, -91, -102, -41, -96, -16, -89, -39, -85, 15, -19, -50, 38, 30}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  8
#define INPUT_SAMPLES   761
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_1_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_1(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      8
#define INPUT_SAMPLES       190
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_1_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_1(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    8
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_1_bias[CONV_FILTERS] = {-27, -39, 52, 28, -27, -4, 56, -6, 57, 57, 56, -6, 56, 49, 54, 56}
;

const int16_t conv1d_1_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{106, 18, 132}
, {82, 8, 71}
, {42, -44, 14}
, {-78, -93, 68}
, {-27, -136, 22}
, {-122, 144, 86}
, {70, 5, 94}
, {-31, -57, -109}
}
, {{-7, 25, 97}
, {11, -71, 135}
, {72, 120, -64}
, {-69, -58, 71}
, {-70, 23, 27}
, {83, -101, 60}
, {38, 26, 132}
, {-105, 49, -96}
}
, {{-96, -2, 103}
, {-96, -145, 113}
, {104, -55, 119}
, {84, -134, 91}
, {16, -14, 51}
, {-82, 42, 110}
, {86, 20, 34}
, {-100, -142, -101}
}
, {{105, -12, 41}
, {31, -4, 77}
, {-46, 43, 87}
, {-83, -74, 82}
, {50, -98, 87}
, {14, 85, 7}
, {-35, -62, -2}
, {130, -160, -29}
}
, {{56, 138, -42}
, {-51, -57, 41}
, {-112, -5, -12}
, {-35, 5, 88}
, {123, -122, 17}
, {58, -64, 36}
, {30, 18, -20}
, {-102, 37, -100}
}
, {{79, 104, 89}
, {93, 129, -53}
, {-95, 110, -102}
, {66, 22, -44}
, {-6, -55, 89}
, {-66, -139, -39}
, {-142, -69, 89}
, {-105, -138, -135}
}
, {{59, -122, -106}
, {-156, -22, -79}
, {-54, 54, 68}
, {106, 18, -104}
, {155, 65, -13}
, {25, -2, 129}
, {167, 137, 35}
, {-75, 84, 73}
}
, {{113, -137, 118}
, {-99, 101, 67}
, {-11, -98, -74}
, {-144, -91, -113}
, {-50, 72, 21}
, {21, -47, -35}
, {90, -17, -146}
, {-51, -131, 22}
}
, {{-78, -82, 6}
, {-3, 82, 150}
, {-43, -90, 81}
, {38, 127, 47}
, {-28, -21, 143}
, {-132, 144, 117}
, {-90, 59, 63}
, {-96, -25, 15}
}
, {{109, -25, -61}
, {-93, -129, 49}
, {157, 131, 102}
, {63, -11, 114}
, {-15, 142, -126}
, {-11, 16, 139}
, {-80, 155, 86}
, {-162, 2, 59}
}
, {{-20, -47, 143}
, {35, 110, 62}
, {-68, 19, 95}
, {78, -106, -96}
, {-57, -62, -69}
, {66, 88, -71}
, {84, 57, 148}
, {-81, -89, 95}
}
, {{30, 22, -3}
, {101, -107, 111}
, {-50, -118, -117}
, {135, 141, -145}
, {-20, 103, 137}
, {-112, 104, 0}
, {61, 142, -24}
, {-31, 56, -27}
}
, {{-85, 26, -53}
, {-138, -128, 30}
, {-81, 164, -53}
, {129, 130, -64}
, {141, 152, -62}
, {56, -20, 88}
, {17, 177, -60}
, {-185, -99, -104}
}
, {{122, 46, 62}
, {-86, 10, 1}
, {-76, -126, 153}
, {-63, 32, -101}
, {126, 128, 4}
, {53, 65, -109}
, {-37, 64, -74}
, {91, -108, -91}
}
, {{-76, 47, -122}
, {-6, -14, 75}
, {-64, -121, 43}
, {45, 65, 44}
, {33, 152, 0}
, {136, 77, -67}
, {12, -51, 84}
, {-93, -123, 155}
}
, {{-117, -83, -51}
, {-72, 40, 67}
, {-37, 31, -26}
, {51, -30, -63}
, {147, 78, 162}
, {73, -56, -32}
, {138, 176, 58}
, {-73, 114, 56}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  16
#define INPUT_SAMPLES   188
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_2_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_2(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       47
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_2_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_2(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    16
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_2_bias[CONV_FILTERS] = {43, 2, -6, 29, 45, -24, 49, 4, 41, 4, 24, 51, 11, -4, 44, -24, 0, 43, -6, 13, -7, 42, 41, -18, -7, -9, 57, -14, -7, -5, 36, -35}
;

const int16_t conv1d_2_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-60, 2, 31}
, {16, -56, -102}
, {97, 69, 13}
, {88, -12, 106}
, {-21, 87, 21}
, {23, -46, -50}
, {36, -10, 38}
, {-5, 7, 30}
, {5, -55, 61}
, {90, -49, -61}
, {27, 97, 72}
, {106, 18, -76}
, {55, 41, 113}
, {28, -57, -63}
, {100, 109, 88}
, {22, 60, 45}
}
, {{-22, -30, 57}
, {-41, -89, -106}
, {-85, -51, 67}
, {17, 36, 19}
, {77, -56, -8}
, {28, -74, -94}
, {0, 28, 75}
, {28, -110, 76}
, {31, -86, -80}
, {-88, 80, 3}
, {93, 76, 56}
, {-29, 80, -2}
, {3, 41, -36}
, {-10, -66, -63}
, {45, -63, -103}
, {83, 47, -95}
}
, {{-10, 89, 52}
, {18, 20, 14}
, {-41, 1, 87}
, {-43, -55, -74}
, {18, -16, 51}
, {-63, 24, 27}
, {74, 3, 70}
, {60, 77, -77}
, {78, 54, 39}
, {-32, -39, 59}
, {-82, 5, 5}
, {-41, -26, 15}
, {-64, -97, -95}
, {-101, 94, -76}
, {-105, 16, -21}
, {-18, -56, -68}
}
, {{-11, 81, 46}
, {29, 89, -9}
, {-17, -66, -100}
, {-20, -21, -33}
, {67, -69, 64}
, {77, 48, -8}
, {10, -50, -102}
, {86, -8, 71}
, {94, -84, 27}
, {24, -20, 97}
, {-60, 25, -52}
, {-31, -100, -86}
, {-97, 14, 65}
, {-16, -50, -55}
, {16, -66, 51}
, {133, 2, -75}
}
, {{3, -55, -52}
, {-101, -74, -42}
, {43, 126, 115}
, {-58, 104, 70}
, {55, -110, -19}
, {60, 44, 25}
, {3, -62, 92}
, {-25, -68, -74}
, {40, 44, 118}
, {99, 50, 68}
, {39, -60, 54}
, {19, 100, -3}
, {65, 3, -33}
, {-11, 18, -32}
, {-40, -64, 87}
, {37, 25, 82}
}
, {{-55, -13, -51}
, {-118, -94, -80}
, {42, -10, 78}
, {-96, 61, 4}
, {-56, 54, 69}
, {50, 42, -71}
, {12, -75, -38}
, {73, -68, -38}
, {7, 50, -24}
, {87, 7, 21}
, {53, -107, 66}
, {-71, 97, -1}
, {54, -52, 80}
, {-69, -5, -82}
, {-28, 27, 89}
, {-44, 44, -63}
}
, {{59, 49, -1}
, {76, 15, 23}
, {87, 100, 11}
, {-53, 36, -68}
, {-39, -108, 9}
, {-36, -81, -46}
, {-16, -34, 26}
, {-1, -2, -70}
, {-4, 7, 64}
, {-72, -63, 111}
, {34, 56, 92}
, {60, -43, 6}
, {1, 115, 57}
, {2, 114, 41}
, {38, -57, -48}
, {74, 60, 69}
}
, {{-4, 60, 79}
, {70, 100, -3}
, {-80, 4, 36}
, {-80, 68, -64}
, {-32, -31, -73}
, {-62, -87, 42}
, {-63, -46, 51}
, {31, 1, -99}
, {-49, 91, -106}
, {97, -15, 70}
, {68, -10, -92}
, {12, 103, -14}
, {-16, -109, 15}
, {25, -50, 58}
, {60, -97, -101}
, {122, 112, -71}
}
, {{5, -96, 3}
, {-72, 9, -30}
, {94, -39, 7}
, {10, 86, -60}
, {-24, 88, 97}
, {17, -91, 7}
, {1, -45, -57}
, {11, 5, 64}
, {34, 62, 85}
, {98, -79, 89}
, {-53, -10, 100}
, {-101, 46, 75}
, {3, -41, 3}
, {14, 27, -78}
, {48, -51, 22}
, {87, -11, 78}
}
, {{-62, -97, 96}
, {30, 79, 29}
, {15, -78, -76}
, {-97, -100, 73}
, {50, 29, 48}
, {14, 25, 63}
, {-30, -76, 24}
, {89, -30, 65}
, {48, 1, 14}
, {-37, -16, -77}
, {-30, 34, 3}
, {-50, -69, 45}
, {-72, -78, -39}
, {-37, 34, 34}
, {6, -1, 71}
, {41, 77, 73}
}
, {{15, -90, 92}
, {38, 109, -61}
, {-46, -58, -58}
, {13, -26, 48}
, {-60, -65, -100}
, {-12, -36, 96}
, {-80, -40, 100}
, {5, 92, 85}
, {78, 12, 87}
, {-48, 0, 50}
, {109, -94, -72}
, {90, 52, 105}
, {-26, -62, -57}
, {-43, 20, -79}
, {-33, -8, -71}
, {-67, 73, 48}
}
, {{-117, 27, -30}
, {-23, -72, -92}
, {111, -27, -14}
, {-15, 34, -61}
, {2, 35, 86}
, {63, -94, 109}
, {-8, -28, 22}
, {-32, 82, -4}
, {78, 29, -63}
, {-30, 67, -34}
, {80, 39, 62}
, {8, 47, -9}
, {-15, 116, 28}
, {94, 107, 126}
, {69, 75, 101}
, {-22, -31, -23}
}
, {{-48, 46, 85}
, {-100, -18, 93}
, {-61, 19, -89}
, {-88, -72, -25}
, {83, 90, 88}
, {-11, 25, -64}
, {-75, -7, -46}
, {63, -6, 44}
, {-110, -23, -85}
, {-10, 46, 78}
, {-65, -88, 13}
, {-87, -67, -99}
, {57, 21, -28}
, {-41, -48, -99}
, {65, 23, 13}
, {78, 88, -13}
}
, {{-92, 87, 20}
, {73, -42, -50}
, {-70, -108, -54}
, {-16, 13, 4}
, {4, -23, -64}
, {54, -34, 66}
, {-69, 81, -28}
, {69, 17, -12}
, {19, -47, -71}
, {-53, 52, 39}
, {82, 91, 61}
, {-25, 68, -97}
, {-84, 10, -25}
, {87, -61, -3}
, {44, -13, -87}
, {-26, -40, 7}
}
, {{-22, -15, -50}
, {-64, -16, 16}
, {67, 46, -1}
, {23, -36, 90}
, {-82, -78, 11}
, {-87, 95, -38}
, {97, 124, 96}
, {-1, -56, -32}
, {82, 11, 43}
, {-10, 118, -1}
, {-35, 124, 67}
, {-3, 32, -40}
, {12, 26, 112}
, {45, 109, -40}
, {57, 14, 103}
, {128, 38, 135}
}
, {{69, -72, -50}
, {-31, 48, 55}
, {59, 13, -39}
, {-27, -68, 85}
, {30, 66, -108}
, {-77, -19, -50}
, {42, -34, -88}
, {-95, -14, 17}
, {-15, 70, -82}
, {-65, 50, 36}
, {57, 96, -89}
, {-56, -21, -67}
, {-84, -8, -13}
, {94, -20, -24}
, {29, 31, 42}
, {-15, 58, 84}
}
, {{-83, -81, 31}
, {-17, -79, 24}
, {12, -92, -20}
, {-30, -93, -58}
, {-60, -100, 0}
, {-31, -5, 94}
, {-92, -10, -34}
, {-40, -50, 17}
, {-75, 100, -51}
, {14, 3, -10}
, {16, -62, 17}
, {-95, 73, 73}
, {-72, -17, -18}
, {7, -60, 15}
, {-59, 52, 58}
, {27, -93, -98}
}
, {{41, -85, -70}
, {25, -79, 13}
, {59, 80, 29}
, {-87, -8, -16}
, {-22, 93, 20}
, {-55, 104, -52}
, {11, 80, 26}
, {0, 80, -68}
, {51, -6, -85}
, {36, 89, 111}
, {26, 81, 90}
, {2, -64, -71}
, {-13, 8, 62}
, {36, 20, 13}
, {1, -84, 12}
, {-43, 114, 107}
}
, {{92, 96, 26}
, {-99, -56, -41}
, {5, -95, -17}
, {5, -85, -86}
, {-35, 95, -50}
, {-30, -27, -91}
, {-32, -106, -41}
, {73, -60, 75}
, {83, -48, -59}
, {-39, -21, -49}
, {57, 89, -23}
, {-100, -69, 35}
, {28, -21, -27}
, {-65, 99, 44}
, {-5, -109, -61}
, {78, 79, 70}
}
, {{-68, 63, 14}
, {74, -103, 43}
, {-14, -43, 0}
, {85, 46, 78}
, {41, 59, -21}
, {81, 61, -101}
, {94, 35, 103}
, {97, 66, -49}
, {72, 71, 102}
, {47, -96, -36}
, {7, -39, 3}
, {-67, 76, 28}
, {-93, -31, -78}
, {-83, 40, -39}
, {79, 81, 38}
, {-53, -6, 24}
}
, {{69, -101, 44}
, {48, 67, 6}
, {32, 52, 40}
, {-59, 58, -107}
, {2, -50, 1}
, {-63, -15, -86}
, {-104, -3, -53}
, {-9, -26, 22}
, {-37, -18, 52}
, {33, -90, -40}
, {-91, -55, -103}
, {-64, -35, 16}
, {-78, 35, -45}
, {53, 67, -26}
, {57, -29, 42}
, {-110, -60, -19}
}
, {{-104, -69, -33}
, {14, -58, -53}
, {43, -55, 37}
, {91, 25, -70}
, {-29, -45, -80}
, {96, 36, 41}
, {67, 74, 11}
, {-3, -60, -54}
, {-9, 92, 105}
, {101, 19, 67}
, {-89, 86, -16}
, {-42, -36, -97}
, {35, 58, -3}
, {-53, -78, 103}
, {66, 100, 64}
, {97, -1, -16}
}
, {{44, -59, 2}
, {84, 48, 1}
, {82, 50, 90}
, {-75, -68, -56}
, {59, 101, 45}
, {-27, -52, 59}
, {73, 5, 18}
, {55, 13, 15}
, {68, 60, -75}
, {82, -20, 42}
, {-16, -70, -66}
, {-64, -51, 38}
, {36, 106, -54}
, {56, 67, 104}
, {-21, -86, 11}
, {-79, 4, 56}
}
, {{67, -99, -27}
, {47, 66, -11}
, {-30, 67, -26}
, {-12, -11, -50}
, {-49, 80, 104}
, {66, 55, 27}
, {52, 48, -3}
, {81, -87, 22}
, {-109, -9, 15}
, {-42, -102, -15}
, {-87, 0, -60}
, {94, 82, -90}
, {-62, 68, -92}
, {50, 52, -89}
, {87, -82, -62}
, {27, -106, 0}
}
, {{-64, -45, -97}
, {-21, -1, 85}
, {-111, -60, -99}
, {-105, -40, 83}
, {90, -63, -84}
, {96, 3, 76}
, {-69, 45, 11}
, {-74, -6, 56}
, {77, -85, -97}
, {49, 5, -31}
, {-86, -105, -28}
, {80, 66, -25}
, {-92, -54, 26}
, {-61, 48, -102}
, {-64, 4, 30}
, {31, -76, -44}
}
, {{-89, -74, -18}
, {63, -46, -107}
, {-71, 31, 14}
, {26, 30, 30}
, {-104, -49, -8}
, {45, -108, -75}
, {-99, 6, 58}
, {51, 73, 30}
, {-37, -1, -66}
, {-105, 72, 65}
, {64, -100, -108}
, {-81, 61, -49}
, {-21, -107, -46}
, {-56, -13, -7}
, {-43, -53, -70}
, {-47, -16, -15}
}
, {{-69, 64, -28}
, {-99, -24, -14}
, {50, 46, -88}
, {-66, 97, 70}
, {-95, 37, -42}
, {-10, -84, 22}
, {-71, -21, -40}
, {-68, -12, -66}
, {-61, -92, 69}
, {97, 80, 103}
, {-65, -55, -18}
, {-45, 102, 22}
, {100, 81, -70}
, {10, 24, 103}
, {-24, 3, 37}
, {-31, 40, -4}
}
, {{-64, 14, -85}
, {33, -78, -15}
, {-101, 45, 25}
, {-13, 33, -32}
, {-63, -93, -110}
, {39, -80, 38}
, {9, -96, 72}
, {-34, 18, -44}
, {-80, -112, 6}
, {16, -67, 35}
, {-104, 11, -97}
, {59, 39, 86}
, {-31, 10, -25}
, {-5, -2, -30}
, {41, -52, -82}
, {65, -12, 48}
}
, {{-66, 43, -62}
, {-95, 30, -106}
, {70, -13, -19}
, {30, 10, -34}
, {34, -22, -26}
, {-58, -47, -81}
, {-18, -24, 18}
, {77, 87, -36}
, {-64, 64, 39}
, {89, 23, 12}
, {-81, -96, -54}
, {-43, -31, -110}
, {29, 13, -99}
, {54, -1, 40}
, {37, -104, -88}
, {93, -102, -50}
}
, {{-46, 77, 78}
, {32, -100, 82}
, {83, -102, 50}
, {-92, -63, -12}
, {12, -37, 52}
, {14, 93, 58}
, {-21, 23, 37}
, {-97, 43, -5}
, {-99, -11, -56}
, {80, -59, 5}
, {4, 48, 99}
, {-97, -66, -77}
, {-5, 9, -16}
, {-50, -3, -26}
, {-33, -72, -91}
, {112, 6, 17}
}
, {{0, 73, -10}
, {69, 71, -12}
, {116, -31, 102}
, {58, -81, -91}
, {-99, -20, -54}
, {-78, -30, -73}
, {49, 107, 79}
, {-74, -35, -47}
, {-36, 2, 89}
, {60, -62, -21}
, {78, 57, 96}
, {29, 3, 42}
, {52, -52, 52}
, {-19, -7, 81}
, {77, -46, -40}
, {55, 52, 48}
}
, {{-60, 91, 31}
, {-66, -88, 35}
, {-99, 97, 97}
, {-92, 6, -73}
, {23, -40, -10}
, {8, -39, 34}
, {13, -65, -55}
, {-70, -51, 91}
, {26, -104, -11}
, {43, -21, 91}
, {-26, -48, -17}
, {-21, -24, 70}
, {51, -62, 102}
, {-90, -32, 58}
, {89, -56, 70}
, {-72, 77, -112}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   45
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_3_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_3(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    averagepool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   11
#define POOL_SIZE       8
#define POOL_STRIDE     8
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t average_pooling1d_output_type[INPUT_CHANNELS][POOL_LENGTH];

void average_pooling1d(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned short x;
  long_number_t avg, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
      tmp = 0;
      for (x = 0; x < POOL_SIZE; x++) {
        tmp += input[k][(pos_x*POOL_STRIDE)+x];
      }
#ifdef ACTIVATION_RELU
      if (tmp < 0) {
        tmp = 0;
      }
#endif
      avg = tmp / POOL_SIZE;
      output[k][pos_x] = clamp_to_number_t(avg);
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    flatten.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_DIM [1][32]
#define OUTPUT_DIM 32

//typedef number_t *flatten_output_type;
typedef number_t flatten_output_type[OUTPUT_DIM];

#define flatten //noop (IN, OUT)  OUT = (number_t*)IN

#undef INPUT_DIM
#undef OUTPUT_DIM

/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_SAMPLES 32
#define FC_UNITS 3
#define ACTIVATION_LINEAR

typedef number_t dense_output_type[FC_UNITS];

static inline void dense(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]) {			                // OUT

  unsigned short k, z; 
  long_number_t output_acc; 

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0; 
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ( kernel[k][z] * input[z] ); 

    output_acc = scale_number_t(output_acc);

    output_acc = output_acc + bias[k]; 


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = clamp_to_number_t(output_acc);
#elif defined(ACTIVATION_RELU)
    // ReLU
    if (output_acc < 0)
      output[k] = 0;
    else
      output[k] = clamp_to_number_t(output_acc);
#endif
  }
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_SAMPLES 32
#define FC_UNITS 3


const int16_t dense_bias[FC_UNITS] = {28, -42, 41}
;

const int16_t dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{18, 49, -179, -181, 126, 77, 99, -186, 121, 202, -121, 177, -198, 62, 184, -97, -170, 207, -156, -80, -176, -202, 191, 168, -142, -5, -94, -28, -204, 136, 178, -50}
, {-228, 181, -143, -161, -182, 177, -79, 23, -36, 63, -95, 18, -151, 149, -122, 166, 75, -180, 125, 18, 135, -208, 29, 106, -123, 148, -148, -3, 139, 208, -225, 140}
, {-36, 221, -193, 143, 111, 187, 71, 84, 35, -201, -57, 180, -89, -44, 159, 169, -195, 14, -25, 169, 31, 185, 89, -12, -173, 9, -49, -58, -63, 89, -95, -193}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    model.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    08 july 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef __MODEL_H__
#define __MODEL_H__

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define MODEL_OUTPUT_SAMPLES 3
#define MODEL_INPUT_SAMPLES 16000 // node 0 is InputLayer so use its output shape as input shape of the model
#define MODEL_INPUT_CHANNELS 1

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  //dense_output_type dense_output);
  number_t output[MODEL_OUTPUT_SAMPLES]);

#endif//__MODEL_H__
/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"

 // InputLayer is excluded
#include "max_pooling1d.c" // InputLayer is excluded
#include "conv1d.c"
#include "weights/conv1d.c" // InputLayer is excluded
#include "max_pooling1d_1.c" // InputLayer is excluded
#include "conv1d_1.c"
#include "weights/conv1d_1.c" // InputLayer is excluded
#include "max_pooling1d_2.c" // InputLayer is excluded
#include "conv1d_2.c"
#include "weights/conv1d_2.c" // InputLayer is excluded
#include "max_pooling1d_3.c" // InputLayer is excluded
#include "average_pooling1d.c" // InputLayer is excluded
#include "flatten.c" // InputLayer is excluded
#include "dense.c"
#include "weights/dense.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_output_type dense_output) {

  // Output array allocation
  static union {
    max_pooling1d_output_type max_pooling1d_output;
    max_pooling1d_1_output_type max_pooling1d_1_output;
    max_pooling1d_2_output_type max_pooling1d_2_output;
    max_pooling1d_3_output_type max_pooling1d_3_output;
  } activations1;

  static union {
    conv1d_output_type conv1d_output;
    conv1d_1_output_type conv1d_1_output;
    conv1d_2_output_type conv1d_2_output;
    average_pooling1d_output_type average_pooling1d_output;
    flatten_output_type flatten_output;
  } activations2;


  //static union {
//
//    static input_1_output_type input_1_output;
//
//    static max_pooling1d_output_type max_pooling1d_output;
//
//    static conv1d_output_type conv1d_output;
//
//    static max_pooling1d_1_output_type max_pooling1d_1_output;
//
//    static conv1d_1_output_type conv1d_1_output;
//
//    static max_pooling1d_2_output_type max_pooling1d_2_output;
//
//    static conv1d_2_output_type conv1d_2_output;
//
//    static max_pooling1d_3_output_type max_pooling1d_3_output;
//
//    static average_pooling1d_output_type average_pooling1d_output;
//
//    static flatten_output_type flatten_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  max_pooling1d(
     // First layer uses input passed as model parameter
    input,
    activations1.max_pooling1d_output
  );
 // InputLayer is excluded 
  conv1d(
    
    activations1.max_pooling1d_output,
    conv1d_kernel,
    conv1d_bias,
    activations2.conv1d_output
  );
 // InputLayer is excluded 
  max_pooling1d_1(
    
    activations2.conv1d_output,
    activations1.max_pooling1d_1_output
  );
 // InputLayer is excluded 
  conv1d_1(
    
    activations1.max_pooling1d_1_output,
    conv1d_1_kernel,
    conv1d_1_bias,
    activations2.conv1d_1_output
  );
 // InputLayer is excluded 
  max_pooling1d_2(
    
    activations2.conv1d_1_output,
    activations1.max_pooling1d_2_output
  );
 // InputLayer is excluded 
  conv1d_2(
    
    activations1.max_pooling1d_2_output,
    conv1d_2_kernel,
    conv1d_2_bias,
    activations2.conv1d_2_output
  );
 // InputLayer is excluded 
  max_pooling1d_3(
    
    activations2.conv1d_2_output,
    activations1.max_pooling1d_3_output
  );
 // InputLayer is excluded 
  average_pooling1d(
    
    activations1.max_pooling1d_3_output,
    activations2.average_pooling1d_output
  );
 // InputLayer is excluded 
  flatten(
    
    activations2.average_pooling1d_output,
    activations2.flatten_output
  );
 // InputLayer is excluded 
  dense(
    
    activations2.flatten_output,
    dense_kernel,
    dense_bias, // Last layer uses output passed as model parameter
    dense_output
  );

}
