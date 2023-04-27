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


const int16_t dense_bias[FC_UNITS] = {35, -34, 21}
;

const int16_t dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{-110, -127, -203, 40, 74, 118, 21, 34, 37, -190, 191, -122, 77, -75, 76, -16, 43, 72, 25, 4, 136, 115, 3, -201, 121, 228, 114, 76, 206, -162, 162, 68}
, {-154, 88, 201, 106, -116, -111, -36, 130, 49, -176, 147, 212, 54, 226, -27, -101, 148, -186, -109, -54, -54, -179, -41, 74, 164, -182, -228, -202, 75, 127, -220, 198}
, {-30, -116, 15, -114, 101, -150, 41, 130, -135, 95, -16, -101, 11, -167, 144, -80, -152, 73, 145, 32, -129, 40, 105, -67, 97, 210, 175, 74, 68, -204, 191, 33}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS