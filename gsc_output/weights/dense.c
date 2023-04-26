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