/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    8
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_1_bias[CONV_FILTERS] = {56, -13, 55, 0, -15, 60, 29, 54, 47, 47, 57, 62, 84, -24, 73, 25}
;

const int16_t conv1d_1_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{18, -118, -44}
, {-156, -21, -82}
, {-6, 122, -151}
, {91, -9, -61}
, {77, -28, -36}
, {-140, -8, 43}
, {-3, -9, 143}
, {-109, 88, 55}
}
, {{106, 149, 140}
, {-50, -139, 57}
, {-103, 124, 136}
, {94, 31, 105}
, {-6, 52, 9}
, {8, 81, -57}
, {-132, -14, -115}
, {-122, 97, 58}
}
, {{77, 38, -100}
, {-76, -178, 19}
, {-135, 79, -113}
, {0, 65, 144}
, {-132, 123, 73}
, {134, -111, -102}
, {18, 158, -88}
, {-176, 96, -169}
}
, {{-71, -41, 48}
, {-49, 120, -34}
, {-44, -15, 19}
, {63, -51, 71}
, {-43, 112, -39}
, {-91, -95, 88}
, {-91, 77, -73}
, {-10, 153, -69}
}
, {{133, -103, 149}
, {-19, 104, 59}
, {2, 117, 104}
, {21, 36, 139}
, {-31, -37, 85}
, {-93, -108, -135}
, {3, -131, 77}
, {59, 57, 53}
}
, {{-23, -90, 59}
, {36, -107, 87}
, {-154, -4, -4}
, {-53, 24, 134}
, {-100, -5, 84}
, {55, 80, -67}
, {-80, 164, -98}
, {65, 41, -164}
}
, {{30, -23, -2}
, {100, -154, 47}
, {29, 118, 102}
, {-125, -57, 34}
, {-99, 108, -87}
, {-15, -98, -24}
, {-25, 131, -66}
, {-101, 4, -68}
}
, {{-86, -138, -160}
, {-142, 50, -11}
, {50, -94, -62}
, {-33, -13, 101}
, {-54, 107, 42}
, {-37, 37, 6}
, {-85, -3, 131}
, {-147, 95, 103}
}
, {{28, -39, 93}
, {-146, -35, -120}
, {73, -129, 17}
, {-126, 123, 77}
, {67, 37, 128}
, {46, -149, 93}
, {96, 156, 120}
, {-71, 87, -38}
}
, {{-139, -34, -85}
, {61, 81, -3}
, {31, 9, 77}
, {1, -9, 68}
, {-98, 23, 109}
, {-105, 106, 10}
, {-3, -48, 36}
, {-112, 0, -104}
}
, {{-54, -19, 77}
, {-165, 29, -21}
, {61, 33, -14}
, {106, -11, -43}
, {123, 87, 11}
, {-116, -56, -58}
, {-28, 43, -27}
, {66, -125, -126}
}
, {{-179, -6, -3}
, {-43, 35, -86}
, {72, -74, -103}
, {96, -111, 61}
, {64, 27, -82}
, {-59, -89, 22}
, {90, 110, -73}
, {-67, -75, 75}
}
, {{-16, -1, 85}
, {-142, -30, 102}
, {-71, -96, -143}
, {78, 147, -69}
, {57, 3, -120}
, {4, 20, -31}
, {33, 50, 149}
, {-157, -155, -80}
}
, {{-116, -11, -8}
, {12, 80, 131}
, {43, 135, 50}
, {-45, -167, -99}
, {-10, 79, -52}
, {-22, -141, 7}
, {91, 139, 97}
, {-23, 124, 25}
}
, {{-176, -83, 38}
, {18, -78, 83}
, {68, -163, -34}
, {46, -101, 44}
, {-57, 8, 6}
, {-14, -46, -35}
, {73, 27, 88}
, {8, -121, 49}
}
, {{-21, -129, 120}
, {64, -143, -20}
, {-79, -133, 33}
, {142, 93, 145}
, {-121, -72, -32}
, {-41, -156, -117}
, {-152, -100, -76}
, {82, 133, -58}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE