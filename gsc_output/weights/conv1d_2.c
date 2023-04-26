/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
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