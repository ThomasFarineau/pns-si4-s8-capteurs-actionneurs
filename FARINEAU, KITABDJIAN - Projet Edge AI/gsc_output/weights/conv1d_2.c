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


const int16_t conv1d_2_bias[CONV_FILTERS] = {32, -6, -12, -22, 37, 34, 43, -3, -12, 18, 2, -15, -15, 2, 42, 28, -4, 36, 33, 0, 29, 63, 32, -3, -27, 41, 35, 44, 32, -8, 36, -7}
;

const int16_t conv1d_2_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{33, 31, -56}
, {-42, -49, 60}
, {97, 71, 14}
, {75, -85, -7}
, {7, 46, -14}
, {1, -10, 3}
, {18, 56, 87}
, {63, -50, 69}
, {-19, 100, 91}
, {107, 64, 49}
, {100, 8, -84}
, {-10, 99, -43}
, {-46, 19, -68}
, {95, -75, -59}
, {96, 26, 102}
, {-4, -38, -120}
}
, {{-88, 33, -44}
, {-47, -40, 83}
, {-80, -37, -11}
, {-103, -33, 10}
, {-39, -50, 23}
, {-108, 17, -117}
, {-81, 10, 61}
, {-52, 68, 41}
, {-89, -15, -76}
, {-33, 80, -88}
, {86, -41, -49}
, {-103, -54, -69}
, {6, 13, 39}
, {-107, -47, 45}
, {-42, -58, 53}
, {-98, 48, 45}
}
, {{64, -94, 24}
, {12, -72, 77}
, {-57, 87, -105}
, {44, 26, 44}
, {8, -69, -62}
, {16, 45, -3}
, {-89, -3, -36}
, {-37, 0, -41}
, {43, -7, 42}
, {19, -68, -100}
, {-93, -4, -22}
, {-37, 75, -23}
, {-1, -98, -57}
, {0, -44, -94}
, {2, 90, -91}
, {-37, 73, -72}
}
, {{11, -14, -86}
, {-14, -32, 99}
, {-59, -2, -82}
, {37, -22, 8}
, {-64, 23, 91}
, {39, 22, 61}
, {8, 58, -68}
, {-9, -12, -62}
, {93, -52, 25}
, {-92, 61, 7}
, {-37, 2, -4}
, {98, 31, -19}
, {-101, 22, -31}
, {74, 97, -49}
, {71, -61, -7}
, {-17, 34, 84}
}
, {{-50, -52, 111}
, {-76, 105, -34}
, {62, -16, 66}
, {-32, -29, 16}
, {5, -64, -4}
, {20, 88, 41}
, {8, 22, -57}
, {70, 80, 17}
, {17, 94, 0}
, {-13, 84, 107}
, {30, 75, 82}
, {-13, 41, -57}
, {145, 130, 131}
, {18, -75, -88}
, {-25, 99, -33}
, {-26, 27, 1}
}
, {{55, 56, 44}
, {51, -5, -59}
, {111, -1, -7}
, {-46, 20, -62}
, {-73, -92, -74}
, {19, 52, -39}
, {78, -59, 9}
, {-25, 89, -94}
, {-29, 53, 80}
, {96, -65, 101}
, {-85, 84, 96}
, {-19, 3, 81}
, {147, 83, 107}
, {-52, -6, 85}
, {-27, 47, 2}
, {29, -96, 104}
}
, {{-62, 98, 93}
, {-26, -31, 0}
, {-25, -72, 107}
, {24, 37, 30}
, {-40, -64, -4}
, {-54, -20, 93}
, {89, -2, 0}
, {8, 70, 22}
, {109, 42, 35}
, {51, 10, -48}
, {74, 58, 104}
, {-18, 65, 113}
, {-32, 41, 13}
, {-95, -68, 26}
, {30, 97, 41}
, {50, 43, -5}
}
, {{-67, 79, -51}
, {63, 50, 6}
, {-48, -54, -98}
, {72, 45, -19}
, {-88, 5, 108}
, {13, 67, 110}
, {68, -47, 51}
, {50, 5, 19}
, {-10, -88, 46}
, {-54, 88, 17}
, {94, 18, -84}
, {-27, -11, 27}
, {-26, 37, -87}
, {-3, 54, -8}
, {11, 20, 105}
, {-2, 69, 76}
}
, {{3, -15, -103}
, {45, 72, 25}
, {88, -49, -2}
, {73, -67, 8}
, {2, -78, 98}
, {-58, 83, 78}
, {-88, -47, 13}
, {-22, 36, 74}
, {-5, -8, 6}
, {-53, -48, -43}
, {48, 39, -46}
, {73, -3, -98}
, {-106, -76, 47}
, {-7, -68, 65}
, {-92, -42, -5}
, {0, -49, 58}
}
, {{96, 84, 89}
, {-86, -82, 0}
, {-10, 71, -38}
, {95, 83, 53}
, {32, -34, 92}
, {5, 21, 11}
, {-39, 15, 55}
, {92, 28, 10}
, {-19, -34, 105}
, {-65, -27, 22}
, {5, 71, 80}
, {2, 85, -11}
, {67, 67, 88}
, {21, 98, 61}
, {-24, -45, -65}
, {-97, 12, 43}
}
, {{-76, 25, 84}
, {58, -11, 49}
, {50, 93, 25}
, {-63, 25, 78}
, {-54, -37, -31}
, {102, 64, 36}
, {-45, 84, 65}
, {-44, -88, -79}
, {40, -57, -72}
, {-5, -37, -13}
, {-46, 77, -28}
, {7, -62, -93}
, {77, -20, 75}
, {-29, -63, 36}
, {53, 29, -89}
, {-22, -63, -17}
}
, {{-93, -81, 48}
, {32, -26, 98}
, {-8, 93, 98}
, {117, 66, 68}
, {63, 79, -11}
, {98, 46, -94}
, {-38, -53, -17}
, {-59, -91, 90}
, {-104, 74, 63}
, {45, -11, -37}
, {59, 21, 95}
, {-8, -70, 21}
, {-57, -142, 2}
, {27, 18, -25}
, {14, 16, 17}
, {-2, 21, 23}
}
, {{-49, 84, 73}
, {-108, -94, -105}
, {-62, 61, -96}
, {98, -80, -24}
, {1, -108, -26}
, {0, 46, -112}
, {28, -75, -74}
, {68, 92, -29}
, {-3, -76, 17}
, {-1, 91, -86}
, {32, -3, 53}
, {-101, -63, -36}
, {76, 0, -75}
, {-45, 98, -79}
, {28, -99, 43}
, {47, -81, 21}
}
, {{24, -25, 74}
, {-9, 10, 19}
, {-65, 24, -46}
, {33, -74, -20}
, {104, -11, -38}
, {102, 116, 70}
, {-29, -63, -74}
, {32, -81, 89}
, {-90, -46, -8}
, {-122, -90, -24}
, {15, 10, -57}
, {80, -6, 34}
, {-20, -35, -106}
, {-52, 54, 86}
, {-64, -45, -60}
, {55, 88, -68}
}
, {{-8, 78, 82}
, {-1, -107, 31}
, {-7, 41, -67}
, {7, 42, 26}
, {49, 98, 82}
, {-23, 37, 48}
, {-90, -79, -31}
, {-7, 53, -34}
, {-63, 41, -86}
, {90, 103, -22}
, {94, 56, -41}
, {-14, 86, 40}
, {101, 49, 82}
, {-35, -47, 63}
, {116, 1, -50}
, {88, -118, -33}
}
, {{41, -20, 11}
, {55, -54, -9}
, {112, 65, 95}
, {-103, 12, 0}
, {47, -103, -23}
, {9, 115, -27}
, {5, 42, 22}
, {29, -44, 45}
, {-48, -67, -8}
, {53, 91, 43}
, {37, 42, 26}
, {80, 98, 77}
, {-44, 56, 97}
, {-50, -102, 74}
, {-49, -48, 39}
, {46, 71, -33}
}
, {{79, 52, -78}
, {-67, -107, 7}
, {-48, -57, 61}
, {77, -92, -13}
, {95, 2, 60}
, {-64, -28, -51}
, {36, 12, -12}
, {15, 9, 56}
, {-78, 29, -98}
, {-34, 29, 82}
, {19, -26, -103}
, {-88, 0, 14}
, {-3, 11, -26}
, {52, -82, 5}
, {19, -70, 88}
, {70, -35, 111}
}
, {{102, 98, -13}
, {66, 38, 84}
, {-55, 125, -28}
, {-36, -103, 33}
, {82, -23, 50}
, {124, -16, 112}
, {-76, 39, 30}
, {-67, 85, -4}
, {76, 72, -64}
, {-86, -58, 106}
, {-4, -22, -70}
, {5, 88, -32}
, {174, 66, 79}
, {-37, 38, 63}
, {155, 20, 147}
, {71, 2, -80}
}
, {{70, 4, -54}
, {5, -66, 26}
, {81, -6, -7}
, {-64, -64, -27}
, {78, -104, -58}
, {-39, 92, 20}
, {66, -6, 0}
, {4, 35, -67}
, {-9, 50, -70}
, {82, 38, 100}
, {116, 104, -16}
, {20, -7, -27}
, {43, 94, 122}
, {76, 53, 42}
, {-35, 2, 72}
, {33, 8, -44}
}
, {{-43, 74, -99}
, {-99, 43, 8}
, {-100, -16, -53}
, {-104, -80, -102}
, {-51, -86, -95}
, {-6, 1, -23}
, {82, 20, -81}
, {27, 39, -70}
, {-100, -38, -69}
, {9, -81, -25}
, {-68, -93, -27}
, {-102, 82, 46}
, {-43, -66, -7}
, {41, -43, 55}
, {27, 12, -104}
, {-88, 59, -35}
}
, {{55, 98, -57}
, {-11, -25, 102}
, {-76, 97, 42}
, {69, -70, -32}
, {34, -104, 97}
, {40, -50, -43}
, {-22, -54, 5}
, {52, 94, 22}
, {72, 5, 2}
, {-15, 52, 66}
, {-45, -99, 99}
, {51, 57, -48}
, {34, 105, 40}
, {38, 33, 18}
, {-46, 65, 48}
, {1, 5, 73}
}
, {{-17, -26, -15}
, {-65, -54, 69}
, {117, 130, 13}
, {-46, -34, -105}
, {-10, -66, -95}
, {68, 74, 98}
, {118, 79, -14}
, {118, 44, 122}
, {0, -62, -62}
, {-58, -53, -63}
, {65, 76, 59}
, {-4, 4, -13}
, {115, 131, 5}
, {-19, -83, 63}
, {75, 168, 50}
, {-104, -126, 45}
}
, {{-23, 40, -32}
, {28, -14, -60}
, {77, -13, 7}
, {71, 13, 25}
, {-53, -20, -40}
, {-6, -88, -8}
, {-100, -65, -92}
, {-58, -59, -39}
, {94, 96, 82}
, {43, 94, 77}
, {43, 49, -17}
, {87, 3, 90}
, {-60, -69, 116}
, {36, -33, 47}
, {-72, -81, -54}
, {19, -58, -36}
}
, {{65, -22, 33}
, {17, -34, 49}
, {4, 55, 56}
, {98, 59, 83}
, {23, 12, -2}
, {30, -31, 3}
, {-55, 51, 97}
, {-57, 48, -44}
, {-25, 11, -78}
, {-52, 75, 75}
, {-49, -84, -100}
, {-63, -32, -6}
, {-74, -76, 56}
, {48, -7, -46}
, {114, 44, 57}
, {17, 6, 97}
}
, {{-30, -124, 49}
, {64, 91, -96}
, {-15, -79, 86}
, {-76, -84, -99}
, {-101, 19, -72}
, {68, -31, 21}
, {56, -58, 38}
, {-1, 36, 63}
, {42, 13, -9}
, {80, -25, 68}
, {0, 31, -99}
, {-62, 36, -5}
, {31, 58, 8}
, {-56, 95, 13}
, {-100, -15, -68}
, {-57, 83, 68}
}
, {{62, 125, 11}
, {70, -58, -77}
, {97, 36, 121}
, {-45, -19, 47}
, {-107, 64, -129}
, {77, 90, 37}
, {-51, 25, -25}
, {59, 66, 27}
, {95, -30, -77}
, {38, 71, 81}
, {6, 37, -58}
, {133, 93, 135}
, {32, 103, 84}
, {-26, -4, 1}
, {92, 173, 13}
, {-31, 55, -13}
}
, {{0, -54, 30}
, {46, 84, 47}
, {-28, -54, 91}
, {61, -55, -88}
, {-3, -13, 0}
, {11, 118, 48}
, {-83, 7, 96}
, {0, 50, -13}
, {-3, -72, 75}
, {96, -95, -25}
, {81, 81, -84}
, {0, 60, -23}
, {167, 149, 5}
, {47, -98, -49}
, {9, 155, 142}
, {63, 7, -69}
}
, {{43, 51, 14}
, {-76, -45, -42}
, {111, 50, -20}
, {-21, -20, 1}
, {-5, 23, -64}
, {-34, 140, 140}
, {35, 41, -82}
, {82, 112, -32}
, {-21, -8, 83}
, {-53, -11, -26}
, {18, 124, 5}
, {33, 97, 115}
, {45, 124, 96}
, {-47, -37, 42}
, {137, 80, 123}
, {-61, 35, 9}
}
, {{-17, 93, -70}
, {29, -18, -95}
, {61, -71, 67}
, {-38, 9, -50}
, {-45, 71, 86}
, {72, -26, -18}
, {47, -70, 83}
, {-92, 57, 79}
, {40, 59, 73}
, {-101, 23, -10}
, {-71, 99, -37}
, {93, 23, 34}
, {137, 150, 52}
, {-75, -76, -63}
, {-72, -63, -28}
, {75, -97, 65}
}
, {{25, 46, -98}
, {82, 44, -42}
, {15, 2, 19}
, {0, -67, -70}
, {103, -4, 102}
, {29, 7, -27}
, {58, -29, -63}
, {8, 51, -85}
, {-1, 79, -87}
, {76, 60, -15}
, {-45, 44, 33}
, {-32, -58, 50}
, {20, 84, -107}
, {-35, -4, -68}
, {-91, -23, 64}
, {35, -80, 121}
}
, {{90, 84, 113}
, {21, -82, 72}
, {0, 11, 98}
, {15, 0, 26}
, {25, -90, -71}
, {24, 133, 88}
, {48, -87, 54}
, {86, -47, -17}
, {100, 45, 13}
, {34, 64, -40}
, {15, 115, 123}
, {-29, -1, 131}
, {61, 148, 75}
, {34, -78, -54}
, {136, 65, 87}
, {-53, -122, 38}
}
, {{-115, 38, 44}
, {90, 15, 80}
, {30, -96, 1}
, {-82, 108, 7}
, {66, -81, 66}
, {-55, 111, 31}
, {-80, -23, 10}
, {10, -97, 73}
, {-105, 58, -18}
, {53, -75, 24}
, {78, 80, -27}
, {64, -101, 96}
, {55, -69, -103}
, {-77, -71, 9}
, {-65, 99, 4}
, {91, -8, -64}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE