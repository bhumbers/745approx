#include <stdio.h>
#include <stdlib.h>

#include <assert.h>

////////////////////////////////////////////////////////////////////////////////////////
//Right now, you're thinking to yourself... dead god, is that an 8-bit 2D image embedded in a C file?
//The answer is yes.... yes it is.
static const unsigned int img[] = {
33,82,53,40,33,21,57,49,27,86,54,94,53,45,60,6,12,57,70,44,44,57,38,11,28,49,57,24,61,40,31,37,21,9,56,43,29,85,86,39,50,43,47,61,33,38,12,76,18,28,62,19,62,46,85,47,12,63,44,6,38,98,18,38,8,30,39,47,9,30,22,63,31,48,27,41,44,85,27,44,42,54,45,20,26,42,0,67,44,19,35,59,23,45,17,59,63,58,42,86,
21,15,52,73,16,35,45,64,23,63,44,26,48,66,46,28,25,55,10,77,21,43,19,26,67,74,52,57,4,53,48,59,61,21,97,27,47,32,33,29,21,29,19,86,17,33,0,19,63,24,75,29,74,45,40,40,44,22,81,25,47,50,24,56,61,40,62,18,21,16,66,44,35,5,17,47,53,60,52,49,72,38,61,27,61,47,63,17,50,14,11,53,26,18,54,20,77,53,28,53,
37,35,71,38,44,32,52,50,9,0,69,51,42,38,29,85,35,18,50,39,24,56,33,33,42,60,15,24,37,41,60,43,17,14,62,18,71,59,46,46,41,25,24,6,29,59,30,9,27,15,38,45,23,34,48,30,17,14,41,24,11,29,46,67,30,62,22,60,30,62,11,44,43,36,26,41,64,53,32,18,42,55,7,61,45,36,26,55,35,59,63,9,2,0,19,34,27,86,57,39,
41,34,50,61,33,71,49,23,74,32,48,13,32,76,63,49,20,39,47,18,48,53,1,26,25,38,49,36,42,30,80,61,91,17,43,41,58,57,62,7,39,39,2,33,17,54,20,30,36,13,37,3,42,38,31,30,79,56,28,45,29,73,48,46,34,21,0,46,39,53,50,41,40,46,46,36,33,53,58,70,61,15,36,46,33,34,31,50,60,29,20,54,38,36,37,61,42,58,10,67,
31,66,52,7,48,47,25,62,51,30,27,32,0,13,45,63,3,36,52,28,43,0,25,43,49,31,47,34,56,37,33,22,45,43,43,27,63,31,30,9,26,39,34,23,27,65,54,34,16,34,6,0,43,17,55,31,25,3,69,21,61,11,14,16,8,66,30,13,24,30,2,27,17,34,15,25,22,0,40,49,27,11,45,34,24,34,51,41,11,27,32,43,22,10,29,69,42,24,89,65,
54,52,64,29,51,7,22,16,34,46,31,27,33,36,65,29,68,72,22,43,27,49,54,64,42,30,15,22,13,25,1,46,8,0,30,32,49,42,10,38,53,28,26,9,21,23,13,42,19,29,46,32,26,46,0,52,16,60,37,52,78,70,71,15,45,67,52,24,0,21,40,11,26,4,61,26,0,41,46,19,30,50,36,58,12,29,35,47,46,18,0,42,34,42,23,24,49,27,73,5,
48,33,53,38,30,29,22,49,26,38,19,19,41,4,51,15,38,55,20,43,34,47,73,89,17,3,28,29,93,42,63,35,14,63,50,8,41,42,28,29,46,39,20,12,25,41,0,44,20,19,48,155,255,195,234,255,234,144,130,52,50,51,52,37,39,10,2,32,54,22,46,36,0,54,30,24,21,18,69,38,34,28,39,34,25,22,34,41,41,23,57,52,70,20,17,54,48,45,53,27,
77,58,60,34,24,16,48,33,51,53,23,98,43,89,36,48,61,16,68,60,30,58,43,41,36,39,46,59,20,26,17,34,59,39,56,85,41,60,62,37,7,18,39,36,22,37,34,38,42,91,179,136,255,198,255,206,87,125,255,211,255,112,55,36,20,28,33,28,48,17,13,53,54,37,49,63,39,47,16,54,49,29,45,6,17,22,57,23,2,33,21,58,28,47,28,7,73,13,55,15,
64,7,48,59,12,55,54,77,37,10,72,51,38,31,76,22,55,69,0,43,37,11,39,33,48,34,59,41,51,29,50,9,71,42,46,41,22,31,50,17,34,32,22,9,20,15,39,39,133,121,255,160,176,207,255,153,255,138,138,157,145,145,212,96,77,29,0,19,44,24,27,36,23,36,57,32,36,22,25,16,42,68,49,15,60,12,17,0,38,38,29,54,31,26,19,55,40,38,58,34,
37,53,40,55,72,42,13,23,38,10,42,13,33,30,89,48,61,63,0,27,10,48,35,44,49,0,3,29,15,71,32,27,37,73,62,73,65,28,13,28,39,26,5,28,14,27,161,255,255,255,142,255,255,255,226,255,92,13,133,179,83,25,71,36,166,39,27,37,26,21,47,16,17,14,9,36,52,35,47,12,25,8,15,39,43,14,0,15,29,33,17,34,29,18,74,34,0,30,29,73,
57,59,43,18,13,45,14,47,20,58,58,45,48,84,49,42,23,0,43,49,36,16,13,20,62,44,44,49,33,47,55,56,29,36,28,28,21,19,41,34,48,17,10,16,27,97,176,108,255,251,145,255,254,255,30,127,114,41,149,184,64,183,140,84,88,74,49,17,30,22,32,30,12,62,20,31,41,19,51,24,28,12,48,23,0,18,46,44,60,62,47,50,60,8,66,15,28,36,73,32,
17,36,26,64,63,77,59,95,41,54,32,36,29,26,27,0,79,63,39,39,16,39,24,44,11,21,21,44,52,28,45,35,41,60,46,63,48,29,31,17,31,36,28,32,167,246,255,184,130,116,85,252,150,152,255,83,255,208,101,96,7,124,136,56,21,56,41,37,19,19,34,26,29,55,34,29,18,35,34,1,18,30,21,31,0,29,20,37,29,51,53,58,32,35,51,24,17,57,56,53,
66,0,31,54,63,69,50,29,57,71,55,64,23,23,46,47,60,40,46,22,40,30,39,44,34,37,47,38,78,13,41,0,32,38,62,38,43,8,67,44,30,30,53,153,129,255,255,124,177,255,216,68,255,161,52,42,230,48,155,133,161,42,49,154,91,59,27,38,24,31,47,44,54,53,64,25,43,7,4,14,2,0,29,12,20,39,41,21,58,16,33,11,40,8,40,47,26,61,12,28,
38,24,44,49,25,73,49,8,57,59,83,29,0,26,42,59,50,82,4,55,72,27,37,51,19,26,28,11,25,18,76,20,19,70,25,46,8,30,30,49,10,45,197,253,255,255,235,170,206,184,255,159,240,255,251,0,255,126,102,12,108,79,36,88,114,46,43,17,46,28,49,25,24,11,58,0,4,41,11,33,23,42,40,65,62,39,33,38,25,59,23,34,39,38,45,50,45,0,19,42,
21,51,64,0,55,28,66,44,44,16,67,52,81,37,51,46,99,73,35,26,16,48,58,35,11,16,26,12,44,43,12,44,31,8,28,37,42,64,44,37,44,54,189,102,132,255,219,207,201,207,255,255,230,0,98,7,121,114,241,122,86,78,105,155,104,24,68,43,25,34,22,14,67,21,42,0,29,23,14,18,25,35,14,14,31,50,24,24,18,1,48,17,36,27,16,50,0,39,32,13,
46,63,57,63,48,24,57,29,34,58,82,80,33,46,22,119,59,75,32,68,49,28,34,9,6,28,40,17,27,46,83,29,40,24,40,6,44,29,43,8,45,68,89,202,255,239,176,205,255,182,255,255,207,255,56,100,148,253,159,93,66,41,84,195,168,26,124,32,26,42,16,24,20,14,0,51,40,39,20,23,8,27,21,17,8,8,63,29,52,58,21,65,24,12,54,27,51,32,25,7,
60,38,56,46,71,0,40,45,31,36,77,26,61,0,22,62,74,88,48,30,48,34,35,63,48,68,45,9,41,48,38,44,50,48,37,20,37,32,47,37,70,37,241,255,255,243,255,117,114,199,57,235,255,255,152,223,0,43,142,97,37,57,48,88,133,27,62,32,47,33,24,26,52,50,72,20,64,36,34,28,25,35,36,29,42,28,24,48,30,20,27,38,47,28,38,18,43,34,51,50,
35,50,67,66,23,36,61,51,48,33,39,46,62,54,38,67,35,0,27,25,14,63,35,24,51,35,36,45,19,70,70,62,51,54,0,24,22,41,30,36,46,31,185,227,225,183,156,119,255,134,255,158,166,166,174,195,131,33,0,83,86,59,21,46,134,81,40,29,41,26,12,24,20,17,32,7,29,18,32,24,31,16,18,34,28,29,32,50,6,47,40,25,46,28,13,26,30,28,39,32,
47,20,69,29,7,20,58,59,50,32,55,36,38,82,70,57,34,50,27,39,39,33,18,40,31,55,51,35,47,59,21,55,20,19,53,46,28,21,25,31,24,4,0,94,255,105,255,230,39,227,236,164,63,178,207,52,47,57,13,17,22,118,17,0,123,54,78,19,29,22,23,46,9,19,37,30,39,54,38,17,6,43,41,19,12,28,30,42,47,18,25,23,46,39,44,35,37,38,40,62,
70,1,103,48,9,68,51,32,101,55,42,52,44,8,93,75,41,0,57,72,32,28,57,31,58,31,39,19,50,29,27,52,54,25,47,29,21,21,8,41,27,12,63,199,255,255,215,220,56,63,18,26,30,175,101,50,33,83,33,92,52,39,87,50,67,85,34,49,31,28,35,30,10,52,40,44,58,13,45,18,32,36,44,33,1,29,33,24,17,79,0,26,26,43,55,42,45,39,31,21,
25,83,38,45,51,35,29,55,31,59,41,87,12,81,22,65,54,58,48,58,68,15,24,33,35,35,32,22,55,67,81,84,24,39,62,34,55,20,24,75,18,9,69,255,225,51,59,255,70,133,128,54,66,33,215,44,68,56,70,85,165,96,38,27,57,96,27,37,34,43,48,20,36,40,15,38,38,56,0,44,51,55,0,24,38,28,50,36,42,0,65,43,45,54,47,2,65,23,48,51,
77,53,41,71,32,33,60,31,59,42,0,42,40,57,48,77,87,7,39,31,49,23,0,15,28,17,20,56,88,20,21,59,3,46,28,16,52,36,2,23,4,47,68,89,255,255,99,208,71,116,83,63,110,195,64,55,93,208,186,76,224,12,131,30,73,35,49,0,19,23,27,72,43,45,69,48,30,12,41,38,32,58,31,18,22,34,11,75,24,51,25,34,33,23,54,32,18,64,28,31,
35,41,50,21,56,69,13,55,44,16,86,75,81,40,71,47,97,59,70,6,41,20,41,43,21,32,30,12,70,64,9,40,14,52,35,10,17,63,22,31,39,15,63,231,255,231,115,109,255,255,111,107,175,92,139,91,89,161,34,52,255,115,54,4,53,22,54,35,40,51,31,52,43,47,43,13,82,25,27,54,42,12,34,32,39,40,23,30,35,21,37,38,4,31,58,47,9,41,47,44,
17,45,30,50,19,14,41,53,58,101,32,0,22,75,35,6,55,63,73,21,39,26,18,31,42,14,8,21,45,9,12,0,47,68,18,8,18,32,56,12,48,42,17,96,181,229,105,102,255,140,17,255,85,255,165,60,85,197,191,255,0,170,127,55,37,38,41,20,56,44,37,50,56,46,49,0,37,40,29,42,18,41,31,23,0,36,25,22,7,61,33,12,19,18,22,39,51,32,42,30,
25,41,50,47,48,77,44,78,44,10,64,39,33,72,88,0,104,60,73,61,54,12,48,68,11,50,65,44,32,36,52,35,75,14,51,40,33,15,44,30,22,9,0,135,115,146,225,255,172,169,68,255,233,255,255,47,81,122,156,255,66,121,81,74,78,65,42,28,16,45,43,65,43,36,63,16,25,0,33,27,35,20,31,16,18,28,37,0,34,69,44,22,50,53,40,45,68,43,25,20,
72,42,37,26,59,30,48,41,77,57,56,47,56,28,8,96,86,96,20,22,34,29,30,11,55,29,48,47,74,4,43,41,38,79,88,72,39,26,31,33,0,7,12,0,160,212,71,223,178,255,139,255,242,198,78,68,66,48,148,175,176,192,58,87,113,23,38,40,23,33,17,37,77,36,75,74,22,31,19,36,17,19,32,37,29,31,17,56,30,33,64,32,0,2,15,36,32,49,31,34,
103,14,32,110,39,45,45,22,58,62,3,69,46,31,48,18,29,63,62,60,58,0,69,44,20,20,29,38,35,66,20,18,0,77,67,32,58,15,39,36,53,33,27,14,114,105,94,182,229,255,255,140,198,216,193,72,41,40,28,197,66,141,54,123,25,75,37,31,43,36,34,24,67,55,54,37,46,35,22,52,17,13,38,52,53,42,18,40,7,0,41,36,44,31,43,82,73,14,15,4,
33,38,42,68,59,20,60,20,60,70,44,49,27,52,18,99,107,66,9,84,49,26,62,39,36,34,46,47,72,83,33,43,41,40,63,50,67,48,15,16,2,7,39,20,22,255,127,255,125,255,88,44,117,0,104,57,43,42,33,255,162,69,111,94,137,56,25,51,24,14,60,31,41,46,51,43,45,40,34,38,27,19,41,37,28,42,55,54,65,64,52,0,34,36,55,72,29,23,69,50,
53,25,57,15,61,70,24,66,32,64,59,51,50,8,40,0,47,16,61,36,12,50,25,38,43,7,60,52,106,78,51,62,73,18,44,48,41,79,42,45,43,17,12,18,7,83,8,255,255,199,85,255,191,236,186,255,2,61,100,130,255,151,95,42,85,108,30,28,23,39,70,52,55,22,73,21,42,52,63,32,25,32,21,19,18,16,38,8,35,36,19,60,37,33,25,21,19,52,35,22,
34,39,36,63,71,41,73,81,0,55,35,51,37,22,77,49,118,30,48,67,19,31,32,52,51,44,18,0,32,90,94,2,37,19,29,47,18,32,48,23,53,12,21,11,23,17,255,199,22,0,195,254,255,255,176,169,99,69,108,55,90,56,26,12,96,85,23,30,47,70,33,82,39,40,35,68,54,14,17,59,40,53,28,15,43,23,39,17,8,43,32,59,26,61,66,33,80,56,63,11,
23,39,32,106,4,49,57,0,30,53,46,86,77,55,44,103,87,87,61,78,10,23,24,57,49,22,27,21,70,64,107,36,73,56,56,85,61,32,31,0,39,31,19,37,24,8,67,182,220,154,212,201,196,130,193,96,68,42,40,114,162,84,124,12,81,20,12,18,14,19,34,25,48,25,35,56,45,59,44,14,42,43,49,22,44,17,22,20,30,45,56,56,49,64,24,76,29,69,53,67,
29,39,45,34,38,65,14,25,44,60,84,49,56,75,68,101,61,104,107,19,41,24,33,11,21,40,2,43,62,22,82,61,46,43,49,73,10,0,40,16,32,22,11,17,14,15,11,166,255,204,159,14,255,255,255,194,147,54,75,156,134,47,100,24,113,30,20,23,7,15,23,31,28,9,4,85,98,44,79,37,64,40,22,21,4,31,37,11,4,41,16,57,7,11,32,15,62,45,10,72,
56,103,65,53,48,47,75,60,103,54,0,0,55,32,56,45,57,86,53,89,50,28,0,57,41,49,23,34,47,96,48,24,30,67,28,40,40,70,40,58,37,10,10,18,2,22,7,98,255,252,163,255,98,255,186,149,103,128,80,68,0,0,158,105,168,16,37,23,21,18,32,32,25,4,36,97,95,23,62,8,31,37,13,17,20,59,39,4,23,11,66,18,16,37,43,56,6,47,29,61,
10,65,58,69,44,71,0,58,44,8,49,42,72,70,29,70,79,37,0,62,50,56,57,28,30,6,61,46,90,48,81,39,68,49,64,45,67,90,38,0,45,37,21,18,25,26,18,7,48,255,255,248,210,255,231,75,24,142,180,187,255,78,130,225,51,19,36,25,24,19,19,10,15,24,47,56,39,39,73,46,27,27,59,4,32,27,28,21,54,38,33,16,37,10,44,29,25,4,64,20,
78,5,45,61,56,45,85,85,47,63,54,6,46,44,60,151,110,68,51,86,63,16,13,24,58,45,67,41,73,0,40,69,35,72,18,58,17,59,38,38,39,27,23,4,19,20,12,29,52,210,233,255,133,207,0,130,69,255,242,71,65,57,74,48,17,18,13,2,25,10,30,19,33,26,27,65,61,80,72,5,52,62,29,13,42,39,54,29,41,44,34,75,32,15,43,26,41,27,67,80,
71,57,73,31,55,54,67,6,37,22,46,26,71,37,91,29,116,64,19,54,42,18,22,40,10,63,57,77,26,77,21,26,44,0,56,80,42,34,51,39,58,41,5,13,20,11,22,27,33,114,255,255,135,222,255,203,159,159,114,117,106,73,150,98,26,11,29,15,27,31,18,26,39,10,35,52,62,53,53,13,108,45,62,41,31,21,32,73,61,35,67,20,13,12,45,31,34,46,45,106,
43,57,55,59,66,73,26,54,17,71,64,71,88,80,31,100,69,5,82,65,32,29,20,24,26,10,20,32,25,65,56,8,71,32,42,40,6,61,59,40,44,10,13,10,23,13,4,67,88,92,255,14,113,78,255,197,242,31,110,100,54,0,52,8,11,29,38,22,10,24,16,16,76,36,17,74,29,77,43,0,62,40,13,24,40,40,42,37,27,44,53,12,3,48,16,58,30,68,81,0,
70,79,38,81,51,21,102,38,103,70,97,26,61,52,118,41,98,42,70,31,48,55,45,65,48,42,43,6,33,23,15,49,45,49,18,47,19,47,73,52,79,31,6,8,35,33,43,66,90,191,237,230,59,158,36,146,47,21,62,67,63,79,13,24,34,14,5,10,20,9,50,45,29,32,39,27,75,107,13,49,0,47,0,16,45,45,51,40,46,19,39,46,13,28,45,57,29,118,81,64,
77,65,14,80,67,74,30,82,38,38,30,94,45,157,80,111,75,56,61,65,33,42,42,22,29,78,57,59,28,41,85,51,64,51,29,57,37,53,11,60,57,52,30,31,41,78,105,124,57,29,214,111,0,176,46,140,255,110,75,69,59,147,0,23,9,33,28,22,18,38,43,52,96,105,101,14,108,81,68,25,99,111,134,123,37,43,38,46,14,59,0,63,51,0,41,30,32,101,36,63,
69,56,79,62,44,101,51,48,23,92,32,30,72,107,82,4,47,84,7,56,0,39,88,17,65,61,33,30,18,93,73,8,31,48,27,74,32,64,63,57,41,10,25,29,132,133,78,173,96,208,255,255,172,255,255,245,126,106,34,43,198,67,20,34,32,39,15,50,6,92,63,61,44,110,88,16,99,19,50,90,77,111,133,102,50,29,71,32,52,47,28,3,22,12,33,27,41,62,22,48,
37,49,59,16,77,64,58,57,42,43,66,5,95,61,21,28,86,51,72,0,51,32,42,77,53,92,52,20,51,67,45,35,65,34,0,62,123,58,47,31,5,58,126,0,71,143,21,125,84,79,255,187,255,247,178,255,102,74,38,80,197,4,50,39,34,36,16,60,8,39,6,25,75,55,67,43,157,30,168,80,64,54,206,78,3,51,46,44,37,59,32,53,47,15,33,22,27,31,7,12,
51,66,62,53,31,54,76,36,89,68,67,27,82,100,0,116,13,58,67,49,75,60,51,31,45,42,1,78,57,96,82,32,24,73,49,4,77,64,31,49,50,19,139,121,76,113,132,97,0,69,70,206,49,255,52,139,185,48,52,38,32,33,67,44,18,32,56,49,61,68,1,125,173,127,54,98,124,129,126,75,53,31,63,94,41,45,87,22,34,56,39,8,28,49,18,16,28,15,69,82,
46,13,153,26,105,93,43,82,74,42,33,61,2,122,22,136,50,71,1,44,35,29,54,66,91,42,50,44,57,72,0,56,39,10,53,41,59,127,134,54,37,33,67,211,61,66,153,131,34,143,20,159,72,152,194,255,215,138,63,6,42,48,126,11,21,66,44,62,47,88,49,115,80,19,61,0,59,63,100,74,73,160,73,151,83,28,26,19,45,144,251,48,50,54,64,44,31,21,15,58,
41,18,51,9,46,83,23,61,40,55,82,36,76,140,123,77,106,20,41,40,80,52,30,85,94,64,59,27,51,13,43,52,20,39,17,47,72,165,119,182,119,66,168,74,81,118,63,158,93,140,36,89,99,56,66,0,89,84,64,48,102,112,216,47,27,96,43,131,28,80,128,80,123,0,157,147,27,26,105,64,80,123,123,69,9,19,57,15,42,25,242,99,135,35,28,32,16,18,53,53,
37,60,106,94,109,16,31,13,55,114,103,132,26,95,97,105,47,66,78,63,59,0,29,104,50,36,35,60,51,69,50,61,60,31,104,92,75,61,191,170,121,8,152,208,52,42,16,68,23,121,65,56,43,53,26,36,77,58,53,37,98,92,137,13,19,75,89,36,48,5,43,128,93,36,170,66,53,14,62,16,85,100,58,64,28,38,34,13,30,61,255,153,62,131,64,54,55,41,37,29,
65,83,100,16,25,61,58,78,94,86,52,37,79,111,125,49,59,57,54,46,101,7,57,53,53,83,42,83,67,65,46,1,33,44,107,117,53,198,63,175,153,193,140,109,75,124,184,133,69,31,19,17,41,34,18,94,137,53,73,134,27,143,141,15,30,53,45,135,142,99,75,65,126,68,64,100,78,24,33,40,110,150,40,52,27,35,62,50,27,63,27,148,61,165,42,101,58,81,57,83,
50,28,48,41,125,54,13,64,84,66,65,0,45,74,69,55,25,89,55,40,31,115,53,77,45,38,49,73,49,60,52,51,87,60,41,62,146,26,142,120,130,41,126,84,171,87,124,155,43,158,41,102,48,99,145,142,27,210,153,68,152,144,143,23,30,84,112,46,147,74,126,223,88,220,168,60,23,15,13,71,164,36,31,14,74,59,28,33,20,18,108,85,142,11,109,58,57,31,67,21,
31,0,17,95,59,18,96,114,0,28,42,137,92,82,0,52,70,55,76,51,59,67,61,50,30,70,7,42,19,69,73,97,35,53,120,122,191,73,206,158,118,45,80,34,117,122,170,81,16,79,48,82,119,255,16,0,254,255,32,147,178,135,89,21,28,90,49,155,0,56,64,218,120,112,29,51,2,14,27,98,22,29,0,72,37,49,39,33,31,40,123,224,198,88,135,144,0,60,119,92,
55,123,92,65,99,7,91,97,48,78,76,79,138,52,83,76,94,69,41,35,12,57,74,31,64,10,36,55,36,25,15,80,57,100,68,222,199,105,119,101,71,113,0,123,165,159,125,74,8,44,66,135,219,255,129,255,231,237,104,152,70,83,103,37,21,41,82,135,89,116,52,232,82,73,183,55,42,17,33,51,15,98,131,136,21,58,76,0,88,24,71,187,126,125,60,87,112,141,111,159,
74,0,58,11,55,67,86,11,66,82,44,30,105,122,28,98,19,100,39,102,73,35,23,43,17,61,113,32,0,60,54,0,96,39,83,138,180,88,169,111,105,188,0,131,26,98,113,24,32,158,0,146,141,255,241,249,153,91,90,130,98,159,94,27,25,58,40,37,191,187,80,36,255,25,83,29,12,24,52,45,103,110,43,73,1,38,0,28,14,14,32,90,104,102,163,48,88,123,22,130,
85,102,47,40,39,47,84,61,67,51,32,41,57,110,58,86,69,36,48,28,59,48,52,39,40,48,31,42,49,62,40,14,63,18,87,36,33,203,136,90,114,160,75,144,97,196,209,34,102,99,214,62,56,115,255,109,218,134,184,124,96,135,89,17,14,35,65,53,94,70,128,176,159,110,53,76,28,15,47,63,15,114,16,11,99,40,60,55,64,51,23,121,143,83,59,57,2,81,19,73,
87,51,73,36,49,124,37,71,53,28,60,46,74,22,35,52,36,23,62,77,102,76,101,39,46,48,93,36,21,37,45,29,38,72,84,16,101,108,153,205,67,125,52,93,130,129,61,30,73,120,131,92,174,255,51,117,50,0,144,128,181,140,39,30,23,15,120,44,55,27,118,132,23,83,125,14,37,26,40,28,49,20,116,83,6,34,24,80,79,95,48,0,84,49,88,25,45,48,65,103,
67,114,99,72,63,89,30,47,0,59,71,60,32,27,67,0,66,0,81,21,29,35,60,2,48,71,67,25,8,26,20,7,68,37,116,53,170,128,94,0,72,91,89,114,226,166,34,31,92,25,101,135,97,141,223,105,131,50,151,85,204,183,132,19,17,32,66,81,136,96,193,150,189,0,65,3,8,18,49,105,20,26,101,129,56,53,8,123,66,40,12,38,62,102,55,101,78,27,68,30,
67,65,81,73,79,48,35,30,55,36,81,61,68,72,22,5,63,32,43,43,24,18,23,37,23,35,83,60,25,51,35,72,1,72,90,0,138,76,44,44,111,83,94,62,90,186,7,47,110,158,116,99,255,255,181,77,124,149,98,176,181,102,163,34,9,1,46,121,90,61,128,45,70,160,73,47,32,18,63,71,128,110,117,8,115,29,97,109,81,74,7,24,101,26,83,117,19,94,20,64,
18,83,97,59,34,59,50,95,47,30,57,49,67,54,87,31,5,39,48,86,46,62,42,12,0,71,0,34,31,14,58,40,83,48,173,145,65,211,169,78,91,80,104,166,162,82,22,99,41,64,169,113,209,240,114,29,89,94,169,36,102,87,229,38,16,30,11,0,242,88,79,83,151,79,85,45,38,27,10,87,133,34,119,33,0,152,218,32,82,60,88,29,87,32,27,78,75,38,25,67,
41,59,80,3,98,66,107,85,81,13,66,78,72,59,54,26,27,47,46,14,61,49,75,74,32,47,29,33,9,55,68,48,110,139,126,141,158,107,60,132,33,94,127,255,214,120,30,150,84,23,159,75,255,209,156,48,28,93,236,78,54,185,255,40,5,16,16,98,94,161,109,73,54,121,44,34,18,18,64,99,73,113,170,117,88,65,103,73,0,99,101,39,74,82,15,23,34,47,100,92,
52,51,71,71,70,37,95,124,94,33,73,102,12,62,30,41,34,27,27,56,59,25,104,63,65,30,30,19,13,58,72,19,150,95,99,72,201,120,82,100,13,131,60,233,115,217,31,202,152,94,159,101,111,111,125,72,53,0,76,149,164,94,186,28,20,18,27,165,207,123,78,63,225,163,36,4,22,23,60,13,128,99,66,116,56,183,94,99,163,13,23,0,52,125,17,113,60,61,65,134,
32,37,80,85,17,88,70,95,61,44,80,6,49,0,0,77,40,37,13,25,0,75,72,49,61,54,41,40,13,43,57,115,115,98,128,118,158,98,162,130,84,137,130,162,175,27,29,175,116,173,155,178,139,123,88,43,45,71,156,126,112,173,61,78,6,24,45,101,124,236,145,98,76,63,51,8,17,18,78,38,85,0,0,65,141,0,38,139,131,27,21,41,30,27,115,59,0,44,55,69,
29,25,32,28,19,72,73,47,63,5,28,72,71,33,80,45,60,43,38,43,53,23,62,66,38,20,52,43,33,55,0,85,71,147,34,79,136,123,118,155,82,102,0,255,87,96,32,178,65,255,159,21,210,139,101,23,0,0,195,85,42,37,233,213,3,14,41,101,184,139,69,105,115,20,34,22,11,17,64,126,145,74,128,122,74,64,134,76,17,144,89,82,17,38,63,24,60,65,57,85,
11,16,8,14,17,49,42,20,78,39,52,45,40,0,72,36,58,49,53,4,23,37,44,32,11,21,26,24,19,44,38,57,206,172,123,43,92,149,104,90,39,9,34,102,88,20,30,173,107,103,114,86,137,69,85,82,70,144,218,60,227,99,236,49,14,6,37,96,116,104,66,75,81,51,30,23,11,25,32,39,197,107,97,93,105,179,74,91,66,47,59,82,54,19,43,65,56,6,47,38,
42,29,26,15,20,0,24,57,38,20,67,12,62,25,71,63,28,27,28,58,94,62,31,45,41,44,41,7,0,49,86,107,170,99,46,84,204,121,136,12,184,127,147,134,124,90,28,134,90,108,116,142,245,186,71,53,55,180,224,89,129,165,68,169,24,16,34,162,55,222,219,89,43,34,17,22,23,15,0,255,44,198,108,151,169,155,76,52,23,45,65,28,21,29,59,57,27,42,55,33,
19,26,30,60,34,40,29,6,36,13,48,46,50,63,86,25,34,19,0,29,38,65,82,63,17,44,71,29,45,103,4,120,207,132,58,84,51,124,56,255,171,128,202,71,120,14,26,76,114,132,122,130,174,202,112,67,39,80,143,212,5,92,147,164,64,7,48,40,99,72,63,0,71,7,7,9,15,8,123,0,213,190,126,76,27,104,82,50,71,125,62,49,60,20,19,27,80,0,60,58,
3,34,24,27,17,54,13,19,17,38,20,30,52,72,56,60,47,55,48,60,42,103,33,51,31,41,87,26,28,97,139,92,131,88,22,103,0,99,74,152,149,157,105,180,102,34,29,193,94,87,55,158,204,122,83,85,74,87,129,122,148,39,192,27,160,9,42,42,77,91,44,34,50,32,26,3,14,19,44,241,167,67,144,161,107,142,86,70,83,87,99,63,48,69,9,9,50,17,39,43,
13,18,25,8,18,24,36,28,19,39,54,47,65,24,66,53,48,54,10,42,76,15,18,23,39,25,43,22,66,50,77,51,230,51,62,43,102,0,61,104,38,117,133,193,126,25,81,71,54,75,120,85,0,102,88,224,59,25,255,137,175,93,219,101,145,19,45,115,67,97,90,92,48,52,55,15,27,3,72,81,179,140,179,110,80,50,93,83,0,23,15,80,96,40,50,31,34,24,38,21,
30,19,12,28,29,13,36,56,13,31,17,21,29,50,66,40,14,62,65,52,47,54,72,34,44,33,33,58,123,123,42,77,114,111,59,98,51,115,187,138,104,255,104,107,62,5,84,13,93,74,137,0,114,165,180,0,160,44,255,118,236,180,255,27,20,29,58,86,58,78,100,74,40,62,31,45,22,8,0,217,202,0,98,133,49,51,63,160,52,74,80,0,115,100,15,42,48,23,25,25,
25,28,45,33,15,45,22,32,38,60,47,51,60,44,31,61,83,53,45,47,73,50,62,14,52,45,40,17,230,19,41,189,136,112,48,75,170,143,67,106,0,159,38,104,45,5,25,23,130,150,111,153,39,34,146,29,93,57,180,33,92,155,61,146,146,94,21,59,90,60,58,86,29,43,45,3,28,21,101,197,62,228,28,118,97,133,101,87,108,86,144,85,41,46,59,32,34,0,50,62,
10,21,16,6,12,25,12,22,33,29,34,27,54,46,24,17,8,48,55,43,79,76,46,0,53,0,46,240,106,159,80,148,244,157,57,190,161,155,68,126,104,180,67,99,16,3,79,217,27,53,255,125,128,108,163,117,94,209,255,202,210,55,98,197,93,137,21,70,150,75,84,32,54,44,33,29,28,18,143,175,93,48,89,102,132,119,127,214,95,58,134,78,70,43,72,44,14,42,60,30,
26,22,5,34,21,11,29,30,23,42,30,53,57,2,15,62,35,61,65,55,20,14,44,80,22,24,67,80,73,101,118,109,255,147,179,89,150,115,132,74,137,97,186,10,8,4,139,115,10,25,74,115,53,112,26,88,47,127,67,232,157,124,123,94,93,99,0,67,59,68,79,71,29,23,36,23,41,4,41,64,82,158,104,110,133,97,119,19,94,92,134,79,145,68,38,60,47,59,21,68,
12,24,22,13,31,27,24,15,43,21,30,42,18,45,38,17,63,42,29,39,40,45,51,49,55,95,223,151,163,126,102,106,148,64,109,79,82,39,120,187,229,76,162,22,2,0,178,143,10,51,102,194,135,67,170,209,0,232,101,217,97,180,149,176,173,135,11,55,50,33,19,45,44,29,28,17,22,11,76,163,118,59,125,71,62,191,70,23,87,76,54,48,71,144,66,69,14,23,40,42,
13,6,45,46,12,33,16,19,25,15,46,0,25,26,36,85,33,7,19,14,37,65,41,38,135,70,0,87,0,76,107,78,25,9,91,0,255,79,132,153,69,16,28,47,13,7,14,110,110,74,95,202,109,185,122,37,71,171,231,63,142,0,0,151,111,36,9,10,52,142,0,70,89,58,21,14,13,7,64,104,247,97,194,0,14,111,120,35,80,152,99,41,85,157,60,83,38,48,39,46,
3,27,14,0,13,15,28,25,40,9,6,30,42,45,56,19,35,38,42,40,42,15,33,74,91,176,116,109,126,25,28,48,69,58,136,92,121,143,62,212,154,255,30,31,2,6,120,28,66,30,119,75,99,45,202,86,9,140,146,176,78,167,118,142,120,83,11,47,39,8,73,76,26,64,37,20,19,0,0,93,92,124,139,227,128,212,105,113,155,66,81,89,69,46,60,82,32,7,49,64,
71,51,24,41,12,23,19,30,14,10,2,57,16,54,33,7,39,27,62,53,36,30,53,150,117,53,31,147,66,64,13,38,20,52,109,198,212,151,145,204,77,28,73,11,18,29,71,89,121,161,93,202,153,115,92,66,212,107,67,81,71,94,132,117,48,92,29,86,40,119,82,109,69,14,21,20,21,5,71,128,54,159,47,12,209,119,185,0,175,114,5,122,116,81,17,0,46,42,35,40,
15,26,13,19,26,38,29,20,23,25,38,44,44,44,43,17,53,37,55,15,42,36,168,124,193,233,163,109,199,69,58,11,30,11,38,97,81,197,24,97,118,121,26,25,20,36,53,69,50,32,125,162,53,93,215,130,131,40,255,221,102,58,164,132,62,52,14,59,19,56,47,4,80,68,31,40,12,20,29,68,177,209,118,255,23,94,166,63,89,81,161,109,77,111,43,98,38,1,32,36,
22,35,30,19,36,11,8,6,15,26,15,25,31,18,39,37,21,30,15,25,23,13,42,228,125,87,90,127,103,107,26,17,17,19,143,113,171,80,74,87,98,67,34,44,127,22,112,34,96,136,193,81,205,110,131,101,180,53,135,94,167,95,8,191,149,15,9,32,51,31,29,11,44,23,22,24,19,19,30,45,140,61,154,84,209,170,176,134,143,177,156,113,118,2,54,22,27,41,32,37,
33,35,24,15,37,13,24,0,13,15,28,36,16,34,24,9,21,28,38,4,6,51,165,65,18,102,40,54,0,137,67,46,20,25,183,145,149,37,154,165,169,86,21,127,215,171,112,0,129,182,255,15,170,159,188,231,255,80,183,84,117,76,92,95,112,30,33,68,45,51,59,35,59,46,36,51,42,36,31,39,101,57,228,180,86,60,69,8,59,97,72,58,97,130,120,79,33,54,15,45,
31,29,24,3,15,11,48,3,18,34,0,45,24,0,41,59,3,55,12,38,12,62,158,18,41,125,104,98,78,114,0,30,9,175,239,43,83,82,134,83,100,32,16,125,130,207,67,173,224,168,180,192,113,135,35,229,205,124,75,145,140,181,96,127,0,78,54,64,57,42,61,86,44,74,42,14,0,30,13,33,30,9,119,116,97,104,136,178,16,91,80,64,17,34,100,112,57,25,48,18,
40,21,42,18,38,35,24,12,7,19,38,31,34,30,35,0,34,22,26,13,20,105,146,241,201,82,94,89,46,24,45,15,2,62,87,43,77,13,101,174,87,11,6,235,98,255,90,152,57,165,30,57,142,186,97,114,139,117,172,63,66,20,54,171,8,38,0,51,51,87,68,46,17,47,19,12,26,12,24,37,35,80,17,80,100,255,62,186,143,111,126,118,81,49,3,121,50,32,44,37,
29,27,16,19,27,32,22,14,45,34,54,9,38,13,10,37,32,18,39,5,0,168,93,49,135,48,118,147,27,8,28,18,13,5,111,118,72,17,76,51,92,28,89,132,0,38,239,67,161,167,154,137,22,167,163,131,99,92,102,82,59,54,123,29,40,58,48,62,74,22,61,51,33,23,36,50,11,38,32,17,0,91,151,131,195,100,246,88,80,95,28,69,33,124,100,2,2,12,20,38,
20,33,37,32,17,19,26,39,20,18,32,25,37,6,24,22,29,4,25,27,83,53,170,82,105,0,71,49,171,38,30,21,26,1,58,52,28,115,94,39,47,11,35,204,191,76,99,63,114,95,123,29,129,142,67,52,101,100,147,108,63,20,40,35,63,72,81,37,38,37,36,9,45,75,53,29,34,12,41,77,29,57,100,179,79,137,219,152,131,0,194,63,51,56,44,72,61,23,18,44,
46,10,38,20,19,26,34,24,31,32,20,25,28,24,14,20,29,29,22,14,74,71,36,105,137,29,102,83,52,60,13,6,16,17,32,37,33,147,129,59,32,10,67,110,150,73,125,106,44,108,167,126,133,1,17,21,68,140,66,2,30,70,9,28,58,72,42,63,51,35,62,38,70,19,35,26,25,22,34,151,42,57,154,127,117,123,63,101,115,103,45,95,71,43,47,60,27,25,47,4,
14,44,35,35,18,24,36,21,8,18,10,9,15,30,21,0,15,17,18,29,128,95,60,94,30,30,23,77,50,60,12,13,24,18,60,26,17,118,14,15,17,22,14,0,163,183,135,135,149,98,102,197,88,18,68,88,24,89,99,81,99,80,61,0,52,22,42,41,32,21,34,30,7,10,28,20,3,70,155,53,99,253,65,141,139,123,113,79,78,166,86,73,33,30,50,50,46,26,50,17,
26,44,17,28,15,41,4,27,37,18,13,33,19,23,0,20,17,14,9,15,9,151,125,95,100,71,100,105,35,31,0,7,17,14,18,22,39,6,46,35,15,11,11,122,138,113,134,27,0,35,74,80,142,60,60,47,123,127,93,64,54,79,54,42,17,22,0,32,28,35,21,18,14,18,4,40,221,142,92,179,117,58,67,161,51,20,0,103,26,148,33,41,30,26,71,45,56,9,29,62,
19,9,22,34,46,7,15,21,12,23,11,13,16,5,19,27,20,12,27,32,56,70,57,72,84,60,61,36,35,20,15,0,21,7,22,61,107,44,24,27,2,12,45,140,16,118,53,71,75,54,39,108,34,68,41,58,19,215,67,162,106,95,34,34,9,28,14,8,7,19,21,10,18,16,31,89,61,45,142,111,145,78,69,129,181,60,99,93,44,95,54,42,40,48,25,56,32,49,78,69,
21,32,0,27,12,12,13,34,16,15,12,20,9,15,20,16,11,17,24,33,41,0,123,72,115,99,99,103,90,10,14,10,4,2,20,168,117,39,13,23,6,13,95,201,135,123,83,108,55,136,16,82,97,51,72,35,62,133,78,49,63,61,18,26,22,21,28,30,22,34,30,21,24,57,108,187,69,74,64,98,0,142,198,89,78,125,81,104,70,73,65,35,56,48,52,71,41,47,18,49,
46,0,14,10,5,27,26,8,14,23,26,41,26,41,19,55,17,21,59,41,10,88,79,48,83,52,44,144,56,27,14,12,0,21,33,147,89,19,14,28,20,22,132,89,129,140,93,139,49,169,144,168,122,92,139,51,140,124,71,55,85,36,12,31,27,17,27,30,30,17,27,36,30,27,165,102,53,87,10,42,58,56,105,124,45,55,66,63,52,85,60,39,50,62,34,59,47,46,43,38,
41,36,18,32,47,25,20,17,46,32,10,21,2,14,18,36,27,6,49,131,2,79,31,52,58,82,52,98,2,15,8,10,9,15,93,129,20,12,13,15,11,38,207,156,36,162,105,128,115,160,117,242,117,85,104,205,152,42,52,39,100,22,23,56,62,23,11,15,12,0,7,75,91,27,35,49,151,114,66,49,124,108,31,103,96,176,80,55,42,58,90,35,46,42,63,44,32,22,15,14,
50,21,29,29,23,25,19,34,15,27,24,17,15,31,34,22,6,25,44,71,25,51,31,8,81,97,72,50,15,31,6,8,7,4,3,67,16,13,8,12,29,3,179,227,100,234,99,98,85,0,193,59,0,84,113,161,0,29,61,69,62,10,28,39,9,24,20,33,16,25,198,211,5,74,80,154,128,60,0,67,139,35,108,42,64,75,50,88,31,9,27,61,53,41,29,19,17,28,17,8,
22,18,8,23,13,21,11,15,14,22,19,0,31,21,23,6,18,43,58,117,99,84,7,255,185,156,255,255,0,10,14,23,2,11,25,35,11,7,6,17,45,88,141,168,180,167,130,79,51,122,0,162,94,25,65,52,72,20,50,28,3,34,60,77,72,31,34,61,55,63,130,138,122,57,104,101,115,114,48,48,20,65,62,45,65,60,53,41,40,51,38,60,58,7,30,61,36,42,21,47,
23,41,23,22,31,3,31,18,29,5,18,30,14,18,15,26,31,11,96,95,27,221,255,255,83,255,187,166,246,53,30,33,110,109,45,15,12,12,15,15,83,137,78,186,68,131,138,105,50,114,130,161,132,124,84,67,142,64,53,33,23,47,28,32,23,25,20,62,99,34,213,106,50,91,93,6,100,25,14,24,74,19,46,25,100,30,16,0,33,49,20,17,46,11,17,33,44,27,52,25,
12,19,14,19,19,6,28,12,22,23,13,36,26,28,14,10,118,79,88,16,203,219,242,19,52,74,48,41,17,92,22,30,0,17,8,10,9,14,5,17,50,178,231,131,151,167,111,33,45,111,91,76,38,34,100,30,45,49,18,25,22,30,16,24,26,13,88,113,66,56,132,66,118,109,74,83,35,40,56,76,82,79,37,39,18,25,13,35,43,41,10,47,36,31,55,13,0,21,21,39,
11,25,17,0,1,11,36,20,19,25,19,14,17,26,13,58,45,17,19,22,144,147,37,31,165,255,140,92,25,53,3,17,2,10,16,11,5,14,8,30,66,176,70,0,110,51,89,0,35,26,54,50,103,42,63,55,98,51,23,31,18,31,15,21,43,237,255,184,254,65,48,94,19,79,37,59,42,53,71,65,30,46,46,12,28,25,40,28,29,39,10,22,15,18,44,31,24,40,37,37,
20,24,13,4,27,0,24,8,24,21,27,31,19,14,29,34,7,31,26,13,166,52,70,175,255,255,163,13,80,56,8,26,8,64,53,23,12,15,14,48,121,57,82,137,146,132,60,35,94,44,136,67,73,94,78,74,24,50,0,22,21,8,15,21,55,59,84,130,128,255,220,68,7,51,34,26,49,65,28,40,65,60,73,50,4,25,12,14,17,40,25,34,22,36,36,6,53,42,42,37,
14,3,13,9,16,21,15,28,28,23,24,16,19,6,11,29,34,12,13,7,88,22,178,0,18,203,0,96,0,43,32,0,23,22,42,22,74,17,35,14,106,44,121,84,163,24,151,65,71,68,149,83,62,63,73,62,49,38,17,8,21,14,27,33,17,223,181,81,206,18,255,119,79,56,60,53,36,12,56,41,55,51,51,6,62,30,23,13,43,49,27,18,15,3,20,34,29,30,45,16,
11,2,15,22,47,35,25,50,31,33,27,0,27,7,27,16,10,24,39,17,68,237,255,104,92,255,152,108,97,69,22,19,31,53,61,48,76,148,53,25,51,12,46,72,122,88,100,113,91,168,247,112,88,75,53,46,15,30,25,0,15,16,16,14,232,255,133,242,68,89,65,255,135,0,56,31,59,34,28,37,19,43,60,28,13,19,17,59,36,42,21,34,60,16,25,39,39,27,32,36,
1,23,19,30,11,27,24,14,33,22,23,16,25,22,16,38,8,41,11,20,58,220,197,255,244,183,204,68,181,0,25,17,35,52,51,54,85,19,109,73,69,54,62,14,87,81,54,111,53,33,139,48,50,90,91,65,0,11,19,9,6,12,31,141,255,252,163,255,108,0,208,189,255,84,23,47,60,10,35,4,28,39,24,25,45,17,27,25,27,46,29,52,10,50,15,33,25,16,27,33,
18,33,23,12,22,24,22,24,18,3,31,12,36,56,68,72,59,92,80,8,8,224,156,109,102,174,181,225,37,2,0,0,20,65,48,98,74,50,54,96,117,65,80,83,112,42,33,60,22,11,32,32,25,2,68,29,12,8,31,30,12,13,159,255,229,0,183,240,214,204,57,161,254,197,16,50,20,33,31,17,22,14,46,34,32,31,25,26,45,51,20,61,27,17,53,35,22,29,44,60,
23,32,14,15,15,29,35,16,14,35,35,52,69,41,42,18,47,33,76,53,142,235,172,145,239,255,191,154,115,13,41,7,20,57,77,28,90,55,72,98,64,77,67,72,75,117,40,80,25,30,54,60,34,93,23,38,23,11,21,28,22,111,88,0,207,218,89,2,255,126,126,192,7,255,34,22,17,12,26,33,15,14,21,23,22,25,24,31,9,44,45,4,38,10,26,29,35,41,23,13,
27,34,23,13,8,21,5,22,62,58,68,45,104,14,56,64,89,70,65,83,212,237,113,149,150,149,157,201,132,0,24,22,14,38,40,33,53,57,89,15,99,8,62,63,97,91,32,117,124,60,60,173,158,25,22,36,35,119,50,68,137,0,255,32,119,255,228,179,142,79,255,139,29,215,40,8,3,14,5,25,42,28,32,18,21,35,0,53,36,47,16,39,17,32,38,27,23,47,17,51,
21,17,29,28,22,27,108,92,156,77,92,114,91,42,14,43,36,39,58,83,199,130,125,121,113,255,198,0,19,32,22,38,6,24,41,11,29,57,42,74,53,56,69,62,56,69,39,72,48,135,130,45,33,11,30,12,16,22,40,98,205,167,202,0,108,157,98,255,223,91,48,63,0,27,67,20,32,16,32,39,39,29,34,52,11,22,15,32,85,26,37,10,46,56,55,38,31,32,35,10,
30,34,14,12,128,72,196,75,105,7,100,59,51,104,51,58,80,39,76,236,255,183,182,68,159,254,57,140,134,30,11,27,16,17,32,36,33,7,31,28,52,25,72,44,29,35,53,38,21,40,0,2,0,27,11,20,58,70,188,255,84,255,92,255,114,153,31,72,255,219,152,19,27,45,23,14,0,40,23,36,24,9,24,21,45,26,25,71,29,51,55,30,23,28,48,46,22,63,0,6
};
////////////////////////////////////////////////////////////////////////////////////////

int intcmp(const void *aa, const void *bb)
{
    const int *a = aa, *b = bb;
    return (*a < *b) ? -1 : (*a > *b);
}

//Simple reflection of a 1D index to given [0, size) bounds
int reflect_index(int idx, int size) {
  if (idx < 0) idx = -idx;
  if (idx >= size) idx = size+size - idx - 1;
}

void filter(double* input, int inputLen, double* output, int outputRows, int outputCols)
{
  const int NUM_META_PARAMS;
  int halfKw = input[0];  //filter kernel half width
  int halfKh = input[1];  //filter kernel half height

  // double* img = &(input[NUM_META_PARAMS]);

  //Verify correct image size
  // assert (intputLen - NUM_META_PARAMS == w*h);

  //Loop row-wise over all image pixels
  int kw = 2*halfKw + 1;
  int kh = 2*halfKh + 1;
  int* neighborhood = malloc(kw*kh*sizeof(double)); //filled with vals in each pixels local neighborhood
  for (int y = 0; y < outputRows; y++) {
    for (int x = 0; x < outputCols; x++) {
      //Apply median filter to this pixel based on its neighborhood
      int medianVal = 0;
      for (int ky = 0; ky < kh; ky++) {
        for (int kx = 0; kx < kw; kx++) {
          int ny = reflect_index(y + ky - halfKh, outputRows);
          int nx = reflect_index(x + kx - halfKw, outputCols);
          neighborhood[ky*kw + kx] = img[ny*outputCols + nx];
        }
      }
      qsort(neighborhood, kw*kh, sizeof(int), intcmp);
      int medianValIdx = (kh/2)*kw + kw/2;
      output[y*outputCols + x] = neighborhood[medianValIdx];
      // output[y*outputCols + x] = img[y*outputCols + x]; //TEST
    }
  }
  free(neighborhood);
}