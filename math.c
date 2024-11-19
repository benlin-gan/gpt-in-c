#include "math.h"
#include <stdbool.h>
#include <stdio.h>
void print_bfloat(bfloat16 b){
	uint32_t c = 0;
	((bfloat16*) &c)[0] = b;
	printf("%f", *(float*) &c);
}
