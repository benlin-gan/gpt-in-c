#pragma once
#include <stdint.h>
#include <stddef.h>
typedef int16_t bfloat16;
struct mat{
	size_t M;
	size_t N;
	bfloat16* buff;
};
void print_bfloat(bfloat16 b);	
