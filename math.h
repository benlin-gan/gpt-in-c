#pragma once
#include <stdint.h>
#include <stddef.h>
typedef int16_t bfloat16;
struct mat{
	size_t M;
	size_t N;
	bfloat16* buff;
};
typedef struct mat mat;
struct vec{
	size_t N;
	bfloat16* buff;
};
typedef struct vec vec;
void mm(mat* a, mat* b, mat* out);
struct sablock{
	mat k;
	mat o;
	mat q;
	mat v;
};
void sa(struct sablock* s, mat* ctx);
void print_bfloat(bfloat16 b);	
void print_mat(mat* m);
void to_npy(mat* m, char* path);
bfloat16 truncate_f32(float);

