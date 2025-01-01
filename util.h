#pragma once
#include <stdint.h>
#include <stddef.h>
#include "json.h"
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
void mm(const mat* a, const mat* b, mat* out);
void sa(const mat* k, const mat* o, const mat* q, const mat* v, mat* ctx);
void udg(const mat* gate, const mat* up, const mat* down, mat* ctx);
void print_bfloat(bfloat16 b);	
void print_mat(mat* m);
void to_npy(const mat* m, char* path);
char* tokenize(int);
bfloat16 truncate_f32(float);
float to_float32(bfloat16);
mat* embed(int*, int, const char*);
void rms_norm(mat*, const mat*, float);
const mat* extract_mat(struct json*, char* base, char* name);
