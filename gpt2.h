#pragma once
#include "json.h"
struct grid{
	size_t shape[8];
	size_t sh; //length of shape
	size_t vind; //number of virtual indices
	float* buff;	
};
typedef struct grid grid;
struct tblock{
	const grid* ln1;
	const grid* ln1b;
	const grid* q;
	const grid* qb;
	const grid* k;
	const grid* kb;
	const grid* v;
	const grid* vb;
	const grid* o;
	const grid* ob;
	const grid* ln2;
	const grid* ln2b;
	const grid* up;
	const grid* upb;
	const grid* down;
	const grid* downb;
};
typedef struct tblock tblock;
const tblock* extract_tblock(struct json* j, char* base, int i);
void tmove(const tblock* t, grid* ctx);
const grid* extract2grid(struct json* j, char* base, char* name);
void dump_grid(const grid* m, char* path);
grid* embedgpt(int*, size_t, const grid*, const grid*);
