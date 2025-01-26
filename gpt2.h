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
	grid* ln1;
	grid* ln1b;
	grid* q;
	grid* qb;
	grid* k;
	grid* kb;
	grid* v;
	grid* vb;
	grid* o;
	grid* ob;
	grid* ln2;
	grid* ln2b;
	grid* up;
	grid* upb;
	grid* down;
	grid* downb;
};
typedef struct tblock tblock;
struct gpt2{
	int fd;
	char* base;
	struct json* j;
	grid* te;
	grid* pe;
	grid* lnf;
	grid* lnfb;
	grid* head;
	tblock** blocks;
};
typedef struct gpt2 gpt2;
void destroy_model(gpt2*);
gpt2* load_model(char* path);
grid* logits(gpt2*, int*, size_t);
tblock* extract_tblock(struct json* j, char* base, int i);
void tmove(const tblock* t, grid* ctx);
grid* extract2grid(struct json* j, char* base, char* name);
void dump_grid(const grid* m, char* path);
grid* embedgpt(int*, size_t, const grid*, const grid*);
void destroy_grid(grid*);
