#pragma once
#include "json.h"
#include <stdbool.h>
struct grid{
	size_t shape[8];
	size_t sh; //length of shape
	size_t vind; //number of virtual indices
	float* buff;	
};
typedef struct grid grid;
struct tblock{
	grid* ln1; //d
	grid* ln1b; //d
	grid* q; //d x (h * hdim) 
	grid* qb; //(h * hdim)
	grid* k; //d x (h * hdim) 
	grid* kb; //(h * hdim)
	grid* v; //d x (h * hdim) 
	grid* vb; //(h * hdim)
	grid* o; //(h * hdim) x d
	grid* ob; //d
	grid* ln2; //d
	grid* ln2b; //d
	grid* up; //d X D
	grid* upb; //D
	grid* down; //D x d
	grid* downb; //d

	grid* kcache; //n x t x h x hdim
	grid* vcache; //n x t x h x hdim
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
	int ctx[1024];
	int* offsets;
	char* toks;
	unsigned short* merges;
};
typedef struct gpt2 gpt2;
void destroy_model(gpt2*);
gpt2* load_model(char* path);
grid* logits(gpt2*, int*, size_t, bool);
void loopgen(gpt2*, int*, size_t);
tblock* extract_tblock(struct json* j, char* base, int i);
void tmove(tblock* t, grid* ctx, bool caching);
grid* extract2grid(struct json* j, char* base, char* name);
void dump_grid(const grid* m, char* path);
grid* embedgpt(int*, size_t, const grid*, const grid*);
void destroy_grid(grid*);
void print_tok(gpt2*, int);
