#include "util.h"
#include "gpt2.h"
#include "json.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/mman.h>
void print_tuple(size_t* arr, size_t n){
	printf("(");
	for(size_t i = 0; i < n; i++){
		if(i != 0) printf(", ");
		printf("%zu", arr[i]);
	}
	printf(")");
}
size_t total_addressable(const grid* m){
	size_t out = 1;
	for(size_t i = 0; i < m->sh; i++){
		out *= m->shape[i];
	}
	return out;
}
size_t actual_addressable(const grid* m){
	size_t out = 1;
	for(size_t i = m->vind; i < m->sh; i++){
		out *= m->shape[i];
	}
	return out;
}
void dump_grid(const grid* m, char* path){
	int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 00644);
	write_npy_header(fd, m->shape, m->sh);
	size_t bsize = actual_addressable(m); 
	size_t repeat = total_addressable(m) / bsize;
	for(size_t i = 0; i < repeat; i++){
		write(fd, m->buff, 4 * bsize);
	}
}
void address_interp_rec(size_t* out, const size_t* shape, size_t sh, size_t i){
	if(sh == 0) return;
	out[sh - 1] = i % shape[sh - 1];
	address_interp_rec(out, shape, sh - 1, i / shape[sh - 1]);
}
size_t* address_interp(const grid* m, size_t i){
	size_t* out = malloc(8 * sizeof(size_t));
	address_interp_rec(out, m->shape, m->sh, i);
	return out;
}
float* lookup(const grid* m, const size_t* ii){
	bool valid = true;
	for(size_t i = 0; i < m->sh; i++){
		if(ii[i] >= m->shape[i]){
			valid = false;
			break;
		}
	}
	if(!valid){
		fprintf(stderr, "indexing error: (");
		for(size_t i = 0; i < m->sh; i++){
			if(i != 0) fprintf(stderr, ", ");
			fprintf(stderr, "%zu", ii[i]);
		}
		fprintf(stderr, ") in grid of shape (");
		for(size_t i = 0; i < m->sh; i++){
			if(i != 0) fprintf(stderr, ", ");
			fprintf(stderr, "%zu", m->shape[i]);
		}
		fprintf(stderr, ")\n");
		exit(1);
	}
	size_t out = 0;
	for(size_t i = m->vind; i < m->sh; i++){ //skip virtual indices
		out += ii[i];
		if(i != m->sh - 1) out *= m->shape[i + 1];	
	}
	return &m->buff[out];
}
void scale(grid* m, float f){
	for(size_t i = 0; i < actual_addressable(m); i++){
		m->buff[i] *= f;
	}
}
float swish(float x){
	return 0.5 * x * (1 + tanh(sqrt(M_2_PI) * (x + 0.044715 * x * x * x)));
}
void swishb(grid* m){
	for(size_t i = 0; i < actual_addressable(m); i++){
		m->buff[i] = swish(m->buff[i]);
	}
}
grid* shallow_copy(const grid* m){
	grid* out = malloc(sizeof(grid));
	memcpy(out, m, sizeof(grid));	
	return out;
}
grid* deep_copy(const grid* m){
	grid* out = malloc(sizeof(grid));
	memcpy(out, m, sizeof(grid));	
	size_t to_copy = actual_addressable(m);
	out->buff = malloc(to_copy * 4);
	memcpy(out->buff, m->buff, to_copy * 4);
	return out;
}
grid* new_grid(const size_t* s, size_t n){
	grid* out = malloc(sizeof(grid));
	out->sh = n;
	out->vind = 0;
	size_t to_alloc = 1;
	for(size_t i = 0; i < n; i++){
		out->shape[i] = s[i];
		to_alloc *= s[i];
	}
	out->buff = malloc(sizeof(float) * to_alloc);
	return out;
}
void destroy_grid(grid* g){
	free(g->buff);
	free(g);
}
void ln(grid* x, const grid* m, const grid* b){
	//x : ? x d
	//m, b : d
	size_t d = m->shape[m->sh - 1];
	size_t n = actual_addressable(x)/d;
	for(size_t i = 0; i < n; i++){
		float sum = 0.0;
		float sqsum = 0.0;
		for(size_t j = 0; j < d; j++){
			float f = x->buff[i * d + j];
			sum += f;
			sqsum += f * f;
		}
		float mean = sum / d;
		float var = (sqsum - sum * sum / d) / (d - 1);
		for(size_t j = 0; j < d; j++){
			float f = x->buff[i * d + j];
			f = (f - mean)/(sqrt(var + 1e-5)) * m->buff[j] + b->buff[j];
			x->buff[i * d + j] = f;
		}
	}
}
void mask(grid* m){
	if(m->vind != 0){
		fprintf(stderr, "unimplemented\n");
		exit(1);
	}
	for(size_t i = 0; i < total_addressable(m); i++){
		size_t* ii = address_interp(m, i);
		size_t q = ii[m->sh - 2];
		size_t k = ii[m->sh - 1];
		if(q < k){
			m->buff[i] = -INFINITY;
		}
		free(ii);
	}
}
void smax(grid* m){
	//last dimension:
	if(m->vind != 0){
		fprintf(stderr, "unimplemented\n");
		exit(1);
	}
	size_t d = m->shape[m->sh - 1];
	size_t n = actual_addressable(m)/d;
	for(size_t i = 0; i < n; i++){
		float norm = 0.0;	
		for(size_t j = 0; j < d; j++){
			float f = m->buff[i * d + j];
			if(f > -INFINITY){
				m->buff[i * d + j] = expf(f);
			}else{
				m->buff[i * d + j] = 0.0;
			}
			norm += m->buff[i * d + j];
		}
		for(size_t j = 0; j < d; j++){
			m->buff[i * d + j] /= norm;
		}
	}
}
grid* tp(const grid* m, size_t a, size_t b){
	//copying transpose
	grid* out = new_grid(m->shape, m->sh);
	size_t temp = out->shape[a];
	out->shape[a] = out->shape[b];
	out->shape[b] = temp;
	for(size_t i = 0; i < total_addressable(m); i++){
		size_t* idx = address_interp(m, i);
		temp = idx[a];
		idx[a] = idx[b];
		idx[b] = temp;
		*lookup(out, idx) = m->buff[i];
		free(idx);
	}
	return out;
}
grid* broadcast(const grid* a, const grid* b){
	//return broadcast version of b using virutal indices
	grid* out = shallow_copy(b);
	size_t pfix = a->sh - b->sh;
	for(size_t i = 0; i < a->sh; i++){
		if(i >= pfix){
			out->shape[i] = b->shape[i - pfix];
		}else{
			out->shape[i] = a->shape[i];
		}
	}
	out->vind = pfix;
	out->sh += pfix;
	return out;	
}
void madd(grid* a, const grid* b){
	// a <- a + b

	//elementwise operation guard:
	size_t pfix = a->sh - b->sh;
	for(size_t i = 0; i < b->sh; i++){
		if(a->shape[i + pfix] != b->shape[i]){
			fprintf(stderr, "madd: incompatible dimensions, cannot broadcast\n");
			exit(1);
		}
	}
	grid* bcast = NULL;
	if(a->sh < b->sh){
		fprintf(stderr, "madd: cannot mutable add to a without allocating new memory");
		exit(1);
	}else if(b->sh < a->sh){
		bcast = broadcast(a, b); //remember the memory leak here
		b = bcast;
	}
	for(size_t i = 0; i < total_addressable(a); i++){
		size_t* ii = address_interp(a, i);
		a->buff[i] += *lookup(b, ii);
		free(ii);
	}
	if(bcast != NULL){
		free(bcast);
	}
}
grid* matmul(const grid* a, const grid* b){
	//matmul on the two inner-most indices of a, b
	if(a->sh < 2){
		fprintf(stderr, "matmul: a must have at least 2 dimensions\n");
		exit(1);
	}
	if(b->sh < 2){
		fprintf(stderr, "matmul: b must have at least 2 dimensions\n");
		exit(1);
	}
	if(a->shape[a->sh - 1] != b->shape[b->sh - 2]){
		fprintf(stderr, "matmul: incompatible dimensions x\n");
		exit(1);
	} 
	size_t L = a->shape[a->sh - 1]; //dimension of joined index
	size_t pfix = a->sh - b->sh;
	for(size_t i = 0; i < b->sh - 2; i++){
		if(a->shape[i + pfix] != b->shape[i]){
			fprintf(stderr, "matmul: incompatible dimensions y\n");
			exit(1);
		}
	}
	grid* bcast = NULL;
	if(a->sh < b->sh){
		bcast = broadcast(b, a); //note memory error here
		a = bcast;
	}else if(b->sh < a->sh){
		bcast = broadcast(a, b);
		b = bcast;
	}
	size_t shape[8];
	for(size_t i = 0; i < a->sh - 2; i++){
		shape[i] = a->shape[i];	
	}
	shape[a->sh - 2] = a->shape[a->sh - 2];
	shape[b->sh - 1] = b->shape[b->sh - 1];
	grid* out = new_grid(shape, a->sh);
	for(size_t i = 0; i < total_addressable(out); i++){
		size_t* ii = address_interp(out, i);
		size_t j = ii[out->sh - 2];
		size_t k = ii[out->sh - 1];
		out->buff[i] = 0.0;	
		/*printf("out: ");
		print_tuple(ii, a->sh);
		printf("\n");*/
		for(size_t l = 0; l < L; l++){
			ii[out->sh - 1] = l;
			float x = *lookup(a, ii);
			ii[out->sh - 1] = k;
			ii[out->sh - 2] = l;
			float y = *lookup(b, ii);
			out->buff[i] += x * y;
			ii[out->sh - 2] = j;
			ii[out->sh - 1] = k;
		}
		free(ii);
	}
	if(bcast != NULL){
		free(bcast);
	}
	return out;
}
grid* sea(const grid* q, const grid* qb, const grid* k, const grid* kb, const grid* v, const grid* vb, const grid* o, const grid* ob, const grid* ctx){
	//ctx: n x t x d
	//size_t t = ctx->shape[1];
	size_t d = ctx->shape[2];
	//q, k, v: d x (h * hdim)
	size_t h = 12;
	size_t hdim = d/h;

	//qc, kc, vc: n x t x h x hdim
	grid* qc = matmul(ctx, q);
	madd(qc, qb);
	qc->shape[qc->sh - 1] = h;
	qc->shape[qc->sh++] = hdim; //splitting d -> h * hdim
	grid* kc = matmul(ctx, k);
	madd(kc, kb);
	kc->shape[kc->sh - 1] = h;
	kc->shape[kc->sh++] = hdim;
	grid* vc = matmul(ctx, v);
	madd(vc, vb);
	vc->shape[vc->sh - 1] = h;
	vc->shape[vc->sh++] = hdim;

	//qct, kct, vct: n x h x t x hdim
	grid* qct = tp(qc, 1, 2); 
	grid* kct = tp(kc, 1, 2);
	grid* vct = tp(vc, 1, 2);
	destroy_grid(qc);
	destroy_grid(kc);
	destroy_grid(vc);
	//kctt: n x h x hdim x t
	grid* kctt = tp(kct, 2, 3);
	destroy_grid(kct);
	//score: n x h x t x t
	grid* score = matmul(qct, kctt);
	destroy_grid(qct);
	destroy_grid(kctt);

	//scale
	scale(score, 1.0/sqrt(hdim));
	//mask
	mask(score);
	//do the score => weights conversion
	smax(score);
	
	//score: n x h x t x t
	//vct: n x h x t x hdim
	//av : n x h x t x hdim
	grid* av = matmul(score, vct);
	destroy_grid(score);
	destroy_grid(vct);
	//avt: n x t x (h * hdim)
	grid* avt = tp(av, 1, 2);
	avt->shape[avt->sh - 2] *= avt->shape[avt->sh - 1];
	avt->sh--; //collapse (h * hdim) -> d
	destroy_grid(av);
	//o: (h * hdim) x d 
	//final: n x t x d
	grid* final = matmul(avt, o);
	madd(final, ob);
	destroy_grid(avt);
	return final;
}
grid* mix(const grid* up, const grid* upb, const grid* down, const grid* downb, const grid* ctx){
	//ctx: n x t x d
	//up: d x D
	//down: D x d
	grid* sps = matmul(ctx, up);
	madd(sps, upb);
	swishb(sps);
	grid* final = matmul(sps, down);
	destroy_grid(sps);
	madd(final, downb);
	return final;
}
void tmove(const tblock* t, grid* ctx){
	grid* dctx = deep_copy(ctx);
	ln(dctx, t->ln1, t->ln1b);
	grid* psa = sea(t->q, t->qb, t->k, t->kb, t->v, t->vb, t->o, t->ob, dctx);
	madd(ctx, psa);
	destroy_grid(dctx);
	destroy_grid(psa);
	dctx = deep_copy(ctx);
	ln(dctx, t->ln2, t->ln2b);
	grid* pmix = mix(t->up, t->upb, t->down, t->downb, dctx);
	madd(ctx, pmix);
	destroy_grid(dctx);
	destroy_grid(pmix);
}
grid* embedgpt(int* s, size_t seqlen, const grid* te, const grid* pe){
	size_t shape[8];
	shape[0] = 1;
	shape[1] = seqlen;
	shape[2] = te->shape[te->sh - 1];
	grid* out = new_grid(shape, 3);
	for(size_t i = 0; i < seqlen; i++){
		for(size_t j = 0; j < shape[2]; j++){
			out->buff[i * shape[2] + j] = te->buff[s[i] * shape[2] + j] + pe->buff[i * shape[2] + j];
		}
	}
	return out;
}
grid* logits(gpt2* gpt, int* s, size_t seqlen){
	grid* ctx = embedgpt(s, seqlen, gpt->te, gpt->pe);
	for(int i = 0; i < 12; i++){
		tmove(gpt->blocks[i], ctx);
	}
	ln(ctx, gpt->lnf, gpt->lnfb);
	grid* out = matmul(ctx, gpt->head);
	destroy_grid(ctx);
	return out;
}
grid* extract2grid(struct json* j, char* base, char* name){
	struct node* curr = j->start;
	bool found = false;
	for(; curr != NULL; curr = curr->next){
		if(strcmp(curr->title, name) == 0){
			found = true;
			break;
		}
	}
	if(!found){
		return NULL;
	}
	grid* out = malloc(sizeof(grid));
	struct json* fields = (struct json *) curr->data;
	struct json* shape_array = (struct json *) fields->start->next->data;
	struct json* offsets_array = (struct json *) fields->start->next->next->data;
	curr = shape_array->start;
	out->sh = 0;
	while(curr != NULL){
		out->shape[out->sh++] = *(size_t*) curr->data;
		curr = curr->next;
		if(out->sh == 8){
			fprintf(stderr, "too many indices\n");
			exit(1);
		}
	}
	out->buff = (float*) (base + *(size_t*) offsets_array->start->data);
	return out;
}
tblock* extract_tblock(struct json* j, char* base, int i){
	tblock* out = malloc(sizeof(tblock));
	char names[1024];
	sprintf(names, "h.%d.ln_1.weight", i);
	out->ln1 = extract2grid(j, base, names);
	sprintf(names, "h.%d.ln_1.bias", i);
	out->ln1b = extract2grid(j, base, names);
	sprintf(names, "h.%d.ln_2.weight", i);
	out->ln2= extract2grid(j, base, names);
	sprintf(names, "h.%d.ln_2.bias", i);
	out->ln2b = extract2grid(j, base, names);
	sprintf(names, "h.%d.attn.c_attn.weight", i);
	grid* packed = extract2grid(j, base, names);
	size_t d = packed->shape[0];
	size_t shape[2];
	shape[0] = d;
	shape[1] = d;
	grid* q = new_grid(shape, 2);
	grid* k = new_grid(shape, 2);
	grid* v = new_grid(shape, 2);
	for(size_t i = 0; i < d; i++){
		for(size_t j = 0; j < d; j++){
			q->buff[i * d + j] = packed->buff[i * 3 * d + j];
		}
		for(size_t j = 0; j < d; j++){
			k->buff[i * d + j] = packed->buff[i * 3 * d + d + j];
		}
		for(size_t j = 0; j < d; j++){
			v->buff[i * d + j] = packed->buff[i * 3 * d + d + d + j];
		}
	}
	out->q = q;
	out->k = k;
	out->v = v;
	free(packed);
	sprintf(names, "h.%d.attn.c_attn.bias", i);
	grid* packedb = extract2grid(j, base, names);
	grid* qb = new_grid(shape, 1);
	grid* kb = new_grid(shape, 1);
	grid* vb = new_grid(shape, 1);
	free(qb->buff);
	free(kb->buff);
	free(vb->buff);
	qb->buff = packedb->buff;
	kb->buff = packedb->buff + d;
	vb->buff = packedb->buff + d + d;
	free(packedb);
	out->qb = qb;
	out->kb = kb;
	out->vb = vb;
	sprintf(names, "h.%d.attn.c_proj.weight", i);
	out->o = extract2grid(j, base, names);
	sprintf(names, "h.%d.attn.c_proj.bias", i);
	out->ob = extract2grid(j, base, names);

	sprintf(names, "h.%d.mlp.c_fc.weight", i);
	out->up = extract2grid(j, base, names);
	sprintf(names, "h.%d.mlp.c_fc.bias", i);
	out->upb = extract2grid(j, base, names);

	sprintf(names, "h.%d.mlp.c_proj.weight", i);
	out->down = extract2grid(j, base, names);
	sprintf(names, "h.%d.mlp.c_proj.bias", i);
	out->downb = extract2grid(j, base, names);

	return out;
}
gpt2* load_model(char* path){
	gpt2* out = malloc(sizeof(gpt2));
	int f = open(path, O_RDONLY);
	char* d = mmap(NULL, 1000000000, PROT_READ, MAP_PRIVATE, f, 0);
	out->fd = f;
	out->base = d;
	size_t s = *(size_t*) d;
	char* p = d + s + 8;
	size_t ptr = 8;
	struct json* j = jstring_to_json(d, &ptr, s + 8);
	out->te = extract2grid(j, p, "wte.weight");
	out->pe = extract2grid(j, p, "wpe.weight");
	out->lnf = extract2grid(j, p, "ln_f.weight");
	out->lnfb = extract2grid(j, p, "ln_f.bias");
	out->head = tp(out->te, 0, 1);
	out->blocks = malloc(12 * sizeof(tblock*));
	for(int i = 0; i < 12; i++){
		out->blocks[i] = extract_tblock(j, p, i);
	}
	out->j = j;
	return out;
}
void destroy_tblock(tblock* t){
	destroy_grid(t->q);
	destroy_grid(t->k);
	destroy_grid(t->v);
	free(t->qb);
	free(t->kb);
	free(t->vb);
	free(t->ln1);
	free(t->ln1b);
	free(t->ln2);
	free(t->ln2b);
	free(t->o);
	free(t->ob);
	free(t->down);
	free(t->downb);
	free(t->up);
	free(t->upb);
	free(t);
}
void destroy_model(gpt2* gpt){
	destroy_grid(gpt->head);	
	for(int i = 0; i < 12; i++){
		destroy_tblock(gpt->blocks[i]);
	}
	munmap(gpt->base, 1000000000);
	close(gpt->fd);
	free(gpt->te);
	free(gpt->pe);
	free(gpt->lnf);
	free(gpt->lnfb);
	free(gpt->blocks);
	destroy_json(gpt->j);
	free(gpt);
}
