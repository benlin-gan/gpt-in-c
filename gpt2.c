#include "util.h"
#include "gpt2.h"
size_t total_addressable(const grid* m){
	size_t out = 1;
	for(size_t i = 0; i < m->sh; i++){
		out *= m->shape[i];
	}
	return out;
}
size_t actual_addressable(const grid* m){
	size_t out = 1;
	for(size_t i = vind; i < m->sh; i++){
		out *= m->shape[i];
	}
	return out;
}
size_t* address_interp(const mat* m, size_t i){
	size_t* out = malloc(8 * sizeof(size_t));
	address_interp_rec(out, m->shape, m->sh, i);
	return out;
}
void address_interp_rec(size_t* out, const size_t* shape, size_t sh, size_t i){
	if(sh == 0) return;
	out[sh - 1] = i % shape[sh - 1];
	address_interp_rec(out, shape, sh - 1, i / shape[sh - 1]);
}
float* lookup(const mat* m, const size_t* ii){
	size_t out = 0;
	for(size_t i = vind; i < sh; i++){ //skip virtual indices
		out += ii[i];
		if(i != sh - 1) out *= shape[i + 1];	
	}
	return &m->buff[out];
}
void scale(grid* m, float f){
	for(size_t i = 0; i < actual_addressable(m); i++){
		m->buff[i] *= f;
	}
}
void silub(grid* m){
	for(size_t i = 0; i < actual_addressable(m); i++){
		m->buff[i] = silu(m->buff[i]);
	}

}
void ln(grid* x, const grid* m, const grid* b){
	//x : n x t x d
	for(size_t i = 0; i < actual_addressable(m); i++){
		m->buff[i] = 
	}

}
void mask(grid* m){
	for(size_t i = 0; i < total_addressable(m); i++){
		size_t* ii = address_interp(m, i);
		size_t q = ii[m->sh - 2];
		size_t k = ii[m->sh - 1];
		if(q < k){
			m->buff[i] = -INF;
		}
	}
}
grid* tp(const grid* m, size_t a, size_t b){
	//copying transpose
	grid* out = new_grid(m->shape, m->s);
	size_t temp = out->shape[a];
	out->shape[a] = out->shape[b];
	out->shape[b] = temp;
	for(size_t i = 0; i < total_addressable(m); i++){
		size_t* idx = address_interp(m, i);
		temp = idx[a];
		idx[a] = idx[b];
		idx[b] = temp;
		*lookup(out, idx) = m->buff[i];
	}
	return out;
}
const grid* broadcast(const grid* a, const grid* b){
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
	out->vdim = pfix;
	return out;	
}
void madd(grid* a, const grid* b){
	// a <- a + b

	//elementwise operation guard:
	size_t pfix = a->sh - b->sh;
	for(int i = 0; i < b->sh; i++){
		if(a->shape[i + pfix] != b->shape[i]){
			fprintf(stderr, "madd: incompatible dimensions, cannot broadcast\n");
			exit(1);
		}
	}
	if(a->sh < b->sh){
		fprintf(stderr, "madd: cannot mutable add to a without allocating new memory");
		exit(1);
	}else if(b->sh < a->sh){
		b = broadcast(a, b); //remember the memory leak here
	}
	for(size_t i = 0; i < total_addressable(a); i++){
		size_t* ii = address_interp(m, i);
		a->buff[i] += *lookup(b, ii);
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
	for(int i = 0; i < b->sh - 2; i++){
		if(a->shape[i + pfix] != b->shape[i]){
			fprintf(stderr, "matmul: incompatible dimensions y\n");
			exit(1);
		}
	}
	if(a->sh < b->sh){
		a = broadcast(b, a); //note memory error here
	}else if(b->sh < a->sh){
		b = broadcast(a, b);
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
		for(size_t l = 0; l < L; l++){
			ii[out->sh - 1] = l;
			float x = *lookup(a, ii);
			ii[out->sh - 1] = k;
			ii[out->sh - 2] = l;
			float y = *lookup(b, ii);
			out->buff[i] += x * y;
		}
		free(ii);
	}
	return out;
}
grid* shallow_copy(const grid* m){
	grid* out = malloc(sizeof(grid));
	memcpy(out, m, sizeof(grid));	
	return out;
}
grid* new_grid(size_t* s, size_t n){
	grid* out = malloc(sizeof(grid));
	out->sh = n;
	out->vdim = 0;
	size_t to_alloc = 1;
	for(int i = 0; i < n; i++){
		out->shape[i] = s[i];
		to_alloc *= s[i];
	}
	out->buff = malloc(sizeof(float) * to_alloc);
	return out;
}
void sea(const grid* q, const grid* k, const grid* v, const grid* o, grid* ctx){
	//ctx: n x t x d
	size_t t = ctx->shape[1];
	size_t d = ctx->shape[2];
	//q, k, v: d x (h * hdim)
	size_t h = 12;
	size_t hdim = d/h;

	//qc, kc, vc: n x t x h x hdim
	grid* qc = matmul(ctx, q);
	qc->shape[qc->sh - 1] = h;
	qc->shape[qc->sh++] = hdim; //splitting d -> h * hdim
	grid* kc = matmul(ctx, k);
	kc->shape[kc->sh - 1] = h;
	kc->shape[kc->sh++] = hdim;
	grid* vc = matmul(ctx, v);
	vc->shape[vc->sh - 1] = h;
	vc->shape[vc->sh++] = hdim;

	//qct, kct, vct: n x h x t x hdim
	grid* qct = tp(qc, 1, 2); 
	grid* kct = tp(kc, 1, 2);
	grid* vct = tp(vc, 1, 2);
	free(qc);
	free(kc);
	free(vc);
	//kctt: n x h x hdim x t
	grid* kctt = tp(kct, 2, 3);
	free(kct);
	//score: n x h x t x t
	grid* score = matmul(qct, kctt);
	free(qct, kctt);

	//scale
	scale(score, 1.0/sqrt(hdim))
	//mask
	mask(score);
	//do the score => weights conversion
	smax(score, score->sh-1);
	
	//score: n x h x t x t
	//vct: n x h x t x hdim
	//av : n x h x t x hdim
	grid* av = matmul(score, vct);
	free(score);
	free(vct);
	//avt: n x t x (h * hdim)
	grid* avt = tp(av, 1, 2);
	avt[avt->sh - 2] *= avt[avt->sh - 1];
	avt->sh--; //collapse (h * hdim) -> d
	free(av);
	//o: (h * hdim) x d 
	//final: n x t x d
	grid* final = matmul(avt, o);
	free(avt);
	return final;
}
void mix(const grid* up, const grid* upb, const grid* down, const grid* downb, grid* ctx){
	//ctx: n x t x d
	//up: d x D
	//down: D x d
	grid* sps = matmul(ctx, up);
	madd(sps, upb);
	silub(sps);
	grid* final = matmul(sps, down);
	madd(final, downb);
	return final;
}
