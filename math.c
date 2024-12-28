#include "math.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
void print_bfloat(bfloat16 b){
	uint32_t c = 0;
	((bfloat16*) &c)[1] = b; //little endian
	printf("%f", *(float*) &c);
}
float to_float32(bfloat16 b){
	uint32_t c = 0;
	((bfloat16*) &c)[1] = b;
	return *(float*) &c;
}
bfloat16 truncate_f32(float f){
	return ((bfloat16*) &f)[1];
}
void print_mat(mat* m){
	for(size_t i = 0; i < m->M; i++){
		for(size_t j = 0; j < m->N; j++){
			if(j != 0) printf(" ");
			print_bfloat(m->buff[i * m->N + j]);
		}
		printf("\n");
	}
} 
void to_npy(mat* m, char* path){
	int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 00644);
	char lol[10] = "xNUMPY";
	lol[0] = -109; 
	lol[6] = 1; //major version
	lol[7] = 0; //minor version
	char header[65536];
	sprintf(header, "{'descr': '<f4', 'fortran_order': False, 'shape': (%zu, %zu), }", m->M, m->N);
	unsigned short hlen = strlen(header);
	short k = 16 - (hlen + 11) % 16;
	for(short i = 0; i < k; i++){
		strcat(header, " ");
	}
	strcat(header, "\n");
	*((unsigned short*) (lol + 8)) = (hlen + k + 1);
	write(fd, lol, 10);
	write(fd, header, hlen + k + 1);
	for(size_t i = 0; i < m->M; i++){
		for(size_t j = 0; j < m->M; j++){
			float f = to_float32(m->buff[i * m->N + j]);
			write(fd, (char*) &f, 4);
		}
	}
}
void mm(mat* a, mat* b, mat* c){
	if(a->N != b->M){
		fprintf(stderr, "Dimension Error: a has %zu x %zu and b has %zu x %zu\n", a->M, a->N, b->M, b->N);
		exit(1);
	}	
	c->M = a->M;
	c->N = b->N;
	c->buff = malloc(c->M * c->N * sizeof(bfloat16));
	for(size_t i = 0; i < c->M; i++){
		for(size_t j = 0; j < c->N; j++){
			float f = 0.0;
			for(size_t k = 0; k < a->N; k++){
				f += to_float32(a->buff[i * a->N + k]) * to_float32(b->buff[k * b->N + j]);
			}
			c->buff[i * c->N + j] = truncate_f32(f);
		}
	}
}
void mv(mat* a, vec* b, vec* c){
	mat* bm = malloc(sizeof(mat));
	bm->M = b->N;
	bm->N = 1;
	bm->buff = b->buff;
	mat* cm = malloc(sizeof(mat));
	mm(a, bm, cm);
	c->N = cm->M;
	c->buff = cm->buff;
}
/*
void cs(float** c, float** s, float base, size_t d){
	// c,s : output
	// base: period of lowest frequency turner
	// d: dimension
	size_t hd = d/2;
	float* cbuf = malloc(d * sizeof(float));
	float* sbuf = malloc(d * sizeof(float));
	for(size_t i = 0; i < hd; i++){
		float power = - (float) i / (float) hd;
		float freq; //freq = base ^ power;
		float cf; //cos(freq)
		float sf; //sin(freq)
		cbuf[2 * i] = cf;
		cbuf[2 * i + 1] = cf;
		sbuf[2 * i] = sf;
		sbuf[2 * i + 1] = sf;
	}
	*c = cbuf;
	*s = sbuf;
}
void ro(mat* a, vec* c, vec* s){
	// a : (hds * nh, seqlen)

}
void sa(struct sablock* s, mat* ctx){
	//ctx : (d, seqlen)
	//d = 4096
	mat* qc = malloc(sizeof(mat));
	mat* kc = malloc(sizeof(mat));
	mat* vc = malloc(sizeof(mat));

	mm(&s->q, ctx, qc);
	mm(&s->k, ctx, kc);
	mm(&s->v, ctx, vc);
	//qc : 4096 x seqlen => 4 x 8 x 128 x seqlen
	//kc : 1024 x seqlen => 8 x 128 x seqlen
	//vc : 1024 x seqlen => 8 x 128 x seqlen
	for(int h = 0; h < 32; h++){
		//a: seqlen x seqlen
		mat a;
		//smax, rowise

	}
	mat preo;
	//preo: 32 x 128 x seqlen
	mat dctx;
	mm(&preo, &s->o, dctx);
	return madd(ctx, dctx);
}
*/
