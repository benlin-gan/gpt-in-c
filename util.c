#include "util.h"
#include "json.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <math.h>
#define PRINT_DEBUG 0
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
float* to_float_buffer(bfloat16* buf, size_t s){
	bfloat16* out = malloc(2 * s * sizeof(bfloat16));
	for(size_t i = 0; i < s; i++){
		out[2 * i] = 0;
		out[2 * i + 1] = buf[i];
	}
	return (float*) out;
}
void write_npy_header(int fd, size_t* shape, size_t shsize){
	char lol[10] = "xNUMPY";
	lol[0] = -109; 
	lol[6] = 1; //major version
	lol[7] = 0; //minor version
	char header[65536];
	char slist[1000];
	size_t pos = 0;
	for(size_t i = 0; i < shsize; i++){
		size_t len = sprintf(slist + pos, "%zu", shape[i]);
		pos += len;
		if(i != shsize - 1){
		   	sprintf(slist + pos, " ,");
			pos += 2;
		}
	}
	if(shsize == 1){
		slist[pos++] = ',';
	}
	slist[pos] = '\0';
	sprintf(header, "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }", slist);
	unsigned short hlen = strlen(header);
	short k = 16 - (hlen + 11) % 16;
	for(short i = 0; i < k; i++){
		strcat(header, " ");
	}
	strcat(header, "\n");
	*((unsigned short*) (lol + 8)) = (hlen + k + 1);
	write(fd, lol, 10);
	write(fd, header, hlen + k + 1);
}
void to_npy(const mat* m, char* path){
	int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 00644);
	size_t sh[2];
	sh[0] = m->M;
	sh[1] = m->N;
	write_npy_header(fd, sh, 2);
	float* out = to_float_buffer(m->buff, sh[0] * sh[1]);
	write(fd, out, sh[0] * sh[1] * 4);
	free(out);
}
void to_fnpy(const float* f, size_t* shape, size_t shsize, char* path){
	int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 00644);
	size_t l = 4;
	for(size_t i = 0; i < shsize; i++){
		l *= shape[i];	
	}
	write_npy_header(fd, shape, shsize);
	write(fd, f, l);
}
void mm(const mat* a, const mat* b, mat* c){
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
				float g = to_float32(a->buff[i * a->N + k]);
				float h = to_float32(b->buff[k * b->N + j]);
				f += g * h;
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
void cs(float* c, float* s, float base, size_t d, size_t t){
	// c,s : output
	// base: frequency of highest frequency turner
	// d: dimension
	// t: position in sequence
	size_t hd = d/2;
	for(size_t i = 0; i < hd; i++){
		float power = - (float) i / (float) hd;
		float freq = powf(base, power);
		float cf = cosf(t * freq);
		float sf = sinf(t * freq);
		c[i] = cf;
		c[i + hd] = cf;
		s[i] = -sf;
		s[i + hd] = sf;
	}
#if PRINT_DEBUG
	char* fname;
   	asprintf(&fname, "cos%zu.npy", t);
	to_fnpy(c, &d, 1, fname);
	free(fname);
#endif
}
void ro(mat* a, size_t hdim){
	// a : (hdim * numh, seqlen)
	// q or k
	float* c = malloc(hdim * sizeof(float));
	float* s = malloc(hdim * sizeof(float));
	size_t numh = a->M/hdim;
	size_t seqlen = a->N;
	for(size_t t = 0; t < seqlen; t++){
		cs(c, s, 500000, hdim, t);
		for(size_t h = 0; h < numh; h++){
			for(size_t i = 0; i < hdim; i++){
				float f = to_float32(a->buff[h * hdim * seqlen + i * seqlen + t]);	
				float g = to_float32(a->buff[h * hdim * seqlen + (i ^ (hdim/2)) * seqlen + t]);
				a->buff[h * hdim * seqlen + i * seqlen + t] = truncate_f32(f * c[i] + g * s[i]);	
			}
		}
	}

}
void sa(const mat* q, const mat* k, const mat* v, const mat* o, mat* ctx){
	//WTF

	//ctx : (d, seqlen)
	//d = 4096
	mat qc, kc, vc;
	size_t seqlen = ctx->N;

	mm(q, ctx, &qc);
	mm(k, ctx, &kc);
	mm(v, ctx, &vc);

#if PRINT_DEBUG
	to_npy(&qc, "qc0.npy");
	to_npy(&kc, "kc0.npy");
#endif
	//qc : 4096 x seqlen => 4 x 8 x 128 x seqlen
	//kc : 1024 x seqlen => 8 x 128 x seqlen
	//vc : 1024 x seqlen => 8 x 128 x seqlen
	ro(&qc, 128);
	ro(&kc, 128);
#if PRINT_DEBUG
	to_npy(&qc, "qcr0.npy");
	to_npy(&kc, "kcr0.npy");
#endif

	//attns: 4 x 8 x seqlen x seqlen
	size_t shape[10];
	shape[0] = 4;
	shape[1] = 8;
	shape[2] = seqlen;
	shape[3] = seqlen;
	float* attns = malloc(32 * seqlen * seqlen * sizeof(float));
	for(size_t g = 0; g < 4; g++){
		for(size_t h = 0; h < 8; h++){
			for(size_t k = 0; k < seqlen; k++){
				for(size_t q = 0; q < seqlen; q++){
					size_t in = 
						  g * 8 * seqlen * seqlen 
						+ h * seqlen * seqlen
				        + k * seqlen 
					    + q;
					attns[in] = 0.0;
					if(k > q){
						attns[in] = -INFINITY;
					   	continue;	
					}
					for(size_t i = 0; i < 128; i++){
						size_t inq = 
							g * 8 * 128 * seqlen 
					   	  + h * 128 * seqlen
						  + i * seqlen
						  + q; 
						size_t ink = 
							h * 128 * seqlen
						  + k * seqlen
						  + i;
						attns[in] += to_float32(qc.buff[inq]) * to_float32(kc.buff[ink]);
					}
					attns[in] /= sqrt(128.0);
				}
			}
		}
	}
#if PRINT_DEBUG
	to_fnpy(attns, shape, 4, "attn0.npy");
#endif
	//softmax
	for(size_t g = 0; g < 4; g++){
		for(size_t h = 0; h < 8; h++){
			for(size_t q = 0; q < seqlen; q++){
				float norm = 0.0;
				for(size_t k = 0; k < seqlen; k++){
					if(k > q) continue;
					size_t in = 
						  g * 8 * seqlen * seqlen 
						+ h * seqlen * seqlen
				        + k * seqlen 
					    + q;
					attns[in] = expf(attns[in]); 
					norm += attns[in];
				}
				for(size_t k = 0; k < seqlen; k++){
					size_t in = 
						  g * 8 * seqlen * seqlen 
						+ h * seqlen * seqlen
				        + k * seqlen 
					    + q;
					if(k > q) attns[in] = 0.0;
					else attns[in] /= norm;
				}
			}
		}
	}
#if PRINT_DEBUG
	to_fnpy(attns, shape, 4, "smax0.npy");
#endif
	//preo: 4 x 8 x 128 x seqlen
	float* preo = malloc(32 * 128 * seqlen * sizeof(float));
	for(size_t g = 0; g < 4; g++){
		for(size_t h = 0; h < 8; h++){
			for(size_t i = 0; i < 128; i++){
				for(size_t q = 0; q < seqlen; q++){
					size_t in = g * 8 * 128 * seqlen //group
							  + h * 128 * seqlen //hnum
							  + i * seqlen //hdim
						      + q; //seqno
					float acc = 0.0;
					for(size_t v = 0; v < seqlen; v++){
						//join on v
						size_t inv = h * 128 * seqlen
							       + i * seqlen
								   + v;
						size_t ina = g * 8 * seqlen * seqlen
						           + h * seqlen * seqlen
								   + v * seqlen
								   + q;
						acc += to_float32(vc.buff[inv]) * attns[ina];
					}
					preo[in] = acc;
				}
			}
		} 
	}
	free(attns);
	for(size_t i = 0; i < 32 * 128; i++){
		for(size_t t = 0; t < seqlen; t++){
			float curr = to_float32(ctx->buff[i * seqlen + t]);
			for(size_t h = 0; h < 32 * 128; h++){
				curr += to_float32(o->buff[i * 32 * 128 + h]) 
				      * preo[h * seqlen + t];
			}
			ctx->buff[i * seqlen + t] = truncate_f32(curr);
		}
	}
	free(preo);
}
float silu(float f){
	return f/(1 + expf(-f));
}
void udg(const mat* gate, const mat* up, const mat* down, mat* ctx){
	//ctx: d x seqlen
	//up, gate: D x d
	//down : d x D
	//D > d
	size_t D = up->M;
	size_t d = up->N;
	size_t seqlen = ctx->N;
	float* inter = malloc(sizeof(float) * D * seqlen);
	//inter: D x seqlen 
	for(size_t i = 0; i < D; i++){
		for(size_t t = 0; t < seqlen; t++){
			float u = 0.0;
			float g = 0.0;
			for(size_t j = 0; j < d; j++){
				u += to_float32(up->buff[i * d + j]) * to_float32(ctx->buff[j * 4 + t]);
				g += to_float32(gate->buff[i * d + j]) * to_float32(ctx->buff[j * 4 + t]);
			}
			inter[i * seqlen + t] = silu(g) * u;
		}
	}
#if PRINT_DEBUG
	size_t sh[2];
	sh[0] = D;
	sh[1] = seqlen;
	to_fnpy(inter, sh, 2, "inter.npy");
#endif
	for(size_t j = 0; j < d; j++){
		for(size_t t = 0; t < seqlen; t++){
			float acc = to_float32(ctx->buff[j * seqlen + t]);
			for(size_t i = 0; i < D; i++){
				float a = to_float32(down->buff[j * D + i]);
				float b = inter[i * seqlen + t]; 
				acc += a * b;
				//if(j == 0 && t == 0) printf("%f %f %f %f\n", a, b, a * b, acc);
			}
			ctx->buff[j * seqlen + t] = truncate_f32(acc);
		}
	}
	free(inter);
#if PRINT_DEBUG
	to_npy(ctx, "after.npy");
#endif
}
static int* offsets = NULL;
static char* tokens = NULL;
static int olength = -1;
static int tlength = -1;
void init_tokenizer(char* opath, char* tpath){
	if(offsets == NULL || tokens == NULL){
		int ofd = open(opath, O_RDONLY);
		int tfd = open(tpath, O_RDONLY);
		struct stat s;	
		fstat(ofd, &s);
		olength = s.st_size; //128256 in llama-3.1-8b-instruct
		fstat(tfd, &s);
		tlength = s.st_size; //840632 in llama-3.1-8b-instruct
		offsets = malloc(olength * sizeof(int));
		read(ofd, offsets, olength * sizeof(int));
		tokens = malloc(tlength);
		read(tfd, tokens, tlength);
		close(ofd);
		close(tfd);
	}
}
char* tokenize(int i){
	init_tokenizer("llama-3.1-8b-instruct-offsets.txt", "llama-3.1-8b-instruct-tokens.txt");
	int start = i == 0 ? 0 : offsets[i - 1];	
	int end = offsets[i];
	int length = end - start;
	char* out = malloc(length + 1);
	memcpy(out, tokens + start, length);
	out[length] = '\0';
	return out;
}
mat* embed(int* s, int seqlen, const char* embs){
	mat* out = malloc(sizeof(mat));
	out->M = 4096; //llama hidden
	out->N = seqlen;
	out->buff = malloc(out->M * out->N * sizeof(bfloat16));
	bfloat16* e = (bfloat16 *) embs;
	for(size_t i = 0; i < out->N; i++){
		for(size_t j = 0; j < out->M; j++){
			out->buff[j * out->N + i] = e[s[i] * out->M + j];
		}
	}
	return out;
}
void rms_norm(mat* m, const mat* weights, float epsi){
	//m : d x seqlen
	float* rms = malloc(m->N * sizeof(float));
	for(size_t i = 0; i < m->N; i++){
		rms[i] = 0.0;
		for(size_t j = 0; j < m->M; j++){
			float f = to_float32(m->buff[j * m->N + i]);
			rms[i] += f * f;
		}
		rms[i] /= (float) m->M;
		rms[i] += epsi;
		rms[i] = sqrt(rms[i]);
	}
	bfloat16* w = weights->buff;
	for(size_t i = 0; i < m->N; i++){
		for(size_t j = 0; j < m->M; j++){
			float f = to_float32(m->buff[j * m->N + i]);
			f *= (to_float32(w[j]) / rms[i]);
			m->buff[j * m->N + i] = truncate_f32(f);
		}
	}
}
const mat* extract_mat(struct json* j, char* base, char* name){
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
	mat* out = malloc(sizeof(mat));
	struct json* fields = (struct json *) curr->data; 
	struct json* shape_array = (struct json *) fields->start->next->data;
	struct json* offsets_array = (struct json *) fields->start->next->next->data;
	out->M = *(size_t*) shape_array->start->data;
	struct node* npossibly = shape_array->start->next;
	out->N = npossibly != NULL ? *(size_t*)npossibly->data : 1;
	out->buff = (bfloat16*) (base + *(size_t*) offsets_array->start->data);
	return out;
	
}
