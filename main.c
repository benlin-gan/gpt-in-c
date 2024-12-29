#include "json.h"
#include "util.h"
#include <stddef.h>
#include <sys/types.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdlib.h>
int main(int arg, char** argv){	
	int fd = open("model-00001-of-00004.safetensors", O_RDONLY);
	if(fd == -1){
		perror("open");
		return 1;
	}
	char* dat = mmap(NULL, 5000000000, PROT_READ, MAP_PRIVATE, fd, 0);
	size_t q = *(size_t*)dat;
	printf("%zu\n", q);
	write(1, dat + 8, q - 5);
	printf("\n%c %d\n", dat[q], dat[q]);
	printf("%c %d\n", dat[q + 1], dat[q + 1]);
	printf("%c %d\n", dat[q + 2], dat[q + 2]);
	printf("%c %d\n", dat[q + 3], dat[q + 3]);
	printf("%c %d\n", dat[q + 4], dat[q + 4]);
	printf("%c %d\n", dat[q + 5], dat[q + 5]);
	printf("%c %d\n", dat[q + 6], dat[q + 6]);
	printf("%c %d\n", dat[q + 7], dat[q + 7]);
	//q + 8 is the first real character
	printf("%c %d\n", dat[q + 8], dat[q + 8]);
	printf("%f\n", to_float32(*(bfloat16*) &dat[q + 8]));
	printf("%f\n", to_float32(*(bfloat16*) &dat[q + 10]));
	printf("%f\n", to_float32(*(bfloat16*) &dat[q + 12]));
	printf("%f\n", to_float32(*(bfloat16*) &dat[q + 14]));
	size_t p = 8;
	struct json* j = jstring_to_json(dat, &p, q + 7);
	print_titles(j);
	mat r;
	r.M = 2;
	r.N = 2;
	r.buff = malloc(4 * sizeof(bfloat16));
	r.buff[0] = truncate_f32(0.0);
	r.buff[1] = truncate_f32(1.0);
	r.buff[2] = truncate_f32(1.0);
	r.buff[3] = truncate_f32(1.0);
	puts("\n\n");
	print_mat(&r);
	mat s;
	s.M = 2;
	s.N = 2;
	s.buff = malloc(4 * sizeof(bfloat16));
	s.buff[0] = truncate_f32(34.0);
	s.buff[1] = truncate_f32(8.0);
	s.buff[2] = truncate_f32(55.0);
	s.buff[3] = truncate_f32(13.0);
	puts("\n\n");
	print_mat(&s);
	mat t;
	mm(&r, &s, &t);
	puts("\n\n");
	print_mat(&t);
	to_npy(&t, "t.npy");

	printf("%s\n", tokenize(128000));
	printf("%s\n", tokenize(2000));
	printf("%s\n", tokenize(279));
	printf("%s\n", tokenize(1274));


	//prompt: "<|begin_of_text|>for the people"
	int prompt[4];
	prompt[0] = 128000;
	prompt[1] = 2000;
	prompt[2] = 279;
	prompt[3] = 1274;
	char* base = &dat[q + 8];	
	mat* u = embed(prompt, 4, base);	
	to_npy(u, "embed.npy");

	const mat* z = extract_mat(j, base, "model.layers.0.input_layernorm.weight");
	rms_norm(u, z, 1e-5); 
	to_npy(u, "norm0.npy");

	const mat* q0 = extract_mat(j, base, "model.layers.0.self_attn.q_proj.weight");
	const mat* k0 = extract_mat(j, base, "model.layers.0.self_attn.k_proj.weight");
	const mat* v0 = extract_mat(j, base, "model.layers.0.self_attn.v_proj.weight");
	const mat* o0 = extract_mat(j, base, "model.layers.0.self_attn.o_proj.weight");
	sa(q0, k0, v0, o0, u);
	to_npy(u, "sa0.npy");

}
