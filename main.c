#include "json.h"
#include "gpt2.h"
#include <stddef.h>
#include <sys/types.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdlib.h>
int main(int arg, char** argv){	
	gpt2* gpt = load_model("gpt2.safetensors");
	//prompt: "I am"
	int prompt[10];
	prompt[0] = 40;
	prompt[1] = 716;
	print_tok(gpt, 40);
	print_tok(gpt, 716);
	loopgen(gpt, prompt, 2);
	destroy_model(gpt);
	/*
	model* m = init_model();

	//prompt: "<|begin_of_text|>for the people"
	printf("%s", tokenize(128000));
	printf("%s", tokenize(2000));
	printf("%s", tokenize(279));
	printf("%s\n", tokenize(1274));
	int prompt[4];
	prompt[0] = 128000;
	prompt[1] = 2000;
	prompt[2] = 279;
	prompt[3] = 1274;


	mat* u = embed(prompt, 4, m->p1);	
	to_npy(u, "embed.npy");
	printf("embed\n");

	char name[1024];
	for(int l = 0; l < 1; l++){
		sprintf(name, "model.layers.%d.input_layernorm.weight", l);
		const mat* zi = get_mat(m, name);
		rms_norm(u, zi, 1e-5); 
		//to_npy(u, "norm0.npy");
		printf("norm%d\n", l);

		sprintf(name, "model.layers.%d.self_attn.q_proj.weight", l);
		const mat* q = get_mat(m, name);

		sprintf(name, "model.layers.%d.self_attn.k_proj.weight", l);
		const mat* k = get_mat(m, name);

		sprintf(name, "model.layers.%d.self_attn.v_proj.weight", l);
		const mat* v = get_mat(m, name);

		sprintf(name, "model.layers.%d.self_attn.o_proj.weight", l);
		const mat* o = get_mat(m, name);

		sa(q, k, v, o, u);
		//to_npy(u, "sa0.npy");
		printf("sa%d\n", l);

		sprintf(name, "model.layers.%d.post_attention_layernorm.weight", l);
		const mat* zp = get_mat(m, name);
		rms_norm(u, zp, 1e-5);
		//to_npy(u, "normp0.npy");
		printf("normp%d\n", l);

		sprintf(name, "model.layers.%d.mlp.down_proj.weight", l);
		const mat* down = get_mat(m, name);

		sprintf(name, "model.layers.%d.mlp.gate_proj.weight", l);
		const mat* gate = get_mat(m, name);

		sprintf(name, "model.layers.%d.mlp.up_proj.weight", l);
		const mat* up = get_mat(m, name);

		udg(gate, up, down, u);
		sprintf(name, "layer%d.npy", l);
		to_npy(u, name);
		printf("layer%d\n", l);
		//decode(m, u);
	}
	const mat* norm = get_mat(m, "model.norm.weight");
	rms_norm(u, norm, 1e-5);
	to_npy(u, "final_states.npy");

	const mat* head = get_mat(m, "lm_head.weight");
	mat logits;
	mm(head, u, &logits);
	to_npy(&logits, "logits.npy");
	*/
}
