#include "json.h"
#include <stddef.h>
#include <sys/types.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
int main(int arg, char** argv){	
	int fd = open("model-00002-of-00004.safetensors", O_RDONLY);
	if(fd == -1){
		perror("open");
		return 1;
	}
	char* dat = mmap(NULL, 10000, PROT_READ, MAP_PRIVATE, fd, 0);
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
	size_t p = 8;
	struct json* j = jstring_to_json(dat, &p, q + 7);
	print_titles(j);

}
