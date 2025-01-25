#pragma once
struct grid{
	size_t shape[8];
	size_t sh; //length of shape
	size_t vind; //number of virtual indices
	float* buff;	
}
typedef struct grid grid;
