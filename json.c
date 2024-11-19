#include <sys/mman.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include "json.h"

void json_append(struct json* j, struct node* n){
	if(j->start == NULL){
		j->start = n;
		j->end = n;
		return;
	}
	j->end->next = n;
	j->end = n;
}
ssize_t index_of_next(char* str, char c, size_t curr, size_t end){
	size_t i;
	for(i = curr; i < end; i++){
		if(str[i] == c){
			return i;
		}
	}
	return -1;
}

size_t jstring_to_int(char* str, size_t* start, size_t end){
	size_t out = 0;
	for(size_t i = *start; i < end; i++){
		size_t digit = str[i] - '0';
		if(digit > 9){
			*start = i;
			return out;
		}else{
			out *= 10;
			out += digit;
		}
	}
	fprintf(stderr, "Error: no more characters, was parsing integer\n");
	exit(1);
}
char* jstring_to_string(char* str, size_t* start, size_t end){
	//"<str>" -> <str>
	ssize_t endq = index_of_next(str, '"', *start + 1, end);
	if(endq == -1){
		fprintf(stderr, "Error: no more characters, was parsing string\n");
		exit(1);
	}
	char* out = malloc(endq - *start);	
	//length: end - start - 2 + null term
	strncpy(out, str + *start + 1, endq - *start - 1);
	out[endq - *start - 1] = 0;
	*start = endq + 1;
	return out;
}
struct node* jstring_to_node(char* str, size_t* start, size_t end){
	struct node* out = malloc(sizeof(struct node));
	out->title = jstring_to_string(str, start, end);
	if(str[*start] != ':'){
		fprintf(stderr, "Error: key not followed by value\n");
		exit(1);
	}
	*start = *start + 1;
	out->data = jstring_to_unknown(str, start, end, &out->type);
	out->next = NULL;
	return out;
}
struct node* jstring_to_anode(char* str, size_t* start, size_t end){
	struct node* out = malloc(sizeof(struct node));
	out->title = NULL; //anonymous node (i.e. array element)
	out->data = jstring_to_unknown(str, start, end, &out->type);
	out->next = NULL;
	return out;
}
struct json* jstring_to_array(char* str, size_t* start, size_t end){
	if(str[*start] != '['){
		fprintf(stderr, "Error: not an array\n");
		exit(1);
	}
	struct json* out = malloc(sizeof(struct json));
	(*start)++;
	while (true){
		struct node* n = jstring_to_anode(str, start, end);
		json_append(out, n);
		if(str[*start] == ','){
			(*start)++;
		}else if(str[(*start)++] == ']'){
			break;
		}else{
			fprintf(stderr, "Error: not an array\n");
			exit(1);
		}
	}
	return out;
		
}
void* jstring_to_unknown(char* str, size_t* start, size_t end, enum payload_type* t){
	if(*start == end){
		fprintf(stderr, "Error: no more characters, was parsing field\n");
		exit(1);
		
	}
	if(str[*start] == '"'){
		*t = STRING;
		return jstring_to_string(str, start, end);
	}else if(str[*start] >= '0' && str[*start] <= '9'){
		*t = INT;
		size_t i = jstring_to_int(str, start, end);
		size_t* ii = malloc(sizeof(size_t));
		*ii = i;
		return ii;
	}else if(str[*start] == '{'){
		*t = JSON;
		return jstring_to_json(str, start, end);
	}else if(str[*start] == '['){
		*t = ARRAY;
		return jstring_to_array(str, start, end);
	}else{
		fprintf(stderr, "Error: unparsable field\n");
	}
	return NULL;
}
struct json* jstring_to_json(char* str, size_t* start, size_t end){
	if(str[*start] != '{'){
		fprintf(stderr, "Error: not an object\n");
		exit(1);
	}
	struct json* out = malloc(sizeof(struct json));
	(*start)++;
	while (true){
		struct node* n = jstring_to_node(str, start, end);
		json_append(out, n);
		if(str[*start] == ','){
			(*start)++;
		}else if(str[(*start)++] == '}'){
			break;
		}else{
			fprintf(stderr, "Error: not an object\n");
			exit(1);
		}
	}
	return out;
}
void print_titles(struct json* j){
	struct node* n = j->start;
	while(n != NULL){
		printf("%s\n", n->title);
		n = n->next;
	}
}
