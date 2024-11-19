#pragma once
#include <sys/types.h>
struct sstring{
	char* buff;
	size_t len;
	size_t cap;
};
enum payload_type{
	JSON,
	ARRAY,
	STRING,
	INT,
};
struct node{
	char* title;
	void* data;
	enum payload_type type;
	struct node* next;
};
struct json{
	struct node* start;
	struct node* end;
};
size_t jstring_to_int(char* str, size_t* start, size_t end);
char* jstring_to_string(char* str, size_t* start, size_t end);
struct json* jstring_to_json(char* str, size_t* start, size_t end);
struct node* jstring_to_node(char* str, size_t* start, size_t end);
struct node* jstring_to_anode(char* str, size_t* start, size_t end);
struct json* jstring_to_array(char* str, size_t* start, size_t end);
void* jstring_to_unknown(char* str, size_t* start, size_t end, enum payload_type* t);
void print_titles(struct json* j);
