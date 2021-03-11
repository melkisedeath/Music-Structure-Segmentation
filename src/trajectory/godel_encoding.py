import math


def bij_encoding(int_list, max_int):
	out = None
	for i in int_list:
		out += hex(i)
	return out

