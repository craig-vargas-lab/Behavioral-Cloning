from math import ceil
def calc(in_dim, conv, pool, repeat):
	print()
	print("Layers")
	print("Start:", in_dim)
	out_dim = in_dim
	for i in range(repeat):
		for j in range(len(out_dim)):
			out_dim[j] = ceil((out_dim[j] - conv[i] + 1)/pool[i])
		print("Layer {0} output -> {1}".format(i+1, out_dim))
	return out_dim


def main():
	factor = 0.4
	# in_shape = [160*factor, 320*factor]

	top_crop_factor = 60/160
	bot_crop_factor = 25/160
	y_crop_factor = 1 - (top_crop_factor + bot_crop_factor) # 0.53125
	in_shape = [160*y_crop_factor, 320]

	# in_shape = [68*factor, 320*factor]

	conv_filters = [5, 5, 5, 3, 3]
	# conv_filters = [3, 3, 3, 2, 2]

	pool_filters = [2, 2, 2, 1, 1]
	
	x = calc(in_shape, conv_filters, pool_filters, len(conv_filters))


if __name__ == '__main__':
	main()
