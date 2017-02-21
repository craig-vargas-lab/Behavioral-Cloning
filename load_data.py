import numpy as np
import time

start_time = time.time()

DIRECTORY = '/Users/cvar/selfdrivingcar/term_one/projectthree/data'
DATA_PATH = DIRECTORY + '/train.npz'

data = np.load(DATA_PATH)
print("load finished in: {:.2f} secs".format(time.time() - start_time))
print("date is of type:", type(data))
print()


x = data['images']
y = data['steering']

print("Shapes (X,Y):", x.shape, ", ", y.shape)
print()

end_time = time.time()
print("Elapsed time:", end_time - start_time)