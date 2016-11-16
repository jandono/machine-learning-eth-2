import numpy as np
import theano
import theano.tensor as T

X = T.fmatrix('inputs')
print(type(X))
Y = T.scalar('outputs')

SUM = X.sum()
Y = SUM

f = theano.function([X],[Y])

inp = np.random.rand(10,5).astype(np.float32)
print(inp.shape)

result = f(inp)
print (type(result))
print result
