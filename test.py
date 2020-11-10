
import numpy as np

a = np.array([1,2,3,4,5,6]).reshape(2,3)
b = np.array([7,8,9,10,11,12]).reshape(2,3)
print(a)
print(b)
c = a[np.newaxis, :, :]
d = b[:, np.newaxis, :]
e = a.reshape(-1, 2,3)
f = b.reshape(2,-1, 3)
print('--------')
# print(e-f)
# print(c)
# print(d)
# print(e,f)
# print(c-d)
res = np.linalg.norm(c-d, axis=-1)
print(res)
arr = np.array([3,6,5,1,7,4])
s = np.argsort(arr)
print(s)

array = np.array([0,1,1,2,2,2,3,8,8,8,8,5,6, 5])
print(np.bincount(array))
print(np.argmax(np.bincount(array)))
print(np.exp(array.reshape(2,-1)))

def gaussian_kernel(X, Y=None, sigma=5.0):
    Y = X if Y is None else Y
    assert X.shape[1] == Y.shape[1]
    x_norm = np.expand_dims(X.dot(X.T).diagonal(), axis=-1)
    y_norm = np.expand_dims(Y.dot(Y.T).diagonal(), axis=-1)

    x_norm_mat = x_norm.dot(np.ones((Y.shape[0], 1), dtype=np.float64).T)
    y_norm_mat = np.ones((X.shape[0], 1), dtype=np.float64).dot(y_norm.T)
    k = x_norm_mat + y_norm_mat - 2 * X.dot(Y.T)
    k /= - 2 * sigma ** 2
    return np.exp(k)