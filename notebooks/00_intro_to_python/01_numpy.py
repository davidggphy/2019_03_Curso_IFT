# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.0.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NumPy
#
# NumPy is the fundamental package for scientific computing with Python. It contains among other things:
#
# * a powerful N-dimensional array object
# * sophisticated (broadcasting) functions
# * tools for integrating C/C++ and Fortran code
# * useful capabilities from linear algebra, statistics, calculus, random number, etc.
# * operates efficiently using vectorized implementations of the algorithms
#
# Besides its obvious scientific uses, NumPy can also be used as an efficient multi-dimensional container of generic data. Arbitrary data-types can be defined. This allows NumPy to seamlessly and speedily integrate with a wide variety of databases.

# %%

# %% [markdown]
# We start loading the package, using the alias `np`. In order to do that, `numpy` must be properly installed.

# %%
import numpy as np

# %% [markdown]
# ## Arrays

# %% [markdown]
# Arrays are the main object in NumPy, and they consist in **homogeneous multidimensional arrays**, or tensors. It consists in a collection of elements (usually numbers), all of the same type (homogeneous) and indexed by a tuple of positive integers. 
#
# Dimensions are called **axes**, and the number of axes is the **rank**.

# %% [markdown]
# For example, let us consider the coordinates of a point in the space $\mathbb{R}^3$, it consists in an array of rank 1 (only one axis), with length 3.
#
# `[1.,3.,2.]`
#
# We can also consider a matrix of integer values, or rank 2 array, with 2 rows and 3 columns.
#
# `[[ 1, 0, 0],
#  [ 0, 1, 2]]`
#  
#  The element in the position `(1,2)` is 2. **Indices start at 0.**
#

# %% [markdown]
# ## Creating arrays

# %% [markdown]
# The simplest way of creating an `array` is to convert a Python's list into an array, the type of the elements will be deduced automatically.
#
# This is done with the method `np.array()`.

# %%
mylist = [1, 2, 3]
x = np.array(mylist)

# %%
x

# %% [markdown]
# Each array has two attributes:
# * `dtype`: the type of the elements. It is usually int or float, followed by the precision.
# * `shape`: the dimensions of the array.

# %%
x.dtype

# %%
x.shape

# %%
x = np.array([1, 2, 3.5])
x

# %%
x.dtype

# %% [markdown]
# <br>
# To create  higher rank arrays, you can transform a list of lists (of lists of ...) into an `array`.

# %%
m = np.array([[7, 8, 9], [10, 11, 12]])
m

# %%
m.shape

# %% [markdown]
# <br>
# Arrays can be created using predefined constructors, without having to state all elements. Let us explore some options.

# %% [markdown]
# `arange` returns evenly spaced values within a given interval, indicating the step. As with `range`, it spans the **semi-open** set [start, stop).

# %%
x = np.arange(0, 30, 2.3) # de 0 a 30 contando de 2.3 en 2.3
x

# %% [markdown]
# `linspace` returns evenly spaced numbers over a specified interval, indicating the number of points. By default it **includes the endpoint**.

# %%
x = np.linspace(0, 30, 9) # 9 números equiespaciados en [0, 30]
x

# %% [markdown]
# `zeros` y `ones` create arrays with all its elements 0 or 1 respectively.

# %%
np.zeros([4,4])

# %%
 np.ones(11,dtype=bool)

# %%
3*np.ones([2,2,3])

# %% [markdown]
# `eye` created an identity matrix, and `diag` and diagonal matrix.

# %%
np.eye(3)

# %%
np.diag([6,8,9])

# %% [markdown]
# ## Combining  arrays

# %%
x = np.ones([2, 3])
x

# %% [markdown]
# `vstack` stacks arrays in sequence vertically (row wise), it is the same as a concatenation along the first axis (index 0).

# %%
y = np.vstack([x, 2*x])
print('shape = ',y.shape)
y

# %% [markdown]
# `hstack` stacks arrays in sequence horizontally (column wise), it is the same as a concatenation along the second axis (index 1).

# %%
y = np.hstack([x, 2*x, 3*x])
print('shape = ',y.shape)
y

# %% [markdown]
# ## Operations with arrays

# %% [markdown]
# `reshape` gives a new shape to an array without changing its data.
# In the example you can see that it starts filling along the last axis (in this case, the one of length 5).

# %%
x = np.arange(0, 15)
x = np.reshape(x,[3,-1]) # You can also use x.reshape([3,5])
x

# %% [markdown]
# Given an array, we can read and modify each element.

# %%
x[0,1] = 100*x[0,1]
x

# %% [markdown]
# We can specify the order of the filling, it can be C like or F (Fortran) like.

# %%
x = np.arange(0, 15)
x = np.reshape(x,[3,5],order='F') # You can also use x.reshape([3,5],order='F')
x

# %% [markdown]
# You can easily transpose an array with the attribute `.T` or with the method `transpose()`.

# %%
print('Original\n',x,'\nTransposed\n',x.T)

# %% [markdown]
#
# **Warning:** transposing is not the same as reshaping alternating the dimensions.

# %%
x = np.arange(0,15).reshape([3,5]).T
y = np.arange(0, 15).reshape([5,3])
print(x)
print(y)
print(x==y)

# %% [markdown]
# You can also transpose arrays with rank higher than 2.

# %%
x = np.arange(0,24).reshape([2,3,4])
print(x)

# %%
print('Transposed shape = ',x.T.shape, '\nTransposed array')
print(x.T)

# %% [markdown]
# You can use Python's mathematical operators to perform **element-wise** operations on arrays.
#
# `+`, `-`, `*`, `/` , `//` , `%` and `**` work as expected.

# %%
x = np.arange(1,5)
y = x + x
print(x,'+',x,'=',y)

# %%
x = np.arange(1,5)
y = x - x
print(x,'-',x,'=',y)

# %%
x = np.arange(1,5)
y = x * x
print(x,'*',x,'=',y)

# %%
x = np.arange(1,5)
y = x / x
print(x,'/',x,'=',y)

# %%
x = np.arange(1,5)
y = np.arange(11,15)
z = y // x
print(y,'//',x,'=',z)

# %%
x = np.arange(1,5)
y = np.arange(11,15)
z = y % x
print(y,'%',x,'=',z)

# %%
x = np.arange(1,5)
y = x**2
print(x,'**2','=',y)

# %% [markdown]
# `dot` performs scalar product in the case of 1-D arrays and matrix multiplication for 2-D arrays.

# %%
x = np.arange(16).reshape(4,4)
y = np.eye(4) # identity matrix
z = np.dot(x,y) # Should be the same as x
print(x == z) # They coincide

# %%
x = np.arange(16).reshape(4,4)
y = np.eye(4) # identity matrix
z = x * y # This is different, since it is element-wise multiplication
print(x == z) # They do not coincide

# %% [markdown]
# ## Iterating over arrays

# %%
## We can generate arrays following a probability distribution with np.random
matrix = np.random.uniform(0, 1, (4,3)) 
print(matrix)

# %% [markdown]
# Iterate over indices

# %%
for i in range(len(matrix)):
    print(matrix[i])

# %% [markdown]
# Iterate over rows

# %%
for row in matrix:
    print(row)

# %% [markdown]
# Iterate over indices and rows

# %%
for i, row in enumerate(matrix):
    print('row', i, 'is', row)

# %% [markdown]
# You can nest the iterators

# %%
for row in matrix:
    for el in row:
        print(el)

# %%
[print(el)  for row in matrix for el in row]

# %% [markdown]
# ## Mathematical functions

# %% [markdown]
# NumPy has implemented many of the usual mathematical functions, and they are applied element-wise to an array. Some of them are `sin`, `cos`, `exp`, `sqrt`, `log`. Also, you can access the value of $\pi$ with `np.pi`.
#
# You can find a complete list in 
#
# https://docs.scipy.org/doc/numpy/reference/routines.math.html

# %% [markdown]
# ### Element-wise mathematical function

# %%
np.sin(np.linspace(0,np.pi,5))**2

# %%
x = np.linspace(1,10,5)
print('x =',x)
xlog = np.log(x)
print('log(x) =',xlog)
xexp = np.exp(x)
print('exp(x) =',xexp)


# %% [markdown]
# ### <span style="color:red">**Exercise**</span> 

# %% [markdown]
# Create a function that obtains the cosine of the angle between two vectors.
#
# $$ \cos \theta = \frac{\langle v_1, v_2 \rangle}{||v_1|| ||v_2||} $$
#
# where $|| v ||$ is the norm of v, which can be expressed as $\sqrt{\langle v, v \rangle}$

# %% [markdown]
# ### <span style="color:red">**Exercise**</span> 

# %% [markdown]
# Using two nested for loops, implement a function to calculate the mean value of the elements of a matrix.

# %% [markdown] {"heading_collapsed": true}
# ### Mathematical functions on axes of the array

# %% [markdown] {"hidden": true}
# NumPy has plenty of functions to operate on `arrays`. To calculate the sum of elements `sum`, or their product `prod`. Also statistical ones like the mean `mean` and the standard deviation `std`.
#
# You can usually specify along which `axis` to perform the operation with the `axis` option, if it is not specified, the array is flattened.
#
# In general, with the method `np.apply_along_axis(function, axis, array)` we can apply an arbitrary function slicing along a given `axis`.

# %% {"hidden": true}
def arggmax(x):
    return np.argmax(x)
axis = 0
matrix = np.array([[1,3,5,7],[0,2,4,8]])
print(matrix)
print('argmax over axis',axis)
print(np.apply_along_axis(arggmax, axis, matrix))
print(matrix.argmax(axis = axis))

# %% {"hidden": true}
vector = np.array([-4, -2, 1, 3, 5])
matrix = np.array([[1,3,5,7],[0,2,4,8]])
print('vector')
print(vector)
print('matrix')
print(matrix)

# %% {"hidden": true}
print('sum')
print('vector =', vector.sum())
print('matrix =', matrix.sum())
print('matrix (over rows)=', matrix.sum(axis=0))
print('matrix (over columns)=', matrix.sum(axis=1))

# %% {"hidden": true}
print('product')
print('vector =', vector.prod())
print('matrix =', matrix.prod())
print('matrix (over rows)=', matrix.prod(axis=0))
print('matrix (over columns)=', matrix.prod(axis=1))

# %% {"hidden": true}
print('max')
print('vector =', vector.max())
print('matrix =', matrix.max())
print('matrix (over rows)=', matrix.max(axis=0))
print('matrix (over columns)=', matrix.max(axis=1))

# %% {"hidden": true}
print('min')
print('vector =', vector.min())
print('matrix =', matrix.min())
print('matrix (over rows)=', matrix.min(axis=0))
print('matrix (over columns)=', matrix.min(axis=1))

# %% {"hidden": true}
print('mean')
print('vector =', vector.mean())
print('matrix =', matrix.mean())
print('matrix (over rows)=', matrix.mean(axis=0))
print('matrix (over columns)=', matrix.mean(axis=1))

# %% {"hidden": true}
print('standard deviation')
print('vector =', vector.std())
print('matrix =', matrix.std())
print('matrix (over rows)=', matrix.std(axis=0))
print('matrix (over columns)=', matrix.std(axis=1))

# %% {"hidden": true}
print('argmax')
print('vector =', vector.argmax())
print('matrix =', matrix.argmax())
print('matrix (over rows)=', matrix.argmax(axis=0))
print('matrix (over columns)=', matrix.argmax(axis=1))

# %% {"hidden": true}
print('argmin')
print('vector =', vector.argmin())
print('matrix =', matrix.argmin())
print('matrix (over rows)=', matrix.argmin(axis=0))
print('matrix (over columns)=', matrix.argmin(axis=1))

# %% [markdown]
# ### <span style="color:red">**Exercise**</span> 

# %% [markdown]
# Given a matrix M, built a vector wich i-th element is the product of the sum of the elements of the i-th row of M times the product of the elements of the i-th row of M. 
#
# Can you do it with one line of code? **Hint:** Use lambda functions.

# %%
M = np.array([[1,2,3],[3,4,5],[5,6,7]])
print(M)

# %% [markdown]
# ## Indexing/Slicing

# %% [markdown]
# Similarly to lists, you can index arrays using square brackets `[]`.
#
# You can also sliced over with expressions like `[start:stop:step]`.

# %%
vector = np.arange(20)
print(vector)
vector[0], vector[4], vector[-1]

# %%
vector[0:5:2]

# %% [markdown]
# In the case the rank is larger than 1

# %%
matrix = np.arange(34)
matrix.resize((6, 6))
matrix

# %%
matrix[2, 2]

# %%
matrix[1, 3:6]

# %% [markdown]
# We can extract all the rows of `matrix` up to the 2 (not including it) and all columns up to the last one (not including it).

# %%
matrix[:2, :-1]

# %% [markdown]
# Selecting only one row

# %%
matrix[-1, :]

# %% [markdown]
# Selecting only one column 

# %%
matrix[:, 0]

# %% [markdown]
# **Conditional indexing** can also be applied!

# %%
matrix[matrix % 2 == 0]

# %%
matrix[matrix % 2 ==0] # This selects only the even elements in matrix

# %% [markdown]
# ### <span style="color:red">**Exercise**</span> 

# %% [markdown]
# First, select the first column of the given matrix `A`, and save it in the variable `v`.
# Substitute the elements in `v` that are null for $7$.
#
# Calculate the matrix product `A` $\cdot$ `v`. 
#
# Result = `[43,  0, 98]`
#
# **Warning:** Be careful with the pointers.

# %%
A = np.array([[1,2,4],[0,0,0],[0,6,8]])

# %% [markdown]
# ## Vectorization and broadcasting

# %% [markdown]
# If it is possible, explicit `for` loops should be avoided in Python, since it is faster to use vectorized functions (i.e., functions that apply directly to vectors/tensors).
#
# Let us show that with an example. We will calculate the mean value of a list of numbers satisfying a condition.

# %%
vector = np.random.randn(1000000) # 1 million numbers sampled from a normal distribution

# %%
# %%timeit -n1
s = 0
# We use a for loop to access the elements
for n in vector:  
    if n>0:
        s += n
mean = s/len(vector)
mean

# %%
# %%timeit -n1
# We apply the mean method to the vector directly, using the mask with the condition.
mean = vector[vector>0].mean()  
mean

# %% [markdown]
# **Broadcasting** allows universal functions to deal in a meaningful way with inputs that do not have exactly the same shape.
#
# The first rule of broadcasting is that if all input arrays **do not have the same number of dimensions**, a “1” will be repeatedly **prepended** to the shapes of the smaller arrays until all the arrays have the same number of dimensions.
#
# The second rule of broadcasting ensures that **arrays with a size of 1 along a particular dimension act as if they had the size of the array with the largest shape along that dimension**. The value of the array element is assumed to be the same along that dimension for the “broadcast” array.
#
# After application of the broadcasting rules, the sizes of all arrays must match. More details can be found in
#
# https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

# %% [markdown]
# The simplest broadcasting example occurs when an array and a scalar value are combined in an operation.

# %%
scalar = 3.
matrix = np.linspace(0,15,16).reshape([4,4])
matrix / scalar

# %% [markdown]
# A slightly more complicated example occurs when combining a matrix and a vector.

# %%
matrix = np.linspace(0,9,10).reshape([2,-1])
# We need to transform the vector into a "column" vector
vector = np.array([0,2]).reshape([2,1])
# vector = np.array([0,2])[:,np.newaxis]
print('matrix\n',matrix)
print('vector\n',vector)
print('matrix * vector\n',vector + matrix)

# %%
matrix.shape

# %%
vector.shape

# %% [markdown]
# We can broadcast in the other axis, just using a vector with length equal to the first axis (columns).

# %%
matrix = np.linspace(0,9,10).reshape([2,-1])
# We need to transform the vector into a "column" vector
vector = np.array([0,1,2,3,4])
# vector = np.array([0,2])[:,np.newaxis]
print('matrix\n',matrix)
print('vector\n',vector)
print('matrix * vector\n',vector * matrix)

# %% [markdown]
# ### <span style="color:red">**Exercise**</span> 

# %% [markdown]
# Using two given vectors `v1` and `v2`. Build a matrix `A` with elements `A[i][j] = v1[i]**v2[j]`.
# Try to code it in one line using broadcasting.

# %%
v1 = np.array([10,20,30,40,50])
v2 = np.array([0,1,2,3])

# %% [markdown]
# ### <span style="color:red">**Exercise**</span> 

# %% [markdown]
# Given an array `matrix` having the sampling of 5 heads or tails (head = 1, tails = 0) experiments of 100 trials each. Our goal is to determine if the coin is fair.
# Create two vectors `means` and `sigmas_mean`, containing the mean values for each experiment and the standard deviation of the mean $\sigma_{\text{mean}} = \frac{\sigma}{\sqrt{n}}$.
#
# Print, for each experiment, the $95\%$ confidence interval of the mean with the following string:
#
# 'Experiment `i` . CI (95%) = [ `mean - 1.96 sigma_mean` , `mean + 1.96 sigma_mean` ]' ,
#
# being `i` the number of the experiment, `mean` the mean of the experiment and `sigma_mean` the standard deviation of the mean of the experiment.
#
# Finally, observe that the CI shrinks when increasing the number of trials per experiment.

# %%
matrix = np.random.randint(0,2, (5,100))

# %% [markdown]
# ## Extra: EinSum

# %% [markdown]
# Using the einsum function, we can specify operations on NumPy arrays using the Einstein summation convention.
#
# It can often outperform familiar array functions in terms of speed and memory efficiency, thanks to its expressive power and smart loops. 

# %% [markdown]
# We can start practicing with a matrix. Let us get the trace, the diagonal and the total sum.

# %%
A = np.array(np.arange(16)).reshape([4,4])
print('A\n-------\n',A)
print('\nTrace\n-------')
print(np.einsum('ii -> ',A))
print('\nDiagonal\n-------')
print(np.einsum('ii -> i',A))
print('\nSum of all elements\n-------')
print(np.einsum('ij -> ',A))

# %% [markdown]
# We can easily do matrix multiplication in this setting.

# %%
A = np.array(np.arange(15)).reshape([5,3])
print('A\n-------\n',A)
B = np.array(np.arange(12)).reshape([3,4])
print('B\n-------\n',B)
print('A x B\n-------')
print(np.einsum('ij,jl -> il',A,B))

# %% [markdown]
# Compute the scalar product or build a matrix from the outer product of two vectors.

# %%
v1 = 10*  np.array([1,2,3,4,5])
print('v1\n-------\n',v1)
v2 = np.array([0,1,2,3,4])
print('v2\n-------\n',v2)
print('Scalar product\n------')
print( np.einsum('i,i ->',v1,v2)  )
print('Outer product\n------')
print( np.einsum('i,j -> i j',v1,v2)  )
print('Outer product (transposed) \n------')
print( np.einsum('i,j -> j i',v1,v2)  )

# %% [markdown]
# Let us go to another example, involving rank 1 and rank 2 tensors.
#
# As we explained before, we can multiply each row of `B` with each value of `A` using broadcasting, but we can do the same contracting indices. Also, we can keep track of the execution time of the calculation.

# %%
A = np.array([0, 1, 2])
print('A\n-------\n',A)
B = np.array(np.arange(12)).reshape([3,4])
print('B\n-------\n',B)

# %%
# %timeit A.reshape((-1,1))*B
print(A.reshape((-1,1))*B)

# %%
# %timeit np.einsum('i,ij->ij',A,B)
print(np.einsum('i,ij->ij',A,B))

# %% [markdown]
# Now, after that operation, we can sum along the first axis (keeping 3 rows).

# %%
# %timeit np.sum(A.reshape((-1,1))*B,axis=1)
print(np.sum(A.reshape((-1,1))*B,axis=1))

# %%
# %timeit np.einsum('i,ij->i',A,B)
print(np.einsum('i,ij->i',A,B))

# %% [markdown]
# Or even sum all the values of the resulting matrix

# %%
# %timeit np.sum(A.reshape((-1,1))*B)
print(np.sum(A.reshape((-1,1))*B))

# %%
# %timeit np.einsum('i,ij->',A,B)
print(np.einsum('i,ij->',A,B))

# %% [markdown]
# Let us now create a tensor of rank 3, of shape (6,5,5) and calculate the trace of the six 5x5 matrices.

# %%
C = np.array(range(150)).reshape((6,5,5))

# %%
# %timeit  np.einsum('i j j  -> i ',C)
print(np.einsum('i j j  -> i ',C))

# %%
# %timeit  np.trace(C,axis1=1,axis2=2)
print(np.trace(C,axis1=1,axis2=2))

# %% [markdown]
# ### <span style="color:red">**Exercise**</span> 

# %% [markdown]
# Consider the large vector v defined below, and calculate v to the third power using three different methods. Compare the execution time of them.

# %%
v = np.random.normal(size = int(10e7))

# %%

# %%
A = np.array(np.arange(15)).reshape([5,3])
A

# %%
np.prod(A,axis=1)

# %%
