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
# # Python fundamentals

# %% [markdown]
# ## Random things

# %% [markdown]
# Python has a large number of built-in functions, you can check them in
#
# https://www.programiz.com/python-programming/methods/built-in

# %% [markdown]
# You can autocomplete when typing using TAB.
# You can access to the help of a function pressing shift + TAB (one, two or three times). You can also use the `help` method. 

# %%
help(len)

# %% [markdown]
# The last instruction in the cell is printed automatically.
# You can use bash commands within Jupyter.

# %%
! ls

# %%
a = ! pwd
a = a[0]
a

# %%
! which pip

# %% [markdown]
# You can even install packages while you are in the notebook.

# %%
! pip install matplotlib

# %% [markdown] {"heading_collapsed": true}
# ## Variables
# `print`,`type`,`del`

# %% [markdown] {"hidden": true}
# A variable is a named place in the memory where data is stored, and you can access to it later using the variable name.
#
# Python does not require to explicitly declare the variable's type, this is done automatically when assigning a value to it.
# You assign a value to a variable doing:

# %% {"hidden": true}
a = 3 # Integer
b = a + 3. # Float
c = 'I am a string' # String
d = "I am a string" # The same string, you can use single or double quotation marks
e = c == d # Boolean, to check that the two strings are the same
print('a=',a,', b=',b,', c=',c,', e=',e)
a = 10 # Redefinition of a DOES NOT change b
print('a=',a,', b=',b,', c=',c,', e=',e)

# %% [markdown] {"hidden": true}
# You can check the type

# %% {"hidden": true}
print(type(a))
print(type(b))
print(type(c))
print(type(d))

# %% [markdown] {"hidden": true}
# The instruction `del` allows to free variables from memory

# %% {"hidden": true}
del(a)
a

# %% [markdown] {"heading_collapsed": true}
# ## Data Type Conversion
# `str`,`float`,`int`,`bool`,`tuple`,`set`,`list`

# %% [markdown] {"hidden": true}
# Python defines type conversion functions to directly convert one data type to another which is useful in day to day programming.

# %% {"hidden": true}
int3 = 3 # We define an integer
print('This is an int = ',int3)
str3 = str(int3) # Trasnform it into a string
print('The type of the variable str3 is = ',type(str3))
print('This is an str = ',str3)
float3 = float(str3) # Transform the string into a float
print('This is an float = ',float3)
strfloat3 = str(float3)
print('This is an str = ',strfloat3)

# %% [markdown] {"hidden": true}
# Boolean variables are converted into `0`, `1`. In the opposite direction, `0` and `0.0` are converted into `False`, the rest into `True`.

# %% {"hidden": true}
print(int(True))
print(int(False))
print(float(True))
print(float(False))
print(bool(0))
print(bool(0.0))
print(bool(1))
print(bool(9999))
print(bool(0.5))
print(bool(-0.5))
print((True + True)**2)

# %% {"hidden": true}
l1 = ['a','b',1,2,3,1]
print('This is a list = ',l1)
t1 = tuple(l1)
print('This is a tuple = ',t1)
s1 = set(t1)
print('This is a set = ',s1)
l2 = list(s1)
print('This is a list = ',l2)

# %% [markdown] {"heading_collapsed": true}
# ## Variable types

# %% [markdown] {"hidden": true}
# * Booleans: `True` and `False`
# * Numbers
# * Lists
# * Strings
# * Tuples
# * Dictionaries

# %% [markdown] {"heading_collapsed": true, "hidden": true}
# ### Numbers
# `complex`

# %% [markdown] {"hidden": true}
# 1) The integer numbers (e.g. 2, 4, 20) have type `int`.
#
# To do floor division and get an integer result (discarding any fractional result) you can use the `//` operator; to calculate the remainder you can use `%`:

# %% {"hidden": true}
a = 1
b = 2
print(type(a))
print(10//3) # Integer division
print(10%3) # Modulus - remainder of the division of left operand by the right

# %% [markdown] {"hidden": true}
# 2) The ones with a fractional part (e.g. 5.0, 1.6) have type `float`.
#
# Division (/) always returns a float.

# %% {"hidden": true}
c = 0.5
print('a = ',a,', ',type(a),'. b = ',b,', ',type(b))
print('c = ',c,', ',type(c))
print('a/b =', a/b,', ',type(a/b))
print('b/a =', b/a,', ',type(b/a))

# %% [markdown] {"hidden": true}
# 3) You can also define complex numbers with the type `complex`

# %% {"hidden": true}
d = complex(a,b)
print(type(d),d)

# %% [markdown] {"heading_collapsed": true, "hidden": true}
# ### Lists
# `len`,`copy`,`append`, `insert`,  `remove`, `pop`,  `reverse`, `sort`, `index`

# %% [markdown] {"hidden": true}
# Lists are the most versatile of Python's compound data types. A list contains items separated by commas and enclosed within square brackets (`[]`). 
#
# List can be **modified** and its size can be changed. Items belonging to a list can be of **different data type**, although it is common that they share the same type.
#
# The values stored in a list can be accessed using the slice operator (`[ ]` and `[:]`) with **indexes starting at 0** in the beginning of the list and working their way to end -1. 
#
# The plus `+` sign is the list **concatenation** operator, and the asterisk `*` is the **repetition** operator.

# %% {"hidden": true}
l1 = ['hola', 'adios', 17]
l1

# %% {"hidden": true}
print(l1[1:-1])
print(l1[2])

# %% {"hidden": true}
list_concat = ['a','b'] + ['c','d']
print(list_concat)
list_mult = ['1','2'] * 3
print(list_mult)

# %% [markdown] {"hidden": true}
# You can get the length of a list with the command `len`

# %% {"hidden": true}
print(len(list_mult))

# %% [markdown] {"hidden": true, "run_control": {"marked": false}}
# You can create list of lists. 
#
# **IMPORTANT**: If we create a new item using a list, `l1`, what it is saved is the pointer to `l1`. So, if we modify it the list containing it, `l2` will automatically be updated. Notice that this not happens with the other variable `a`. Here we show some examples to show how that works, be sure you understand all of them.

# %% {"hidden": true}
l1 = ['a','b'] # We define a list
a = 1
list_of_lists = [l1,l1,a] # We define a list containing the previous list and other variable, a
print('list_of_lists = ',list_of_lists)

# %% [markdown] {"hidden": true}
# Now we change both, the inner list `l1` and the variable `a`

# %% {"hidden": true}
l1 += ['new']
a = 5
print('list_of_lists = ',list_of_lists)

# %% [markdown] {"hidden": true}
# **Notice that only the part associated to the inner list `l1` has been updated!**
#
# When we modify the `l1` part of `list_of_lists`, also `l1` is updated. But this does not happens with the variable `a`.

# %% {"hidden": true}
list_of_lists[0][2] = 'XXXXX' # We modify the 'new' input in l1
print('l1 = ',l1,'\n')
print('list_of_lists = ',list_of_lists)

# %% {"hidden": true}
list_of_lists[2] = 999
print('a = ',a,'\n')
print('list_of_lists = ',list_of_lists)

# %% [markdown] {"hidden": true}
# You can avoid pointing to the sublist making a **copy of the list**, this is done with the copy method, or using slicing.

# %% {"hidden": true}
l1 = ['a','b']
list_of_lists = [l1 , l1.copy() , l1[:]] # Now the last two entries are copies of l1
print('list_of_lists = ',list_of_lists)

# %% [markdown] {"hidden": true}
# If we modify the inner list, `l1`, only the first entry is gonna be modified.
# But if we change any of the other two entries of `list_of_lists`, they won't affect each other or `l1`.

# %% {"hidden": true}
l1[1] = 'B'
print('l1 = ',l1,'\n')
print('list_of_lists = ',list_of_lists)

# %% {"hidden": true}
list_of_lists[2][1] = 'XXX'
print('l1 = ',l1,'\n')
print('list_of_lists = ',list_of_lists)

# %% [markdown] {"heading_collapsed": true, "hidden": true}
# #### Methods for lists

# %% [markdown] {"hidden": true}
# Python offers methods to work directly on lists. These methods **directly modify the list instead of returning a modified copy of it**.
#
# To check the methods that you can apply to an object in Python, you use the method `dir`.
#
# The methods starting with '__' are considered private. However, unlike other languages, you can use them.

# %% {"hidden": true}
l = [1,'a']
dir(l)

# %% [markdown] {"hidden": true}
# Some useful methods are
# - `append`, appends a new element at the end of the list
# - `insert`, insert a new element in a given index
# - `remove`, remove first occurrence of an element in list
# - `pop`, removes element at given index
# - `reverse`, reverses a list
# - `sort`, sorts elements of a list
# - `index`, gives the smallest index in which an element appears

# %% {"hidden": true}
l = [1,'a']
print(l)
l.append('new')
print(l)

# %% {"hidden": true}
l = [1,'a']
print(l)
l.insert(1,'new')
print(l)

# %% {"hidden": true}
l = [1,'a','b','a']
print(l)
l.remove('a')
print(l)

# %% {"hidden": true}
l = [1,'a','b','a']
print(l)
l.pop(-1)
print(l)

# %% {"hidden": true}
l = [1,'a','b','a']
print(l)
l.reverse()
print(l)

# %% {"hidden": true}
l = ['z','a','b','a','ñ','h','.','1','0.1']
print(l)
l.sort()
print(l)

# %% [markdown] {"hidden": true}
# You need that all the elements have a **compatible type** to apply `sort`.

# %% {"hidden": true}
l = [1,'a','b','a']
print(l)
l.sort()
print(l)

# %% {"hidden": true}
l = ['z','a','b','a','ñ','h','.','1','0.1']
print(l)
l.index('h')

# %% {"hidden": true}
l = ['a','b','b','c','c','c']
print(l)
l.count('b')

# %% [markdown] {"heading_collapsed": true, "hidden": true}
# ### Strings
# `split`

# %% [markdown] {"hidden": true}
# Strings in Python are identified as a contiguous set of characters represented in the quotation marks. Python allows for either pairs of single or double quotes. Subsets of strings can be taken using the slice operator (`[ ]` and `[:]` ) with **indexes starting at 0** in the beginning of the string and working their way from -1 at the end.
#
# The plus `+` sign is the list **concatenation** operator, and the asterisk `*` is the **repetition** operator.

# %% {"hidden": true}
name = "Manolo"
print(type(name))
print("My name is",name)
print("My name is" + name) # Concatenating strings. Notice that in this case we have to add the space manually

# %% {"hidden": true}
string = "Let's play with strings!"

print(string)    # Prints complete string
print(string[0])        # Prints first character of the string
print(string[6:10])      # Prints characters starting from 3rd to 5th
print(string[5:])       # Prints string starting from 3rd character
print(name*2)    # Prints string two times
print(name + '. ' + string)  # Prints concatenated string

# %% [markdown] {"hidden": true}
# The `split` command returns a list of the words of the string, or the string separated by a given character.

# %% {"hidden": true}
string.split()

# %% {"hidden": true}
string.split('s')

# %% [markdown] {"heading_collapsed": true, "hidden": true}
# ### <span style="color:red">**Exercise**</span>

# %% [markdown] {"hidden": true}
# Create two strings, `name` with your first and last name; and an integer variable `age` with your age. 
# Print the following sentece using the two defined variables: 
#
# "My name is `first_name`, and my last name is `last_name` and I am `age` years old."

# %% [markdown] {"heading_collapsed": true, "hidden": true}
# ### <span style="color:red">**Exercise**</span>

# %% [markdown] {"hidden": true}
# Print the name of the folder where the notebook is located. Hint: You can use the `pwd` bash command.

# %% [markdown] {"heading_collapsed": true, "hidden": true}
# ### Tuples

# %% [markdown] {"hidden": true}
# A `tuple` is another sequence data type that is similar to the list. A tuple consists of a number of values separated by commas. Unlike lists, however, tuples are enclosed within parentheses.
#
# The main differences between lists and tuples are: Lists are enclosed in brackets ( `[ ]` ) and their elements and size can be changed, while tuples are enclosed in parentheses ( `( )` ) and cannot be updated. Tuples can be thought of as read-only lists.
#
# The plus `+` sign is the list **concatenation** operator, and the asterisk `*` is the **repetition** operator.

# %% {"hidden": true}
t = ('Manolo', 56)
t

# %% {"hidden": true}
s = (t[1], t[0], t)
print('Tuple s = ',s)
print('Element of s with index 2 =',s[2])

# %% {"hidden": true}
print('Concatenating s and (1,2) = ', s + (1,2) )
print('Duplicating s = ', s * 2)

# %% [markdown] {"hidden": true}
# **The following code is invalid with tuple, because we attempted to update a tuple, which is not allowed.**

# %% {"hidden": true}
s[0] = 57

# %% [markdown] {"heading_collapsed": true, "hidden": true}
# ### Dictionaries
# `keys`,`values`,`items`

# %% [markdown] {"hidden": true}
# A dictionary is a collection which is **unordered**, changeable and indexed. Python dictionaries have **keys** and **values**.
# A dictionary key can be almost any Python type, but are usually numbers or strings. Values, on the other hand, can be any arbitrary Python object.
#
# Dictionaries are enclosed by curly braces (`{ }`) and values can be assigned and accessed using square braces (`[]`).

# %% {"hidden": true}
car_dict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
print(car_dict)

# %% [markdown] {"hidden": true}
# You can add and update elements as with lists.

# %% {"hidden": true}
car_dict['color'] = 'Purple'
car_dict['parts'] = ['door','window','wheels']
car_dict['year'] = 1999
print(car_dict)

# %% [markdown] {"hidden": true}
# **You can access directly to the keys, values and tuples of (key,value).** This will be important when looping over dictionaries.

# %% {"hidden": true}
car_dict.keys()

# %% {"hidden": true}
car_dict.values()

# %% {"hidden": true}
car_dict.items()

# %% [markdown] {"heading_collapsed": true, "hidden": true}
# ### Sets
# `add`,`remove`,`union`,`intersection`

# %% [markdown] {"hidden": true}
# A set is a collection which is **unordered** and **unindexed (faster than lists)**. In Python sets are written with curly brackets. **It automatically eliminates duplicate items in it.** Items belonging to a set can be of different data type.
#
# Note: Sets are unordered, so the items will appear in a **random order**.

# %% {"hidden": true}
fruit_set = {'apple','orange','banana','melon','orange'}
print(fruit_set)

# %% [markdown] {"hidden": true}
# You can modify the set with `add` and `remove`.

# %% {"hidden": true}
print(fruit_set)
fruit_set.add('grapes')
print(fruit_set)
fruit_set.remove('melon')
print(fruit_set)

# %% [markdown] {"hidden": true}
# Python has built-in methods specific to sets like `union` and `intersection`.

# %% {"hidden": true}
fruit_set_2 = {'apple','orange','tomato'}
print('Set 1 = ',fruit_set)
print('Set 2 = ',fruit_set_2)
print('Union = ',fruit_set.union(fruit_set_2))
print('Intersection = ',fruit_set.intersection(fruit_set_2))

# %% [markdown] {"heading_collapsed": true, "hidden": true}
# ### <span style="color:red">**Exercise**</span>

# %% [markdown] {"hidden": true}
# Obtain all the different characters appearing in the given text.

# %% {"hidden": true}
text = '''
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
'''

# %% [markdown] {"heading_collapsed": true}
# ## Control Flow : Conditions and Loops

# %% [markdown] {"heading_collapsed": true, "hidden": true}
# ### Logical operators
# `and`,`or`,`not`,`in`

# %% [markdown] {"hidden": true}
# Python supports the usual logical conditions from mathematics:
#
# - Equals: `a == b`
# - Not Equals: `a != b`
# - Less than: `a < b`
# - Less than or equal to: `a <= b`
# - Greater than: `a > b`
# - Greater than or equal to: `a >= b`
# - Logical operator and: `cond1 and cond2`
# - Logical operator or: `cond1 or cond2`
# - Logical operator negation: `not cond`
#
# You can easily check if an element is (not) in a list / tuple / set with the command `in` (`not in`)

# %% {"hidden": true}
tup = (1,2,3)
print(1 in tup)
li = [1,2,3]
print(2 in li)
se = {1,2,3}
print(3 not in se)

# %% [markdown] {"hidden": true}
# You can use logical expressions with strings

# %% {"hidden": true}
string = 'Do you wanna play?'
print('play' in string)
print('play ' in string)

# %% [markdown] {"heading_collapsed": true, "hidden": true}
# ### If ... elif .. else statement
# `if`,`elif`,`else`

# %% [markdown] {"hidden": true}
# There can be zero or more `elif` parts, and the `else` part is optional. The keyword `elif` is short for ‘else if’, and is useful to avoid excessive indentation. An `if` … `elif` … `elif` … sequence is a substitute for the `switch` or `case` statements found in other languages.
#
# Python relies on **indentation**, using whitespace or tabs, to define scope in the code. Other programming languages often use curly-brackets for this purpose.

# %% {"hidden": true}
a = 200
b = 33
if b > a:
    print("B")
elif a == b:
    print("=")
else:
    print("A")

# %% [markdown] {"hidden": true}
# Short form, it can be difficult to read if the structure is complex.

# %% {"hidden": true}
a = 20
b = 33
print("A") if a > b else print("=") if a == b else print("B")

# %% [markdown] {"hidden": true}
# Inline form, for simple outputs.

# %% {"hidden": true}
a = 35
b = 35
if a > b : print('A')
elif a==b: print('=')
else: print('B')

# %% [markdown] {"hidden": true}
# ### For
# `for`,`range`,

# %% [markdown] {"hidden": true}
# The `for` statement in Python differs a bit from what you may be used to in C or Pascal. Rather than always iterating over an arithmetic progression of numbers, Python’s `for` statement iterates over the items of any sequence (that is either a list, a tuple, a dictionary, a set, or a string), in the order that they appear in the sequence.
#
#
# With the `for` loop we can execute a set of statements, once for each item in a list, tuple, set etc.
#
# `range([start], stop[, step])` allows us to create an iterator in a range of integers. It is 0-index based, meaning list **indexes start at 0, not 1**. It generates numbers up to, but **not including stop**.
#
# In many ways the object returned by `range() behaves as if it is a list, but in fact it isn’t. It is an object which returns the successive items of the desired sequence when you iterate over it, but it doesn’t really make the list, thus saving space.
#
# We say such an object is iterable, that is, suitable as a target for functions and constructs that expect something from which they can obtain successive items until the supply is exhausted. We have seen that the for statement is such an iterator. The function `list()` is another; it creates lists from iterables:

# %% {"hidden": true}
list(range(5))

# %% {"hidden": true}
summ = 0
product = 1
l = []

for i in range(2,10,3):
    l.append(i)
    product *= i
    summ += i
print('The elements are ' + str(l))    
print('The sum is ' + str(summ))
print('The product is ' + str(product))

# %% [markdown] {"hidden": true}
# You can easily loop over a `dict` using the `keys`, `values`and `items` methods. 

# %% {"hidden": true}
car_dict = {'brand': 'Ford',
 'model': 'Mustang',
 'year': 1999,
 'color': 'Purple',
 'parts': ['door', 'window', 'wheels']}

for key,val in car_dict.items():
    print(key,'-->',val)

# %% [markdown] {"heading_collapsed": true, "hidden": true}
# ### break and continue statements
# `break`,`continue`

# %% [markdown] {"hidden": true}
# The `break` statement breaks out of the **innermost** enclosing `for` or `while` loop. Control of the program flows to the statement immediately after the body of the loop.

# %% {"hidden": true}
ingredients = ['onion','tuna','pinneaple','olives']
for ingredient in ingredients:
    print('Pizza with '+ingredient)
    if ingredient == 'pinneaple':
        print('NOOOO!. STOP THAT!')
        break
    print('Yummy!')

# %% [markdown] {"hidden": true}
# The `continue` statement continues with the next iteration of the loop.

# %% {"hidden": true}
ingredients = ['onion','tuna','pinneaple','olives']
for ingredient in ingredients:
    print('Pizza with '+ingredient)
    if ingredient == 'pinneaple':
        print('NOOOO!. NOT THIS ONE!')
        continue
    print('Yummy!')

# %% [markdown] {"heading_collapsed": true, "hidden": true}
# ### Warning: modifying the iterable

# %% [markdown] {"hidden": true}
# If you need to modify the sequence you are iterating over while inside the loop (for example to duplicate selected items), it is recommended that you first make a copy. Iterating over a sequence does not implicitly make a copy. The slice notation makes this especially convenient.
#
# For example, if we want to add the squared of every element of the list to the list itself:

# %% {"hidden": true}
l = [1,2,3,4]
for i in l[:]:
    l.append(i**2)
print(l)

# %% [markdown] {"hidden": true}
# The next cell would create an infinite loop without the `break` statement. 

# %% {"hidden": true}
l = [1,2,3,4]
for i in l:
    l.append(i**2)
    if i > 10000: break # To avoid the infinite loop.
print(l)

# %% [markdown] {"heading_collapsed": true, "hidden": true}
# ### <span style="color:red">**Exercise**</span>

# %% [markdown] {"hidden": true}
# Given a dictionary `diet` associating people with their favorite dishes, and a set of dishes with meat, `meat_set`, obtain a set containing the non vegetarian people.

# %% {"hidden": true}
diet = {'Pepe': ['sausage', 'ham'],
          'Juancho': ['beef', 'ham'],
          'Álvaro': ['beans', 'tomato', 'apple'],
          'Leonor': ['ham', 'olives'],
          'Sandra': ['lettuce', 'kale', 'tofu']}

meat_set = {'sausage', 'ham', 'beef'}

# %% [markdown] {"heading_collapsed": true, "hidden": true}
# ### List Comprehension

# %% [markdown] {"hidden": true}
# List comprehensions provide a concise way to create lists. It consists of brackets containing an expression followed by a for clause, then zero or more for or if clauses. The expressions can be anything, meaning you can put in all kinds of objects in lists.
#
# The result will be a new list resulting from evaluating the expression in the
# context of the for and if clauses which follow it. 
#
# `[ expression for item in list if conditional ]`

# %% {"hidden": true}
power2 = [x**2 for x in range(10)]
print(power2)

# %% {"hidden": true}
even_power2 = [x**2 for x in range(10) if x%2==0]
print(even_power2)


# %% [markdown] {"hidden": true}
# ### <span style="color:red">**Exercise**</span>

# %% [markdown] {"hidden": true}
# Build the list of integers smaller than 100 that are divisible by 3 and 5.

# %% [markdown] {"hidden": true}
# ### <span style="color:red">**Exercise**</span>

# %% [markdown] {"hidden": true}
# Build a list of all prime integers smaller than 100 using list comprehension.

# %% [markdown] {"heading_collapsed": true}
# ## Functions
# `def`,`return`

# %% [markdown] {"hidden": true}
# The function blocks begin with the keyword `def` followed by the function name and parentheses. 
# The function has to be named plus specify what parameter it has. 
# A function can use a number of arguments. Every argument is responding to a parameter in the function. 
#
# The function often ends by returning a **value or tuple** using `return`. 
#

# %% {"hidden": true}
def sum_and_power(x,y):
    return x+y,x**y
print(sum_and_power(3,3))


# %% [markdown] {"hidden": true}
# You can also define functions with **default arguments**. A default argument is an argument that assumes a default value if a value is not provided in the function call for that argument.

# %% {"hidden": true}
def sum_3(x,y,z=None):
    if (z==None):
        return x+y
    else:
        return x+y+z
    
print(sum_3(1, 2))
print(sum_3(1, 2, 3))

# %% [markdown] {"heading_collapsed": true, "hidden": true}
# ### Anonymous functions (lambda functions) and iterables
# `lambda`

# %% [markdown] {"hidden": true}
# A lambda function is a small anonymous function. A lambda function can take any number of arguments, but can only have one expression. 
#
# The syntax is: `lambda arguments : expression`

# %% {"hidden": true}
f = lambda x,y: (x+y,x-y)
f(1,2)

# %% [markdown] {"hidden": true}
# For example, in the `sort` method a custom `key` function can be supplied to customize the sort order.
#
# We will see the usefulness of lambda functions in the next section.

# %% {"hidden": true}
lista = list(range(10))

print(lista)
lista.sort(key = lambda x:x%5)
lista

# %% [markdown]
# ## Iterables
# `enumerate`,`zip`,`map`,`filter`,

# %% [markdown]
# An **iterable** is any Python object capable of returning its members one at a time, permitting it to be iterated over in a for-loop.
#
# Some examples of methods producing iterables are:
# * `enumerate`, iterable that yields pairs containing a count (from start, which defaults to zero) and a value yielded by the iterable argument.
# * `zip`, if multiple iterables are passed, yields tuples containing one element per input.

# %%
l = ['a','b','c']
print('enumerate(l) =',list(enumerate(l)))
for i,c in enumerate(l):
    print('Element',i,'is',c)

# %%
titles = ['Monty Python and the Holy Grail','Life of Brian','The Meaning of Life']
years = [1975,1979,1983]
for title, year in zip(titles,years):
    print('The film "'+title+'" was released on '+str(year)+'.')

# %% [markdown]
# You can use `zip` to easily build a `dict`.

# %%
dict(zip(titles,years))

# %% [markdown]
# The power of lambda functions is better shown when you use them as an anonymous function inside another function, like:
# - `map`, applies a function to all the items in an iterable (list,set,dic, str), returning an iterable.
# - `filter`, creates a list of elements of an iterable for which a function returns true, returning an iterable.
#
# **Note:** For both of them you have to transform the output into the desired type: list, set, tuple, ...

# %% [markdown]
# All even integers smaller than 10.

# %%
list(filter(lambda x: x%2 == 0, range(0,10)))

# %% [markdown]
# `map` and `filter` can be applied to a string.

# %%
vowels = 'aeiou'
vowels += vowels.upper() # upper() convert a strings to uppercase
print(vowels)
list(map(lambda x: [x,x in vowels],'Am I a vowel?'))

# %% [markdown]
# ### <span style="color:red">**Exercise**</span>

# %% [markdown]
# Using `filter` and lambda functions, create a list of all the positive integers smaller than 100 which contain a 3 (this is: 32, 83, ...)
