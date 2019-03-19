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
# # Decision tree step by step

# %%
# %load_ext autoreload
# %autoreload 2
import os

# %%
##### GOOGLE COLAB ######
if not os.path.exists('./decision_trees_utils.py'):
    ! wget https://raw.githubusercontent.com/davidggphy/2019_03_Curso_IFT/master/notebooks/01_classification/decision_trees_utils.py
if not os.path.exists('./plot_confusion_matrix.py'):
    ! wget https://raw.githubusercontent.com/davidggphy/2019_03_Curso_IFT/master/notebooks/01_classification/plot_confusion_matrix.py

# %%
# ! pip install ipywidgets
# ! jupyter nbextension enable --py widgetsnbextension

# %%
# %matplotlib notebook

# # %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.style.use('seaborn-whitegrid')
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from decision_trees_utils import print_cart
from decision_trees_utils import tree_to_nodes

# %% [markdown]
# ## Creating and preprocessing the data

# %%
dic_data_A = {'n': 15 ,'mean': (0,2), 'cov' :((1,0),(0,2)), 'y' : 0 }  # RED
dic_data_B = {'n': 15 ,'mean': (0,0), 'cov' :((3,0),(0,1)), 'y' : 1 }  # BLUE
dic_data = {'A': dic_data_A, 'B' : dic_data_B }

# We sample the points with numpy.random
np.random.seed(1)
samples = {key : np.random.multivariate_normal(dic['mean'], np.array(dic['cov']), dic['n']) 
           for key,dic in dic_data.items()}
     
X = np.concatenate(tuple(samples[key] for key in dic_data.keys() ),axis=0)
Y = np.concatenate(tuple(dic['y']* np.ones(dic['n'], dtype='int') 
                         for key,dic in dic_data.items() ), axis=0)



# Train Test Split. 80% / 20%.
X_train,X_test,Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)

# %%
# We transform into coordinates to make plotting easier.
colors = [
    "#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g ,b  in 150*np.eye(3)[[0,2,1]][np.array(Y_train,dtype='int')]
]
colors_test = [
    "#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g ,b  in 150*np.eye(3)[[0,2,1]][np.array(Y_test,dtype='int')]
]
x0_range = (X[:,0].min()-1,X[:,0].max()+1)
x1_range = (X[:,1].min()-1,X[:,1].max()+1)
x0 = X_train[:,0] 
x1 = X_train[:,1] 
x0_test = X_test[:,0] 
x1_test = X_test[:,1] 

# %%
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.set_xlim(x0_range)
ax.set_ylim(x1_range)
# TRAIN
ax.scatter(x0, x1, s = 17, 
          alpha=1, c=colors )
# TEST
ax.scatter(x0_test, x1_test, s = 17, 
          alpha=0.3, c=colors_test )

ax.figure.canvas.draw()

# %% [markdown]
# ## Logistic Regression with sklearn

# %%
from sklearn.linear_model import LogisticRegression

# %%
clf_log = LogisticRegression(solver='lbfgs').fit(X_train,Y_train)
print('TRAIN SET\n---------\n',sklearn.metrics.classification_report(Y_train,clf_log.predict(X_train)))
print('TEST SET\n--------\n',sklearn.metrics.classification_report(Y_test,clf_log.predict(X_test)))

# %% [markdown]
# We can extract the coefficients of the linear regression to plot the decision boundary

# %%
# Coefficients of the linear regression's bondary. Bias and slopes
bias = clf_log.intercept_[0]
alpha0, alpha1 = tuple(clf_log.coef_[0])
x_line = x0_range
y_line = tuple(-(bias+alpha0*x)/alpha1 for x in x_line)

# %%
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.set_xlim(x0_range)
ax.set_ylim(x1_range)

# TRAIN
ax.scatter(x0, x1, s = 17, 
          alpha=1, c=colors )
# TEST
ax.scatter(x0_test, x1_test, s = 17, 
          alpha=0.3, c=colors_test )
# LINE
ax.plot(x_line,y_line, c='black')
# PLOT ALL
ax.figure.canvas.draw()

# %% [markdown]
# ## Classification Decision Tree with Sklearn

# %%
from sklearn import tree
from decision_trees_utils import add_interacting_boundaries, add_boundaries
clf_gini = tree.DecisionTreeClassifier(criterion='gini').fit(X_train, Y_train)
clf_ent = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train, Y_train)
# clf1 = tree.DecisionTreeClassifier(min_samples_split=5, min_samples_leaf=3).fit(X_train, Y_train)
# clf2 = tree.DecisionTreeClassifier(max_depth=2).fit(X_train, Y_train)
print('TRAIN SET\n---------\n',sklearn.metrics.classification_report(Y_train,clf_gini.predict(X_train)))
print('TEST SET\n---------\n',sklearn.metrics.classification_report(Y_test,clf_gini.predict(X_test)))

# %%
print_cart(clf_gini)

# %%
fig, ax = plt.subplots(figsize=(5,5))

# AXES LIMITS
ax.set_xlim(x0_range)
ax.set_ylim(x1_range)

# TRAIN
ax.scatter(x0, x1, s = 17, 
          alpha=1, c=colors )
# TEST
ax.scatter(x0_test, x1_test, s = 17, 
          alpha=0.3, c=colors_test )

# ax.figure.canvas.draw()

# DECISSION TREE BOUNDARIES
add_boundaries(ax,clf_gini)

# %%
fig, ax = plt.subplots(figsize=(5,5))

# AXES LIMITS
ax.set_xlim(x0_range)
ax.set_ylim(x1_range)

# TRAIN
ax.scatter(x0, x1, s = 17, 
          alpha=1, c=colors )
# TEST
ax.scatter(x0_test, x1_test, s = 17, 
          alpha=0.3, c=colors_test )

# ax.figure.canvas.draw()

# DECISSION TREE BOUNDARIES
add_boundaries(ax,clf_ent)

# %% [markdown]
# Different criteria change the decision boundaries.
#
# We can see how the algorithm works exploring how the boundaries are created step by step.

# %%
fig, ax = plt.subplots(figsize=(5,5))

# AXES LIMITS
ax.set_xlim(x0_range)
ax.set_ylim(x1_range)

# TRAIN
ax.scatter(x0, x1, s = 17, 
          alpha=1, c=colors )
# TEST
ax.scatter(x0_test, x1_test, s = 17, 
          alpha=0.3, c=colors_test )

# ax.figure.canvas.draw()

# DECISSION TREE BOUNDARIES
add_interacting_boundaries(ax,clf_gini)

# %% [markdown]
# ## Some lessons

# %% [markdown]
# **Properties:**
# - The decision tree behaves as a nested set of if else conditions.
# - Interpretable (at least the first nodes)
#     -  We can know why we got a prediction
#     -  Contrast with common sense and domain knowledge
# - Decision trees will overfit
# - But will generalize using next parameters:
#     -  min_samples_leaf
#     -  min_samples_split 
#     -  max_depth 
#     -  max_leaf_nodes
# - Compared to a plain logistic regression, decision trees are slower (no vectorization), but really fast (we can do hyperparameter optimization)
# - We have many hyperparameters (size of data, depth of tree, etc)
# - No standarization / normalization (Using original units we will be able to understand the tree better)
# - Remember that decision boundaries are always orthogonal to the axis, PCA can be useful
# - Feature selection for free (not used if it not important clf.feature_importances_)
# - Ordinal categorical variables are treated nicely (They have an intrinsic order, and grouped with neighbours)
# - Low stability (Small changes in data, can cause a big change in the model)
#

# %% [markdown]
# ## Decision Tree from scratch

# %%
from collections import namedtuple
class NestedNode(namedtuple('NestedNode', 'feature, threshold, left_node, right_node')):
    pass
class Node(namedtuple('Node', 'feature, threshold, left, right')):
    pass

def entropy(n_classes): 
    # We have to avoid having a left/right part with no elements
    # The number does not matter, since it will be multiplied by prob_right/left = 0.
    if np.sum(n_classes)==0:
        return 9999999.
    probs = n_classes/np.sum(n_classes)
    # Regularize null probabilities
    probs = probs[probs > 0.]
    ent = -np.sum(probs * np.log2(probs) )
    return ent


def get_best_split(X, Y, class_values = [0,1], criterion='entropy'):
    best_split = None
    best_criterion = 9999999.
    # Column vector, we will need it to broadcast later
    class_values = np.array(class_values).reshape((1,-1))
    for feature, values_feature in enumerate(X.T): # feature is the string naming each input feature
        values_sorted = np.sort(values_feature)
        for index, value in enumerate(values_sorted[:-1]):
            # LEFT PART, <= value
            n_classes_left = np.sum((Y[X[:,feature] <= value].reshape((-1,1)) == class_values),axis=0)
            prob_left = np.sum(n_classes_left)/Y.size
            if criterion=='entropy':
                criterion_left = entropy(n_classes_left)
                
            # RIGTH PART, > value
            n_classes_right = np.sum((Y[X[:,feature] > value].reshape((-1,1)) == class_values),axis=0)
            prob_right = np.sum(n_classes_right)/Y.size
            if criterion=='entropy':
                criterion_right = entropy(n_classes_right)
            
            criterion_split = prob_left * criterion_left + prob_right * criterion_right 
            
            if criterion_split < best_criterion:
                best_split = (feature, (value+values_sorted[index+1])/2) 
                best_criterion = criterion_split
    return best_split

def train_decision_tree(X, Y, class_values = [0,1], criterion='entropy', min_samples_leaf=1):
    feature, value = get_best_split(X, Y, class_values= class_values, criterion=criterion)
    X_left, Y_left = X[X[:,feature] <= value], Y[X[:,feature] <= value]
    if (np.unique(Y_left).size > 1)and(Y_left.size > min_samples_leaf):
        left_node = train_decision_tree(X_left, Y_left,class_values=class_values, min_samples_leaf=min_samples_leaf) 
    else:
        left_node = None
    X_right, Y_right = X[X[:,feature] > value], Y[X[:,feature] > value]
    if (np.unique(Y_right).size > 1)and(Y_right.size > min_samples_leaf):
        right_node = train_decision_tree(X_right, Y_right,class_values=class_values, min_samples_leaf=min_samples_leaf) 
    else:
        right_node = None
    return NestedNode(feature, value, left_node, right_node)


# %% [markdown]
# Let's apply it

# %%
train_decision_tree(X_train,Y_train, criterion='entropy')

# %% [markdown]
# We can compare it with the sklearn one

# %%
clf_ent = sklearn.tree.DecisionTreeClassifier(criterion='entropy').fit(X_train,Y_train)
tree_to_nodes(clf_ent,feature_names=[0,1])


# %% [markdown]
# ## Exercise: 
#
# Implement the Decission Tree Classifier using the gini index instead of the entropy and compare with the sklearn solution.

# %%
def gini(n_classes): 
    ### TODO
    return gin


# %%

# %% [markdown] {"heading_collapsed": true}
# ## Another implementation
# https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

# %% {"hidden": true}
# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
 
# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini
 
# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)
 
# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root
 
# Print a decision tree
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))
 

dataset = np.hstack([X_train,Y_train.reshape((-1,1))])
tree = build_tree(dataset, 15, 5)
print_tree(tree)

# %% {"hidden": true}
print_cart(clf)

# %% {"hidden": true}
