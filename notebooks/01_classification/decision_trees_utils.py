import numpy as np
import string
import pydotplus
from IPython.display import Image
from collections import deque
from collections import namedtuple
from matplotlib.collections import LineCollection
from itertools import starmap
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from ipywidgets import interact


def print_tree_sklearn(clf, feature_names=list(), class_names=list()):
    number_features = clf.n_features_
    number_classes = clf.n_classes_
    if len(class_names) < number_classes:
        class_names += [string.ascii_uppercase[i]
                        for i in range(len(class_names), number_classes)]
    elif len(class_names) > number_classes:
        print('The maximum number of classes is', number_classes)
        class_names = string.ascii_uppercase[0:number_classes]

    if len(feature_names) < number_features:
        feature_names += ['X' + str(i)
                          for i in range(len(feature_names), number_features)]
    elif len(feature_names) > number_features:
        print('The maximum number of features is', number_features)
        feature_names = feature_names[0:number_features]

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data, feature_names=feature_names,
                    filled=True, rounded=True, class_names=class_names,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return Image(graph.create_png())


class Node(namedtuple('Node', 'feature, threshold, left, right')):
    pass


class NodeRanges(namedtuple('NodeRanges', 'node, max_x, min_x, max_y, min_y')):
    pass


def tree_dic_to_nodes(dic, feature_names=['X0', 'X1']):
    nodes, _ = tree_dic_to_nodes_counter(
        dic, counter=0, feature_names=feature_names)
    return nodes


def tree_dic_to_nodes_counter(dic, counter=0, feature_names=['X0', 'X1']):
    counter_left = -1
    counter_right = -1
    list_left = list()
    list_right = list()

    if dic['left'] is None:
        counter += 1
        counter_left = counter
        list_left = [Node(None, -2.0, -1, -1)]

    elif isinstance(dic['left'], dict):
        counter += 1
        counter_left = counter
        list_left, counter = tree_dic_to_nodes_counter(dic['left'], counter,
                                                       feature_names=feature_names)

    if dic['right'] is None:
        counter += 1
        counter_right = counter
        list_right = [Node(None, -2.0, -1, -1)]

    elif isinstance(dic['right'], dict):
        counter += 1
        counter_right = counter
        list_right, counter = tree_dic_to_nodes_counter(dic['right'], counter,
                                                        feature_names=feature_names)

    list_nodes = [
        Node(feature_names[dic['feature']], dic['threshold'], counter_left,
             counter_right)]
    list_nodes += list_left
    list_nodes += list_right

    return list_nodes, counter


def tree_sklearn_to_nodes(clf, feature_names=['X0', 'X1']):
    number_features = clf.tree_.n_features
    if len(feature_names) < number_features:
        feature_names += ['X' + str(i)
                          for i in range(len(feature_names), number_features)]
    elif len(feature_names) > number_features:
        print('The maximum number of features is', number_features)
        feature_names = feature_names[0:number_features]

    dic_features = dict(enumerate(feature_names))
    nodes = starmap(namedtuple('Node', 'feature, threshold, left, right'),
                    zip(map(lambda x: dic_features.get(x), clf.tree_.feature),
                        clf.tree_.threshold,
                        clf.tree_.children_left,
                        clf.tree_.children_right))
    return list(nodes)


def tree_nodes_to_boundaries_2d(nodes, xrange, yrange,
                                feature_names_2d=['X0', 'X1']):
    assert len(feature_names_2d) == 2, 'feature_names_2d must have len = 2'
    stack = deque()
    stack.append(NodeRanges(node=nodes[0],
                            max_x=xrange[1],
                            min_x=xrange[0],
                            max_y=yrange[1],
                            min_y=yrange[0]))

    linesxxyy = []
    while len(stack):
        node, max_x, min_x, max_y, min_y = stack.pop()

        feature, threshold, left, right = node
        if feature == feature_names_2d[0]:
            linesxxyy.append([(threshold, threshold), (min_y, max_y)])
        elif feature == feature_names_2d[1]:
            linesxxyy.append([(min_x, max_x), (threshold, threshold)])
        else:
            continue
        stack.append(NodeRanges(node=nodes[left],
                                max_x=threshold if feature == feature_names_2d[
                                    0] else max_x,
                                min_x=min_x,
                                max_y=threshold if feature == feature_names_2d[
                                    1] else max_y,
                                min_y=min_y))

        stack.append(NodeRanges(node=nodes[right],
                                max_x=max_x,
                                min_x=threshold if feature == feature_names_2d[
                                    0] else min_x,
                                max_y=max_y,
                                min_y=threshold if feature == feature_names_2d[
                                    1] else min_y))

    return linesxxyy


def plot_tree_boundaries(ax, clf_or_nodes, is_nodes=False, is_sklearn_clf=True):

    if is_nodes:
        nodes = clf_or_nodes
    else:
        if is_sklearn_clf:
            nodes = tree_sklearn_to_nodes(
                clf_or_nodes, feature_names=['X0', 'X1'])
        else:
            nodes = tree_dic_to_nodes(clf_or_nodes, feature_names=['X0', 'X1'])

    linesxxyy = tree_nodes_to_boundaries_2d(nodes,
                                            xrange=ax.get_xlim(),
                                            yrange=ax.get_ylim(),
                                            feature_names_2d=['X0', 'X1'])
    linesxyxy = [np.array(line).T for line in linesxxyy]

    lc = LineCollection(linesxyxy, colors='black', lw=1.5)
    ax.add_collection(lc)
    ax.figure.canvas.draw()
    return None


def plot_interacting_tree_boundaries(ax, clf_or_nodes, is_nodes=False,
                                     is_sklearn_clf=True):

    if is_nodes:
        nodes = clf_or_nodes
    else:
        if is_sklearn_clf:
            nodes = tree_sklearn_to_nodes(
                clf_or_nodes, feature_names=['X0', 'X1'])
        else:
            nodes = tree_dic_to_nodes(clf_or_nodes, feature_names=['X0', 'X1'])

    linesxxyy = tree_nodes_to_boundaries_2d(nodes,
                                            xrange=ax.get_xlim(),
                                            yrange=ax.get_ylim(),
                                            feature_names_2d=['X0', 'X1'])
    linesxyxy = [np.array(line).T for line in linesxxyy]
    n_splits = len(linesxyxy)
    n_collections_min = len(ax.collections)

    @interact
    def show_articles_more_than(step=(0, n_splits, 1)):
        # Lines in format (x,y),(x,y).
        if len(ax.collections) > n_collections_min:
            ax.collections = ax.collections[0:n_collections_min]
        lc = LineCollection(linesxyxy[0:step], colors='black', lw=1.5)
        ax.add_collection(lc)
        ax.figure.canvas.draw()
        return None
    return None


# def cart_plot(clf_or_nodes, xrange, yrange, feature_names=['X0', 'X1'],
#               plot=None, is_nodes=False):
#     if is_nodes:
#         nodes = clf_or_nodes
#     else:
#         nodes = tree_to_nodes(clf_or_nodes)

#     if plot == None:
#         plot = figure()

#     # add_line(args) = plot.line(line_color='black', line_width=2, args)
#     add_line = partial(plot.line, line_color='black', line_width=2)
#     stack = deque()
#     stack.append(NodeRanges(node=nodes[0],
#                             max_x=xrange[1],  # df[feature_names[0]].max(),
#                             min_x=xrange[0],  # df[feature_names[0]].min(),
#                             max_y=yrange[1],  # df[feature_names[1]].max(),
#                             min_y=yrange[0]))  # df[feature_names[1]].min()))

#     while len(stack):
#         node, max_x, min_x, max_y, min_y = stack.pop()
#         feature, threshold, left, right = node
#         if feature == feature_names[0]:
#             add_line(x=[threshold, threshold], y=[min_y, max_y])
#         elif feature == feature_names[1]:
#             add_line(x=[min_x, max_x], y=[threshold, threshold])
#         else:
#             continue
#         stack.append(NodeRanges(node=nodes[left],
#                                 max_x=threshold if feature == feature_names[
#                                     0] else max_x,
#                                 min_x=min_x,
#                                 max_y=threshold if feature == feature_names[1] else max_y, min_y=min_y))
#         stack.append(NodeRanges(node=nodes[right], max_x=max_x,
#                                 min_x=threshold if feature == feature_names[0] else min_x, max_y=max_y,
#                                 min_y=threshold if feature == feature_names[1] else min_y))
#     show(plot)
#     return None


# def cart_plot_stops(clf_tree, xrange, yrange, feature_names=['X0', 'X1'],
#                     plot=None):
#     nodes = tree_to_nodes(clf_tree)

#     if plot == None:
#         plot = figure()

#     # add_line(args) = plot.line(line_color='black', line_width=2, args)
#     add_line = partial(plot.line, line_color='black', line_width=2)
#     stack = deque()
#     stack.append(NodeRanges(node=nodes[0],
#                             max_x=xrange[1],  # df[feature_names[0]].max(),
#                             min_x=xrange[0],  # df[feature_names[0]].min(),
#                             max_y=yrange[1],  # df[feature_names[1]].max(),
#                             min_y=yrange[0]))  # df[feature_names[1]].min()))

#     while len(stack):
#         node, max_x, min_x, max_y, min_y = stack.pop()
#         feature, threshold, left, right = node
#         if feature == feature_names[0]:
#             add_line(x=[threshold, threshold], y=[min_y, max_y])
#         elif feature == feature_names[1]:
#             add_line(x=[min_x, max_x], y=[threshold, threshold])
#         else:
#             continue
#         stack.append(NodeRanges(node=nodes[left],
#                                 max_x=threshold if feature == feature_names[
#                                     0] else max_x,
#                                 min_x=min_x,
#                                 max_y=threshold if feature == feature_names[1] else max_y, min_y=min_y))
#         stack.append(NodeRanges(node=nodes[right], max_x=max_x,
#                                 min_x=threshold if feature == feature_names[0] else min_x, max_y=max_y,
#                                 min_y=threshold if feature == feature_names[1] else min_y))
#         input(prompt='Add new line')
#         show(plot)
#     show(plot)


# def number_of_splits(clf_tree):
#     nodes = tree_to_nodes(clf_tree)
#     number_splits = len([node[0] for node in nodes if node[0] is not None])
#     return number_splits


# def multiple_cart_plots(clf_tree, xrange, yrange, feature_names=['X0', 'X1']):
#     nodes = tree_to_nodes(clf_tree)
#     number_plots = len([node[0] for node in nodes if node[0] is not None])
#     list_plots = []
#     for counter in range(0, number_plots + 1):
#         new_plot = create_cart_plot_counter(
#             counter, nodes, xrange, yrange, feature_names=feature_names)
#         list_plots.append(new_plot)
#     return list_plots


# def create_cart_plot_counter(counter, nodes, xrange, yrange,
#                              feature_names=['X0', 'X1'], plot=None):
#     #     nodes = tree_to_nodes(dtree)
#     if plot == None:
#         plot = figure()
#     # add_line(args) = plot.line(line_color='black', line_width=2, args)
#     add_line = partial(plot.line, line_color='black', line_width=2)
#     stack = deque()
#     stack.append(NodeRanges(node=nodes[0],
#                             max_x=xrange[1],  # df[feature_names[0]].max(),
#                             min_x=xrange[0],  # df[feature_names[0]].min(),
#                             max_y=yrange[1],  # df[feature_names[1]].max(),
#                             min_y=yrange[0]))  # df[feature_names[1]].min()))

#     while len(stack) and (counter > 0):
#         node, max_x, min_x, max_y, min_y = stack.pop()
#         feature, threshold, left, right = node
#         if feature == feature_names[0]:
#             add_line(x=[threshold, threshold], y=[min_y, max_y])
#             counter -= 1
#         elif feature == feature_names[1]:
#             add_line(x=[min_x, max_x], y=[threshold, threshold])
#             counter -= 1
#         else:
#             continue
#         stack.append(NodeRanges(node=nodes[left],
#                                 max_x=threshold if feature == feature_names[
#                                     0] else max_x,
#                                 min_x=min_x,
#                                 max_y=threshold if feature == feature_names[1] else max_y, min_y=min_y))
#         stack.append(NodeRanges(node=nodes[right], max_x=max_x,
#                                 min_x=threshold if feature == feature_names[0] else min_x, max_y=max_y,
# min_y=threshold if feature == feature_names[1] else min_y))

#     return plot