from bokeh.io import output_notebook
from bokeh.plotting import figure
from bokeh.plotting import output_file
from bokeh.plotting import show

import pydotplus as pydot  # Solve problem with

from IPython.display import Image
from collections import deque
from collections import namedtuple
from collections import namedtuple
from functools import partial
from io import StringIO
from itertools import starmap
import pydotplus

from IPython.display import Image
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz


def tree_to_nodes(dtree, feature_names=['X0', 'X1']):
    nodes = starmap(namedtuple('Node', 'feature, threshold, left, right'),
                    zip(map(lambda x: {0: feature_names[0], 1: feature_names[1]}.get(x), dtree.tree_.feature),
                        dtree.tree_.threshold, dtree.tree_.children_left, dtree.tree_.children_right))
    return list(nodes)


class NodeRanges(namedtuple('NodeRanges', 'node, max_x, min_x, max_y, min_y')):
    pass


def cart_plot(clf_tree, xrange, yrange, feature_names=['X0', 'X1'], plot=None):

    nodes = tree_to_nodes(clf_tree)

    if plot == None:
        plot = figure()

    # add_line(args) = plot.line(line_color='black', line_width=2, args)
    add_line = partial(plot.line, line_color='black', line_width=2)
    stack = deque()
    stack.append(NodeRanges(node=nodes[0],
                            max_x=xrange[1],  # df[feature_names[0]].max(),
                            min_x=xrange[0],  # df[feature_names[0]].min(),
                            max_y=yrange[1],  # df[feature_names[1]].max(),
                            min_y=yrange[0]))  # df[feature_names[1]].min()))

    while len(stack):
        node, max_x, min_x, max_y, min_y = stack.pop()
        feature, threshold, left, right = node
        if feature == feature_names[0]:
            add_line(x=[threshold, threshold], y=[min_y, max_y])
        elif feature == feature_names[1]:
            add_line(x=[min_x, max_x], y=[threshold, threshold])
        else:
            continue
        stack.append(NodeRanges(node=nodes[left],
                                max_x=threshold if feature == feature_names[
                                    0] else max_x,
                                min_x=min_x,
                                max_y=threshold if feature == feature_names[1] else max_y, min_y=min_y))
        stack.append(NodeRanges(node=nodes[right], max_x=max_x,
                                min_x=threshold if feature == feature_names[0] else min_x, max_y=max_y,
                                min_y=threshold if feature == feature_names[1] else min_y))
    show(plot)


def cart_plot_stops(clf_tree, xrange, yrange, feature_names=['X0', 'X1'], 
    plot=None):
    nodes = tree_to_nodes(clf_tree)

    if plot == None:
        plot = figure()

    # add_line(args) = plot.line(line_color='black', line_width=2, args)
    add_line = partial(plot.line, line_color='black', line_width=2)
    stack = deque()
    stack.append(NodeRanges(node=nodes[0],
                            max_x=xrange[1],  # df[feature_names[0]].max(),
                            min_x=xrange[0],  # df[feature_names[0]].min(),
                            max_y=yrange[1],  # df[feature_names[1]].max(),
                            min_y=yrange[0]))  # df[feature_names[1]].min()))

    while len(stack):
        node, max_x, min_x, max_y, min_y = stack.pop()
        feature, threshold, left, right = node
        if feature == feature_names[0]:
            add_line(x=[threshold, threshold], y=[min_y, max_y])
        elif feature == feature_names[1]:
            add_line(x=[min_x, max_x], y=[threshold, threshold])
        else:
            continue
        stack.append(NodeRanges(node=nodes[left],
                                max_x=threshold if feature == feature_names[
                                    0] else max_x,
                                min_x=min_x,
                                max_y=threshold if feature == feature_names[1] else max_y, min_y=min_y))
        stack.append(NodeRanges(node=nodes[right], max_x=max_x,
                                min_x=threshold if feature == feature_names[0] else min_x, max_y=max_y,
                                min_y=threshold if feature == feature_names[1] else min_y))
        input(prompt='Add new line')
        show(plot)
    show(plot)


def print_cart(clf, feature_names=None):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data, feature_names=feature_names,
                    filled=True, rounded=True, class_names=['X', 'O'],
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return Image(graph.create_png())


def number_of_splits(clf_tree):
    nodes = tree_to_nodes(clf_tree)
    number_splits = len([node[0] for node in nodes if node[0] is not None])
    return number_splits

def multiple_cart_plots(clf_tree, xrange, yrange, feature_names=['X0', 'X1']):
    nodes = tree_to_nodes(clf_tree)
    number_plots = len([node[0] for node in nodes if node[0] is not None])
    list_plots = []
    for counter in range(0, number_plots + 1):
        new_plot = create_cart_plot_counter(
            counter, nodes, xrange, yrange, feature_names=feature_names)
        list_plots.append(new_plot)
    return list_plots


def create_cart_plot_counter(counter, nodes, xrange, yrange, 
    feature_names=['X0', 'X1'], plot=None):
    #     nodes = tree_to_nodes(dtree)
    if plot == None:
        plot = figure()
    # add_line(args) = plot.line(line_color='black', line_width=2, args)
    add_line = partial(plot.line, line_color='black', line_width=2)
    stack = deque()
    stack.append(NodeRanges(node=nodes[0],
                            max_x=xrange[1],  # df[feature_names[0]].max(),
                            min_x=xrange[0],  # df[feature_names[0]].min(),
                            max_y=yrange[1],  # df[feature_names[1]].max(),
                            min_y=yrange[0]))  # df[feature_names[1]].min()))

    while len(stack) and (counter > 0):
        node, max_x, min_x, max_y, min_y = stack.pop()
        feature, threshold, left, right = node
        if feature == feature_names[0]:
            add_line(x=[threshold, threshold], y=[min_y, max_y])
            counter -= 1
        elif feature == feature_names[1]:
            add_line(x=[min_x, max_x], y=[threshold, threshold])
            counter -= 1
        else:
            continue
        stack.append(NodeRanges(node=nodes[left],
                                max_x=threshold if feature == feature_names[
                                    0] else max_x,
                                min_x=min_x,
                                max_y=threshold if feature == feature_names[1] else max_y, min_y=min_y))
        stack.append(NodeRanges(node=nodes[right], max_x=max_x,
                                min_x=threshold if feature == feature_names[0] else min_x, max_y=max_y,
                                min_y=threshold if feature == feature_names[1] else min_y))

    return plot
