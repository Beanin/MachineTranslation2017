from tree import Tree
import sys
from model import prepareData, trainModels, ChildrenReorderingModel


class Reorderer:
    def reorder(self, root):
        assert False, "Not implemented."


class RecursiveReorderer(Reorderer):
    def reorder(self, tree):
        return self.reorder_recursively(tree.root, [])

    def reorder_recursively(self, head, ordering):
        for node in self.reorder_head_and_children(head):
            if node == head:
                ordering.append(node)
            else:
                self.reorder_recursively(node, ordering)

        return ordering

        # 1. Call 'reorder_head_and_children' to determine
        # order of immediate subtree.
        # 2. Walk through immediate subtree in this order,
        # calling 'reorder_recursively'
        # on children and adding head to 'ordering' when it's reached.

    def reorder_head_and_children(self, head):
        # Reorder the head and children in the desired order.
        assert False, "TODO: implement me in a subclass."


class DoNothingReorderer(RecursiveReorderer):
    # Just orders head and child nodes according to their original index.
    def reorder_head_and_children(self, head):
        all_nodes = (
            [(child.index, child)
             for child in head.children] + [(head.index, head)])

        return [node for index, node in sorted(all_nodes)]


class ReverseReorderer(RecursiveReorderer):
    # Reverse orders head and child nodes according original index
    def reorder(self, head):
        return DoNothingReorderer().reorder(head)[::-1]


class HeadFinalReorderer(RecursiveReorderer):
    """ 0.796"""

    def reorder_children(self, head):
        return [node for node in sorted(head.children,
                                        key=lambda n:n.index)]

    def reorder_head_and_children(self, head):
        order = self.reorder_children(head)
        return order + [head]


def SOVKeyFunc(node):
    """ Gives 0.791 if move obj to the end and
        0.790 if move subj to the beginning """
    if node.label.endswith('subj'):
        return (-1, node.index)
    else:
        return (0, node.index)


class SOVHeadFinalReorderer(RecursiveReorderer):
    def reorder_head_and_children(self, head):
        return sorted(head.children, key=SOVKeyFunc) + [head]


class ClassifierReorderer(HeadFinalReorderer):
    def __init__(self):
        data = prepareData()

        self.models = trainModels(data)

    def reorder_children(self, head):
        length = len(head.children)
        model = self.models.setdefault(length,
                                       ChildrenReorderingModel())
        return model.predict(head)


if __name__ == "__main__":
    if not len(sys.argv) == 3:
        print("python reorderers.py ReordererClass parses")
        sys.exit(0)
    # Instantiates the reorderer of this class name.
    reorderer = eval(sys.argv[1])()

    # Reorders each input parse tree and prints words to std out.
    for line in open(sys.argv[2]):
        t = Tree(line)
        assert t.root
        reordering = reorderer.reorder(t)
        print(' '.join([node.word for node in reordering if node != t.root]))
