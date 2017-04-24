import sys


class Node:
    def __init__(self, index, word, tag, category, label):
        self.index = index
        self.word = word
        self.tag = tag
        self.category = category
        self.label = label
        self.head = None
        self.children = []

    def breadth_first(self):
        yield self
        last = self
        for node in self.breadth_first():
            for child in node.children:
                yield child
                last = child
            if last == node:
                return


class Tree:
    def __init__(self, line):
        p = line.strip().split()
        assert len(p) % 5 == 0
        self.root = Node(-1, '__ROOT__', '__ROOT__', '__ROOT__', '__ROOT__')
        self.nodes = ([Node(int(i / 5), p[i], p[i + 1], p[i + 2], p[i + 3])
                       for i in range(0, len(p), 5)])
        for j, head_index in enumerate(
                [int(p[i + 4]) for i in range(0, len(p), 5)]):
            if head_index == -1:
                self.nodes[j].head = self.root
                self.root.children.append(self.nodes[j])
            else:
                self.nodes[j].head = self.nodes[head_index]
                self.nodes[head_index].children.append(self.nodes[j])

    def __repr__(self):
        r = ''
        for node in self.root.breadth_first():
            if not node.children or not node.head:
                continue
            r += '(%s:%s  (%s))\n' % (
                node.word, node.category, ' '.join(['%s:%s' %
                                                    (child.word, child.label) for child in node.children]))
        return r


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tree.py parses")
        sys.exit(0)
    for line in open(sys.argv[1]):
        t = Tree(line)
        print(' '.join([node.word for node in t.nodes]))
        print(t)
        print()
