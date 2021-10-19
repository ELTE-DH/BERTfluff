import sys
import csv
import pickle
from typing import List

# Loosely based on https://albertauyeung.github.io/2020/06/15/python-trie.html


class TrieNode:
    """A node in the trie structure"""

    def __init__(self, word_id):
        # the item stored in this node
        self.word_id = word_id

        # a counter indicating how many times a word is inserted
        self.counter = 0

        self.is_end = False

        # a dictionary of child nodes keys are characters, values are nodes
        self.children = {}

    def get_repr_string(self, m: int = 0):
        tab = '\t'
        pref = f'{tab*m}{self.word_id}: {self.counter}'
        children = ''
        if len(self.children) > 0:
            children = f'{"".join(node.get_repr_string(m + 1) for node in self.children.values())}{tab*m}\n'
        ret = f'{pref}\n{children}'
        return ret


class Trie:
    """The trie object"""

    def __init__(self):
        """
        The trie has at least the root node.
        The root node does not store any character
        """
        self._trienode_factory = TrieNode
        self.root = self._trienode_factory(-1)

    def insert(self, elems: List[int]):
        """Insert a word into the trie"""
        node = self.root

        # Loop through each character in the word
        # Check if there is no child containing the character, create a new child for the current node
        for elem in elems:
            if elem not in node.children:
                node.children[elem] = self._trienode_factory(elem)  # If an elem is not found, create a new in the trie
            node = node.children[elem]  # Iterate recursively!

        # Increment the counter to indicate that we see this word once more
        node.counter += 1

        # Mark the end of a word
        node.is_end = True

    def _dfs_depth(self, node, prefix, depth: int) -> List:
        """Depth-first traversal of the trie

        Args:
            - node: the node to start with
            - prefix: the current prefix, for tracing a word while traversing the trie
        """

        output = []
        if len(prefix) == depth-1 and node.is_end:
            return [(prefix + [node.word_id], node.counter)]
        elif len(prefix) < depth-1:
            for child in node.children.values():
                output.extend(self._dfs_depth(child, prefix + [node.word_id], depth))

        return output

    def query_fixed_depth(self, prefix, depth: int):
        """Given an input (a prefix), retrieve all words stored in
        the trie with that prefix, sort the words by the number of
        times they have been inserted
        """
        node = self.root

        # Check if the prefix is in the trie
        for word_id in prefix:
            if word_id in node.children:
                node = node.children[word_id]
            else:
                return []  # cannot found the prefix, return empty list

        # Traverse the trie to get all candidates with prefix
        output = self._dfs_depth(node, prefix[:-1], depth)

        # Sort the results in reverse order and return
        return sorted(output, key=lambda x: (-x[1], x[0]))


def main(inp_fn, out_fn):
    trie = Trie()
    with open(inp_fn) as infile:
        csv_reader = csv.reader(infile)
        for text, word in csv_reader:
            if len(word) > 0:
                word = list(map(int, word.split(' ')))
                if len(word) <= 10:
                    trie.insert(word)

    # output_2 = trie.query_fixed_depth([9939], 3)
    with open(out_fn, 'wb') as outfile:
        pickle.dump(trie, outfile)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Paramteres: input_file.csv output_file.pickle')
        exit(1)
    main(sys.argv[1], sys.argv[2])
