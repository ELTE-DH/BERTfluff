# from https://albertauyeung.github.io/2020/06/15/python-trie.html
import csv
import pickle
from typing import List


class TrieNode:
    """A node in the trie structure"""

    def __init__(self, word_id):
        # the character stored in this node
        self.word_id = word_id

        # whether this can be the end of a word
        self.is_end = False

        # a counter indicating how many times a word is inserted
        # (if this node's is_end is True)
        self.counter = 0

        # a dictionary of child nodes
        # keys are characters, values are nodes
        self.children = {}

    def get_repr_string(self, tab: str = '\t', m: int = 0):
        ret = ''.join(tab for _ in range(m))
        ret += str(self.word_id)

        ret += f': {self.counter}'
        if len(self.children) > 0:
            ret += '\n{0}{1}'.format(''.join(node.get_repr_string(tab, m + 1) for node in self.children.values()),
                                     ''.join(tab for _ in range(m)))
        ret += '\n'
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
        self.output = []

    def insert(self, word_ids: List[int]):
        """Insert a word into the trie"""
        node = self.root

        # Loop through each character in the word
        # Check if there is no child containing the character, create a new child for the current node
        for word_id in word_ids:
            if word_id in node.children:
                node = node.children[word_id]
            else:
                # If a character is not found,
                # create a new node in the trie
                new_node = TrieNode(word_id)
                node.children[word_id] = new_node
                node = new_node

        # Mark the end of a word
        node.is_end = True

        # Increment the counter to indicate that we see this word once more
        node.counter += 1

    def dfs(self, node, prefix):
        """Depth-first traversal of the trie

        Args:
            - node: the node to start with
            - prefix: the current prefix, for tracing a
                word while traversing the trie
        """
        if node.is_end:
            self.output.append((prefix + [node.word_id], node.counter))

        for child in node.children.values():
            self.dfs(child, prefix + [node.word_id])

    def dfs_depth(self, node, prefix, depth: int):
        """Depth-first traversal of the trie

        Args:
            - node: the node to start with
            - prefix: the current prefix, for tracing a
                word while traversing the trie
        """
        if len(prefix) == depth-1:
            if node.is_end:
                self.output.append((prefix + [node.word_id], node.counter))
        else:
            for child in node.children.values():
                self.dfs_depth(child, prefix + [node.word_id], depth)

    def query_fixed_depth(self, prefix, depth: int):
        """Given an input (a prefix), retrieve all words stored in
        the trie with that prefix, sort the words by the number of
        times they have been inserted
        """
        # Use a variable within the class to keep all possible outputs
        # As there can be more than one word with such prefix
        self.output = []
        node = self.root

        # Check if the prefix is in the trie
        for word_id in prefix:
            if word_id in node.children:
                node = node.children[word_id]
            else:
                # cannot found the prefix, return empty list
                return []

        # Traverse the trie to get all candidates
        self.dfs_depth(node, prefix[:-1], depth)

        # Sort the results in reverse order and return
        return self.output

    def query(self, prefix: List[int]):
        """Given an input (a prefix), retrieve all words stored in
        the trie with that prefix, sort the words by the number of
        times they have been inserted
        """
        # Use a variable within the class to keep all possible outputs
        # As there can be more than one word with such prefix
        self.output = []
        node = self.root

        # Check if the prefix is in the trie
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                # cannot found the prefix, return empty list
                return []

        # Traverse the trie to get all candidates
        self.dfs(node, prefix[:-1])

        # Sort the results in reverse order and return
        return sorted(self.output, key=lambda x: x[1], reverse=True)


def main():
    trie = Trie()
    with open('../wordlist_tokenized.csv') as infile:
        csv_reader = csv.reader(infile)
        for text, word in csv_reader:
            if len(word) == 0:
                continue
            word = list(map(int, word.split(' ')))
            if len(word) > 10:
                continue
            trie.insert(word)

    # output_1 = trie.query([9939])
    # output_2 = trie.query_fixed_depth([9939], 3)
    with open('../models/trie_words.pickle', 'wb') as outfile:
        pickle.dump(trie, outfile)


if __name__ == '__main__':
    main()
