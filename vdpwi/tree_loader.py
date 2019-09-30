import os
import hashlib
import pickle
import argparse
import os

class Tree(object):
	# This class is implemented by https://github.com/dasguptar/treelstm.pytorch/blob/master/treelstm/tree.py
	def __init__(self):
		self.parent = None
		self.num_children = 0
		self.children = list()

	def add_child(self, child):
		child.parent = self
		self.num_children += 1
		self.children.append(child)

	def size(self):
		if getattr(self, '_size'):
			return self._size
		count = 1
		for i in range(self.num_children):
			count += self.children[i].size()
		self._size = count
		return self._size

	def depth(self):
		if getattr(self, '_depth'):
			return self._depth
		count = 0
		if self.num_children > 0:
			for i in range(self.num_children):
				child_depth = self.children[i].depth()
				if child_depth > count:
					count = child_depth
			count += 1
		self._depth = count
		return self._depth


def read_tree(line):
	parents = list(map(int, line.split()))
	trees = dict()
	root = None
	for i in range(1, len(parents) + 1):
		if i - 1 not in trees.keys() and parents[i - 1] != -1:
			idx = i
			prev = None
			while True:
				parent = parents[idx - 1]
				if parent == -1:
					break
				tree = Tree()
				if prev is not None:
					tree.add_child(prev)
				trees[idx - 1] = tree
				tree.idx = idx - 1
				if parent - 1 in trees.keys():
					trees[parent - 1].add_child(tree)
					break
				elif parent == 0:
					root = tree
					break
				else:
					prev = tree
					idx = parent
	return root

def main():
	parser = argparse.ArgumentParser(description="Generate Tree File")
	parser.add_argument("--dataset_path", help="dataset path")
	parser.add_argument("--dataset", default="sick")
	parser.add_argument("--output_path", default="./")
	args = parser.parse_args()

	train_path = os.path.join(args.dataset_path, "train/")
	dev_path = os.path.join(args.dataset_path, "dev/")
	test_path = os.path.join(args.dataset_path, "test/")
	file_path = [train_path, test_path, dev_path]

	toks_to_parent = dict()
	
	for _dir in file_path:
		os.chdir(_dir)
		f_toks = open("a.toks").readlines()
		f_parent = open("a.parents").readlines()
		for i in range(len(f_toks)):
			tok_line = f_toks[i].rstrip(".\n")
			code = hashlib.sha224(tok_line.encode()).hexdigest()
			if f_toks[i][-2] == ".":
				reduced_fparent = ' '.join(f_parent[i].strip("\n").split()[:-1])
				toks_to_parent[code] = read_tree(reduced_fparent)
			else:
				toks_to_parent[code] = read_tree(f_parent[i].strip("\n"))

		f_toks_b = open("b.toks").readlines()
		f_parent_b = open("b.parents").readlines()
		for i in range(len(f_toks_b)):
			tok_line_b = f_toks_b[i].rstrip(".\n")
			code = hashlib.sha224(tok_line_b.encode()).hexdigest()
			try:
				if f_toks_b[i][-2] == ".":
					reduced_fparent_b = ' '.join(f_parent_b[i].strip("\n").split()[:-1])
					toks_to_parent[code] = read_tree(reduced_fparent_b)
				else:
					toks_to_parent[code] = read_tree(f_parent_b[i].strip("\n"))
			except IndexError:
				print("f_toks: ", f_toks_b[i])
				print("f_parent: ", f_parent_b[i])
				print("error!")

	tree_file = args.dataset + "_toks_tree.pkl"
	output_file = os.path.join(args.output_path, tree_file)
	with open(output_file, "wb") as f:
		pickle.dump(toks_to_parent, f)


if __name__ == "__main__":
	main()

