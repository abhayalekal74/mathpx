from collections import defaultdict
import tensorflow as tf
import math
import sys
import re

tf.logging.set_verbosity(tf.logging.FATAL)

VERTICAL_THRES = 1000 # lt is used, so keep the VERTICAL_THRES as desired_val + 1 
CHAR_SIZE = 100
HORIZ_THRES = 16


class CharBound:
	def __init__(self, c, l, t, r, b):
		self.c = c
		self.l =  l
		self.t = t
		self.r = r
		self.b = b
		self.bound_merged = False

	def print(self):
		print ("\t\tCharBound: {}, {}, {}, {}, {}, {}".format(self.c, self.l, self.t, self.r, self.b, self.bound_merged))


class WordBound:
	def __init__(self):
		self.charBounds = list()
		self.l = 0
		self.t = 0
		self.r = 0
		self.b = 0

	def addCharBound(self, charBound):
		self.charBounds.append(charBound)
		self.calcWordBounds()

	def calcWordBounds(self):
		self.l = min([charBound.l for charBound in self.charBounds])
		self.t = min([charBound.t for charBound in self.charBounds])
		self.r = max([charBound.r for charBound in self.charBounds])
		self.b = max([charBound.b for charBound in self.charBounds])

	def print(self):
		print ("\tWordBound: {}, {}, {}, {}".format(self.l, self.t, self.r, self.b))
		for cb in self.charBounds:
			cb.print()

class LineBound:
	def __init__(self):
		self.wordBounds = list()

	def addWordBound(self, wordBound):
		self.wordBounds.append(wordBound)
		self.orderWords()

	def orderWords(self):
		self.wordBounds.sort(key=lambda x: x.l)

	def print(self):
		print ("LineBound:")
		for wb in self.wordBounds:
			wb.print()

	def latex(self):
		words = list()
		for wb in self.wordBounds:
			chars = list()
			for cb in wb.charBounds:
				if cb.r - cb.l == wb.r - wb.l and cb.c == 'frac':
					continue
				if chars and chars[-1] == '-' and cb.c == '-':
					chars[-1] = '='
				elif cb.c != 'frac': # TODO change to else
					chars.append(cb.c)
			words.append("".join(chars))
		print(" ".join(words))

def set_vertical_thres(char_bounds):
	global VERTICAL_THRES
	global HORIZ_THRES
	global CHAR_SIZE
	sum = 0 
	for bound in char_bounds:
		sum += bound.b - bound.t
	CHAR_SIZE = math.ceil(sum / (len(char_bounds)))
	VERTICAL_THRES = (CHAR_SIZE / 3) + 1 
	HORIZ_THRES = math.ceil(VERTICAL_THRES / 2) + 1
	print ("\nCHAR_SIZE: {}, VERTICAL_THRES: {}, HORIZ_THRES: {}".format(CHAR_SIZE, VERTICAL_THRES, HORIZ_THRES))


def checkIfTwoCharactersAreInSameLine(cur_char, next_char):
	return (cur_char.t >= next_char.t and cur_char.t <= next_char.b) or (cur_char.b >= next_char.t and cur_char.b <= next_char.b) or (next_char.t >= cur_char.t and next_char.t <= cur_char.b) or (next_char.t >= cur_char.t and next_char.t <= cur_char.b)


def merge_bounds(char_bounds):
	# Assigning line index based on y co-ordinate
	char_bounds.sort(key=lambda cb: cb.t)
	line_index = 0
	y_indiced_line_bounds = defaultdict(list)
	y_indiced_line_bounds[line_index].append(char_bounds[0])	
	for cb in char_bounds[1:]:
		if cb.t - y_indiced_line_bounds[line_index][-1].b > VERTICAL_THRES:
			line_index += 1
		y_indiced_line_bounds[line_index].append(cb)
	

	line_bounds = list()
	# Assigning word index based on x co-ordinate
	for ind, cbs in y_indiced_line_bounds.items():
		lb = LineBound()
		cbs.sort(key=lambda cb: cb.l)
		visited = [0] * len(cbs)	
		for i in range(len(cbs)):
			if visited[i] == 1:
				continue
			wb = WordBound()
			wb.addCharBound(cbs[i])
			visited[i] = 1
			for j in range(i + 1, len(cbs)):
				if visited[j] == 1:
					continue
				if cbs[j].l - wb.r <= HORIZ_THRES and checkIfTwoCharactersAreInSameLine(wb, cbs[j]):
					wb.addCharBound(cbs[j])
					visited[j] = 1
			lb.addWordBound(wb)
		line_bounds.append(lb)
	print ("\n\nOCR Output:\n")
	for lb in line_bounds:
		lb.latex()			
	print ()


def merge_bounds_2(char_bounds):
	# Grouping all characters within a line	
	char_bottoms = list()
	for b in char_bounds:
		char_bottoms.append(b.b)
	char_bottoms_sorted = sorted(set(char_bottoms)) 

	char_bottoms = list()

	print (char_bottoms_sorted, len(char_bottoms_sorted))
	
	for i in range(len(char_bottoms_sorted)):
		if ((i + 1) < len(char_bottoms_sorted)) and ((char_bottoms_sorted[i + 1] - char_bottoms_sorted[i]) <= VERTICAL_THRES):
			continue
		char_bottoms.append(char_bottoms_sorted[i])
	print (char_bottoms, len(char_bottoms))

	char_bounds.sort(key=lambda x: x.t) 

	char_bot_counter, char_bounds_counter = 0, 0

	line_char_batches = list() 

	while char_bot_counter < len(char_bottoms) and char_bounds_counter < len(char_bounds):
		char_batch = list()
		while char_bounds_counter < len(char_bounds) and char_bottoms[char_bot_counter] >= char_bounds[char_bounds_counter].t:
			cur_char = char_bounds[char_bounds_counter]
			char_bounds_counter += 1
			if cur_char.c == "frac" and cur_char.r - cur_char.l <= CHAR_SIZE:
				continue
			char_batch.append(cur_char)
		if len(char_batch) > 0:
			char_batch.sort(key=lambda x: x.l)
			line_char_batches.append(char_batch)
		char_bot_counter += 1

	# Lines are already vertically ordered
	# Characters are already horizontally ordered
	# Grouping words within grouped lines
	line_bounds = list()
	for char_batch in line_char_batches:
		lb = LineBound()	
		i = 0
		while i < len(char_batch):
			char_bound = char_batch[i]
			i += 1
			if char_bound.bound_merged:
				continue
			wb = WordBound()
			char_bound.bound_merged = True
			wb.addCharBound(char_bound)
			j = i # i is already incremented
			while j < len(char_batch):
				add_char = char_batch[j]	
				if not add_char.bound_merged:
					if (add_char.l - char_bound.r < HORIZ_THRES): 
						if checkIfTwoCharactersAreInSameLine(char_bound, add_char):
							add_char.bound_merged = True
							wb.addCharBound(add_char)
					else: # Breaking only if next_char is farther away from horiz_thres. Since they are sorted by x, all subsequent chars are farther than horiz_thres
						break
				j += 1			
			lb.addWordBound(wb)
		line_bounds.append(lb)
	for lb in line_bounds:
		lb.print()
	for lb in line_bounds:
		lb.latex()			
	
def get_latex(rectangles):
	char_bounds = list()
	#for r in rectangles.values():
	for r in rectangles:
		if len(r.prediction) > 0:
			char_bounds.append(CharBound(str(r.prediction), max(1, r.left), max(1, r.top), r.right, r.bottom)) 
	"""
	for b in char_bounds:
		print ("{},{},{},{},{}".format(b.word, b.left, b.top, b.right, b.bottom))
	"""
	if len(char_bounds) > 0:
		set_vertical_thres(char_bounds)
		merge_bounds(char_bounds)
	else:
		print ("Could not read char_bounds")
