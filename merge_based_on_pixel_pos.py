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
		self.has_fraction = False

	def addCharBound(self, charBound):
		# If character is the line between numerator and denominator, it should be at least 1.8 times char_size
		if charBound.c == '-' and (charBound.r - charBound.l) > 1.8 * CHAR_SIZE:
			charBound.c = 'frac'
		if charBound.c == 'frac':
			self.has_fraction = True
		if not self.charBounds or not self.charBounds[-1] == '=':
			self.charBounds.append(charBound)
		self.calcWordBounds()

	def removeCharBounds(self, charBounds):
		for charBound in charBounds:
			try:
				self.charBounds.remove(charBound)
			except ValueError:
				pass
		self.calcWordBounds()

	def calcWordBounds(self):
		self.l = min([charBound.l for charBound in self.charBounds])
		self.t = min([charBound.t for charBound in self.charBounds])
		self.r = max([charBound.r for charBound in self.charBounds])
		self.b = max([charBound.b for charBound in self.charBounds])

	def get_all_chars_in_x_range(self, l, r):
		chars = list()
		for cb in self.charBounds:
			if cb.l >= l and cb.r <= r:
				chars.append(cb)
		return chars

	def latex(self):
		chars = list()
		if self.has_fraction and len(self.charBounds) > 1:
			for cb in self.charBounds:
				if cb.c == 'frac':
					if cb.r - cb.l == self.r - self.l and cb.b - cb.t == self.b - self.t and len(self.charBounds) > 1:
						continue
					charBoundsInRange = self.get_all_chars_in_x_range(cb.l, cb.r)
					if charBoundsInRange:
						numerator = list()
						denominator = list()
						for cbir in charBoundsInRange:
							if cbir.b <= cb.t:	
								numerator.append(cbir.c)		
							else:
								denominator.append(cbir.c)
						fractionLatex = "frac({} / {})".format("".join(numerator), "".join(denominator))
						fractionCb = CharBound(fractionLatex, cb.l, cb.t, cb.r, cb.b)
						self.addCharBound(fractionCb)
					charBoundsInRange.append(cb)
					self.removeCharBounds(charBoundsInRange)
		for cb in self.charBounds:
			# If two consecutive '-', replace with '='
			if chars and chars[-1] == '-' and cb.c == '-':
				chars[-1] = '='
				continue
			if cb.c == 'frac' and cb.r - cb.l == self.r - self.l and cb.b - cb.t == self.b - self.t and len(self.charBounds) > 1:
				continue
			chars.append(cb.c)
		return "".join(chars)

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
			words.append(wb.latex())
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

	for lb in line_bounds:
		lb.print()
	print ("\n\nOCR Output:\n")
	for lb in line_bounds:
		lb.latex()			
	print ()
	
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
