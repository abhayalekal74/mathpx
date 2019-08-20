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
		self.bounds_merged = False

	def print(self):
		print ("\t\tCharBound: {}, {}, {}, {}, {}, {}".format(self.c, self.l, self.t, self.r, self.b, self.bounds_merged))


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
		self.charBounds.append(charBound)
		self.calcWordBounds()

	def removeCharBounds(self, charBounds):
		for charBound in charBounds:
			self.charBounds.remove(charBound)
		self.calcWordBounds()

	def hasChars(self):
		return len(self.charBounds) > 0

	def calcWordBounds(self):
		try:
			self.l = min([charBound.l for charBound in self.charBounds])
			self.t = min([charBound.t for charBound in self.charBounds])
			self.r = max([charBound.r for charBound in self.charBounds])
			self.b = max([charBound.b for charBound in self.charBounds])
		except:
			self.l, self.t, self.r, self.b = 0, 0, 0, 0

	def get_all_chars_in_x_range(self, l, r):
		chars = list()
		for cb in self.charBounds:
			if cb.l >= l and cb.r <= r:
				chars.append(cb)
		return chars

	def orderChars(self):
		self.charBounds.sort(key=lambda x: x.l)

	def mergeCharFractions(self):
		removeChars = list()
		if self.has_fraction:
			for cb in self.charBounds:
				if cb.c == 'frac':
					# Remove false fraction divider
					if cb.r - cb.l == self.r - self.l and cb.b - cb.t == self.b - self.t and len(self.charBounds) > 1:
						removeChars.append(cb)
						continue
					# Replace all chars within fraction range with fraction latex
					charBoundsInRange = self.get_all_chars_in_x_range(cb.l, cb.r)
					if charBoundsInRange:
						numerator = WordBound() 
						denominator = WordBound()
						for cbir in charBoundsInRange:
							if cbir == cb: # Should not include fraction in numerator or denominator
								continue
							if cbir.b <= cb.t:	
								numerator.addCharBound(cbir)		
							else:
								denominator.addCharBound(cbir)
						if numerator.hasChars() or denominator.hasChars():
							numerator.orderChars()
							denominator.orderChars()
							fractionLatex = "frac({} / {})".format(numerator.latex(), denominator.latex())
							fractionCb = CharBound(fractionLatex, cb.l, cb.t, cb.r, cb.b)
							self.addCharBound(fractionCb)
					removeChars += charBoundsInRange
		self.removeCharBounds(removeChars)
		self.orderChars()
	
	def latex(self):
		chars = list()
		for cb in self.charBounds:
			# If two consecutive '-', replace with '='
			if chars and chars[-1] == '-' and cb.c == '-':
				chars[-1] = '='
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

	def orderWords(self):
		self.wordBounds.sort(key=lambda x: x.l)

	def print(self):
		print ("LineBound:")
		for wb in self.wordBounds:
			wb.print()

	def latex(self):
		words = list()
		for wb in self.wordBounds:
			wb.mergeCharFractions()
		self.orderWords()
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


def check_if_next_char_in_same_line(cur_word, cur_char, next_char):
	sameLine = (cur_word.t >= next_char.t and cur_word.t <= next_char.b) or (cur_word.b >= next_char.t and cur_word.b <= next_char.b) or (next_char.t >= cur_word.t and next_char.t <= cur_word.b) or (next_char.t >= cur_word.t and next_char.t <= cur_word.b)
	"""
	# See if next char is power
	if 'frac' not in cur_char.c and 'frac' not in next_char.c and next_char.b >= cur_char.t and next_char.b < cur_char.t + (cur_char.b - cur_char.t) / 3:
		next_char.c = '^' + next_char.c
	"""
	return sameLine

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
		for i in range(len(cbs)):
			if cbs[i].bounds_merged:
				continue
			wb = WordBound()
			wb.addCharBound(cbs[i])
			cbs[i].bounds_merged = True
			for j in range(i + 1, len(cbs)):
				if cbs[j].bounds_merged:
					continue
				if cbs[j].l - wb.r <= HORIZ_THRES and check_if_next_char_in_same_line(wb, wb.charBounds[-1], cbs[j]):
					wb.addCharBound(cbs[j])
					cbs[j].bounds_merged = True
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
