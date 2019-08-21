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
		self.boundsMerged = False

	def print(self):
		print ("\t\tCharBound: {}, {}, {}, {}, {}, {}".format(self.c, self.l, self.t, self.r, self.b, self.boundsMerged))


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
			try:
				self.charBounds.remove(charBound)
			except:
				pass
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

	def getAllCharsInXRange(self, l, r):
		chars = list()
		for cb in self.charBounds:
			if cb.l >= l and cb.r <= r:
				chars.append(cb)
		return chars

	def orderChars(self):
		self.charBounds.sort(key=lambda x: x.l)

	def handlePowers(self):
		y_offset = CHAR_SIZE / 3 
		for i in range(1, len(self.charBounds)):
			curChar = self.charBounds[i - 1]
			nextChar = self.charBounds[i]
			if 'frac' not in curChar.c and 'frac' not in nextChar.c and (nextChar.b >= curChar.t - y_offset) and nextChar.b < curChar.t + y_offset and nextChar.t < curChar.t:
				nextChar.c = ' {\pow ' + nextChar.c + '} '
			elif '{\pow' in curChar.c:
				if nextChar.t < curChar.b - y_offset and nextChar.b < curChar.b + y_offset:
					nextChar.c = curChar.c[:-2] + nextChar.c + '} '
					curChar.c = ''
					nextChar.l = curChar.l
					nextChar.t = min(curChar.t, nextChar.t)
					nextChar.b = max(curChar.b, nextChar.b)
			

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
					charBoundsInRange = self.getAllCharsInXRange(cb.l, cb.r)
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
		self.handlePowers()
	
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

def setVerticalThres(charBounds):
	global VERTICAL_THRES
	global HORIZ_THRES
	global CHAR_SIZE
	sum = 0 
	for bound in charBounds:
		sum += bound.b - bound.t
	CHAR_SIZE = math.ceil(sum / (len(charBounds)))
	VERTICAL_THRES = (CHAR_SIZE / 3) + 1 
	HORIZ_THRES = math.ceil(VERTICAL_THRES / 2) + 1
	print ("\nCHAR_SIZE: {}, VERTICAL_THRES: {}, HORIZ_THRES: {}".format(CHAR_SIZE, VERTICAL_THRES, HORIZ_THRES))


def checkIfNextCharInSameLine(curWord, curChar, nextChar):
	sameLine = (curWord.t >= nextChar.t and curWord.t <= nextChar.b) or (curWord.b >= nextChar.t and curWord.b <= nextChar.b) or (nextChar.t >= curWord.t and nextChar.t <= curWord.b) or (nextChar.t >= curWord.t and nextChar.t <= curWord.b)
	"""
	# See if next char is power
	if 'frac' not in curChar.c and 'frac' not in nextChar.c and nextChar.b >= curChar.t and nextChar.b < curChar.t + (curChar.b - curChar.t) / 3:
		nextChar.c = '^' + nextChar.c
	"""
	return sameLine

def mergeBounds(charBounds):
	# Assigning line index based on y co-ordinate
	charBounds.sort(key=lambda cb: cb.t)
	lineIndex = 0
	yIndicedLineBounds = defaultdict(list)
	yIndicedLineBounds[lineIndex].append(charBounds[0])	
	for cb in charBounds[1:]:
		if cb.t - yIndicedLineBounds[lineIndex][-1].b > VERTICAL_THRES:
			lineIndex += 1
		yIndicedLineBounds[lineIndex].append(cb)
	

	lineBounds = list()
	# Assigning word index based on x co-ordinate
	for ind, cbs in yIndicedLineBounds.items():
		lb = LineBound()
		cbs.sort(key=lambda cb: cb.l)
		for i in range(len(cbs)):
			if cbs[i].boundsMerged:
				continue
			wb = WordBound()
			wb.addCharBound(cbs[i])
			cbs[i].boundsMerged = True
			for j in range(i + 1, len(cbs)):
				if cbs[j].boundsMerged:
					continue
				if cbs[j].l - wb.r <= HORIZ_THRES and checkIfNextCharInSameLine(wb, wb.charBounds[-1], cbs[j]):
					wb.addCharBound(cbs[j])
					cbs[j].boundsMerged = True
			lb.addWordBound(wb)
		lineBounds.append(lb)

	for lb in lineBounds:
		lb.print()			
	print ("\n\nOCR Output:\n")
	for lb in lineBounds:
		lb.latex()			
	print ()
	
def getLatex(rectangles):
	charBounds = list()
	#for r in rectangles.values():
	for r in rectangles:
		if len(r.prediction) > 0:
			charBounds.append(CharBound(str(r.prediction), max(1, r.left), max(1, r.top), r.right, r.bottom)) 
	if len(charBounds) > 0:
		setVerticalThres(charBounds)
		mergeBounds(charBounds)
	else:
		print ("Could not read charBounds")
