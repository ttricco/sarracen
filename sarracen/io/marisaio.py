import numpy as np
import pandas as pd
from enum import IntEnum

class MarisaFile():

	class MARISAIO_TAGS(IntEnum):
		startheader = 0
		endheader = 1
		startslice = 2
		endslice = 3
		luascript = 4
		n = 5
		ndim = 6
		dt = 7
		endt = 8
		t = 9
		gamma = 10
		G = 11
		epssq = 12
		munaught = 13
		viscbeta = 14
		alphaviscMax = 15
		alphaviscMin = 16
		alphamagMax = 17
		alphamagMin = 18
		alphauMax = 19
		alphauMin = 20
		integratorType = 21
		boundaryType = 22
		boundaryxl = 23
		boundaryxh = 24
		boundaryyl = 25
		boundaryyh = 26
		boundaryzl = 27
		boundaryzh = 28
		totalge = 29
		totalke = 30
		totalbe = 31
		totalue = 32
		totalpsie = 33
		totals = 34
		totalmomentum = 35
		rx = 36
		ry = 37
		rz = 38
		vx = 39
		vy = 40
		vz = 41
		bx = 42
		by = 43
		bz = 44
		m = 45
		P = 46
		rho = 47
		ue = 48
		ke = 49
		s = 50
		h = 51
		alpha = 52
		alphamag = 53
		alphau = 54
		fgx = 55
		fgy = 56
		fgz = 57
		fhx = 58
		fhy = 59
		fhz = 60
		fbx = 61
		fby = 62
		fbz = 63
		dbx = 64
		dby = 65
		dbz = 66
		ds = 67
		due = 68
		drhodh = 69
		dalpha = 70
		dalphamag = 71
		dalphau = 72
		curlv = 73
		curlvx = 74
		curlvy = 75
		curlvz = 76
		divv = 77
		curlb = 78
		curlbx = 79
		curlby = 80
		curlbz = 81
		divb = 82
		divbsymm = 83
		euleralpha = 84
		eulerbeta = 85
		vsigmax = 86
		psi = 87
		dpsi = 88
		id = 89
		nneigh = 90
		ax = 91
		ay = 92
		az = 93
		diva = 94
		psiheat = 95
		viscvsigmax = 96
		dataset = 97
		dustfrac = 98
		colour = 99

	# 0: sets the reference point at the beginning of the file
	# 1: sets the reference point at the current file position
	# 2: sets the reference point at the end of the file

	def __init__(self, filename):
		self.fp = open(filename, "rb")
		self.ntags = 0
		self.tags = 0
		self.sizes = 0
		self.offsets = 0
		self._parse_tags()
		self.Ns = self._count_slices()
		self.data = 0

		self.params = dict()


	def _read_capture_pattern(self):
		if self.fp.read(7) != b"marisa\x00":
			return 0
		version_major = int.from_bytes(self.fp.read(4), byteorder='little')
		version_minor = int.from_bytes(self.fp.read(4), byteorder='little')
		return 1


	def _read_tag(self):
		tag = int.from_bytes(self.fp.read(4), byteorder='little')
		return tag


	def _read_data(self):
		size = int.from_bytes(self.fp.read(4), byteorder='little')
		data = self.fp.read(size)
		return data


	def _parse_tags(self):
		currentpos = self.fp.tell()

		self.fp.seek(0, 2)
		endpos = self.fp.tell()

		self.fp.seek(0, 0)

		if not self._read_capture_pattern():
			raise AssertionError("Capture pattern not present. Is this a valid data file?")

		tags = []
		offsets = []
		sizes = []

		while (self.fp.tell() < endpos):
			tag = self._read_tag()
			offset = self.fp.tell()
			size = int.from_bytes(self.fp.read(4), byteorder='little')
			self.fp.seek(size, 1)

			tags.append(tag)
			sizes.append(size)
			offsets.append(offset)

		self.tags = np.array(tags)
		self.offsets = np.array(offsets)
		self.sizes = np.array(sizes)

		self.fp.seek(currentpos, 0)


	def _count_slices(self):
		Ns = 0
		for tag in self.tags:
			if tag == self.MARISAIO_TAGS.startslice:
				Ns = Ns + 1
		return Ns

	def read(self, slicenumber):

		if slicenumber < 0:
			slicenumber = self.Ns + slicenumber

		if (slicenumber < 0 or slicenumber >= self.Ns):
			raise ValueError("Invalid slice number")

		verbose = False

		if verbose:
			print("seeking slice number: " + str(slicenumber))

		# find slice

		slicecounter = -1
		slicetagID = -1
		startheaderread = False
		endheaderread = False

		for i in range(len(self.tags)):

			if self.tags[i] == self.MARISAIO_TAGS.startslice:

				slicecounter = slicecounter + 1

				if (slicecounter == slicenumber):
					slicetagID = i

				if verbose:
					print("found slice: ")
					print("   slicecounter: " + str(slicecounter))
					print("   slicetagID: "  + str(slicetagID))

			# all the header bit we will ignore

		if (slicetagID == -1 or slicecounter < slicenumber):
			raise ValueError("Slice number not found")

		columns = []

		done = False
		endsliceread = False
		i = slicetagID
		n = 0
		self.data = pd.DataFrame()
		while (not done):
			self.fp.seek(self.offsets[i], 0)

			tag = self.tags[i]

			if (tag == self.MARISAIO_TAGS.endslice):
				endsliceread = True
				done = True
				i = len(self.tags) + 1
			elif (tag == self.MARISAIO_TAGS.n):
				self.params['n'] = np.frombuffer(self._read_data(), dtype=np.int32)[0]
			elif (tag == self.MARISAIO_TAGS.t):
				self.params['t'] = np.frombuffer(self._read_data(), dtype=np.float64)[0]
			elif (tag == self.MARISAIO_TAGS.totalge):
				self.params['totalge'] = np.frombuffer(self._read_data(), dtype=np.float64)[0]
			elif (tag == self.MARISAIO_TAGS.totalke):
				self.params['totalke'] = np.frombuffer(self._read_data(), dtype=np.float64)[0]
			elif (tag == self.MARISAIO_TAGS.totalue):
				self.params['totalue'] = np.frombuffer(self._read_data(), dtype=np.float64)[0]
			elif (tag == self.MARISAIO_TAGS.totalbe):
				self.params['totalbe'] = np.frombuffer(self._read_data(), dtype=np.float64)[0]
			elif (tag == self.MARISAIO_TAGS.totalpsie):
				self.params['totalpsie'] = np.frombuffer(self._read_data(), dtype=np.float64)[0]
			elif (tag == self.MARISAIO_TAGS.totalmomentum):
				self.params['totalmomentum'] = np.frombuffer(self._read_data(), dtype=np.float64)[0]

			if (tag == self.MARISAIO_TAGS.rx):
				self.data[self.MARISAIO_TAGS.rx.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.ry):
				self.data[self.MARISAIO_TAGS.ry.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.rz):
				self.data[self.MARISAIO_TAGS.rz.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.vx):
				self.data[self.MARISAIO_TAGS.vx.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.vy):
				self.data[self.MARISAIO_TAGS.vy.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.vz):
				self.data[self.MARISAIO_TAGS.vz.name] = np.frombuffer(self._read_data(), dtype=np.double)

			if (tag == self.MARISAIO_TAGS.bx):
				self.data[self.MARISAIO_TAGS.bx.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.by):
				self.data[self.MARISAIO_TAGS.by.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.bz):
				self.data[self.MARISAIO_TAGS.bz.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.psi):
				self.data[self.MARISAIO_TAGS.psi.name] = np.frombuffer(self._read_data(), dtype=np.double)

			if (tag == self.MARISAIO_TAGS.euleralpha):
				self.data[self.MARISAIO_TAGS.rx.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.ax):
				self.data[self.MARISAIO_TAGS.ax.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.ay):
				self.data[self.MARISAIO_TAGS.ay.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.az):
				self.data[self.MARISAIO_TAGS.az.name] = np.frombuffer(self._read_data(), dtype=np.double)


			if (tag == self.MARISAIO_TAGS.m):
				self.data[self.MARISAIO_TAGS.m.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.h):
				self.data[self.MARISAIO_TAGS.h.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.rho):
				self.data[self.MARISAIO_TAGS.rho.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.P):
				self.data[self.MARISAIO_TAGS.P.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.ue):
				self.data[self.MARISAIO_TAGS.ue.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.ke):
				self.data[self.MARISAIO_TAGS.ke.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.s):
				self.data[self.MARISAIO_TAGS.s.name] = np.frombuffer(self._read_data(), dtype=np.double)

			if (tag == self.MARISAIO_TAGS.alpha):
				self.data[self.MARISAIO_TAGS.alpha.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.alphamag):
				self.data[self.MARISAIO_TAGS.alphamag.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.alphau):
				self.data[self.MARISAIO_TAGS.alphau.name] = np.frombuffer(self._read_data(), dtype=np.double)

			if (tag == self.MARISAIO_TAGS.divv):
				self.data[self.MARISAIO_TAGS.divv.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.divb):
				self.data[self.MARISAIO_TAGS.divb.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.divbsymm):
				self.data[self.MARISAIO_TAGS.divbsymm.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.curlb):
				self.data[self.MARISAIO_TAGS.curlb.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.dustfrac):
				self.data[self.MARISAIO_TAGS.dustfrac.name] = np.frombuffer(self._read_data(), dtype=np.double)
			if (tag == self.MARISAIO_TAGS.colour):
				self.data[self.MARISAIO_TAGS.colour.name] = np.frombuffer(self._read_data(), dtype=np.double)

			i = i + 1

		if (not endsliceread):
			raise AssertionError("Did not find end of slice tag")
