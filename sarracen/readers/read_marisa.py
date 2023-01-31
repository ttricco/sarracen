import numpy as np
import pandas as pd
from enum import IntEnum

from ..sarracen_dataframe import SarracenDataFrame


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

def _marisa_read_capture_pattern(fp):
	if fp.read(7) != b"marisa\x00":
		return 0
	version_major = int.from_bytes(fp.read(4), byteorder='little')
	version_minor = int.from_bytes(fp.read(4), byteorder='little')
	return 1


def _marisa_read_tag(fp):
	tag = int.from_bytes(fp.read(4), byteorder='little')
	return tag


def _marisa_read_data(fp):
	size = int.from_bytes(fp.read(4), byteorder='little')
	data = fp.read(size)
	return data


def _marisa_parse_tags(fp):
	currentpos = fp.tell()

	fp.seek(0, 2)
	endpos = fp.tell()

	fp.seek(0, 0)

	if not _marisa_read_capture_pattern(fp):
		raise AssertionError("Capture pattern not present. Is this a valid data file?")

	tags = []
	offsets = []
	sizes = []

	while (fp.tell() < endpos):
		tag = _marisa_read_tag(fp)
		offset = fp.tell()
		size = int.from_bytes(fp.read(4), byteorder='little')
		fp.seek(size, 1)

		tags.append(tag)
		sizes.append(size)
		offsets.append(offset)

	tags = np.array(tags)
	offsets = np.array(offsets)
	sizes = np.array(sizes)

	fp.seek(currentpos, 0)

	return tags, offsets

def _marisa_count_slices(fp, tags):
	Ns = 0
	for tag in tags:
		if tag == MARISAIO_TAGS.startslice:
			Ns = Ns + 1
	return Ns



def read_marisa(filename : str,
                slicenumber: int = 0) -> SarracenDataFrame:
	""" Read data from a Marisa dump file.

	Parameters
	----------
	filename : str
	    Name of the file to be loaded.
	slicenumber : int, default=0
	    The time slice to read from the data file.

	Returns
	-------
	SarracenDataFrame
	"""
	fp = open(filename, "rb")
	ntags = 0
	tags = 0
	sizes = 0
	offsets = 0
	tags, offsets = _marisa_parse_tags(fp)
	Ns = _marisa_count_slices(fp, tags)

	if slicenumber < 0:
		slicenumber = Ns + slicenumber

	if (slicenumber < 0 or slicenumber >= Ns):
		raise ValueError("Invalid slice number")

	verbose = False

	if verbose:
		print("seeking slice number: " + str(slicenumber))

	# find slice

	slicecounter = -1
	slicetagID = -1
	startheaderread = False
	endheaderread = False

	for i in range(len(tags)):

		if tags[i] == MARISAIO_TAGS.startslice:

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
	df = pd.DataFrame()
	params = dict()
	while (not done):
		fp.seek(offsets[i], 0)

		tag = tags[i]

		if (tag == MARISAIO_TAGS.endslice):
			endsliceread = True
			done = True
			i = len(tags) + 1
		elif (tag == MARISAIO_TAGS.n):
			params['n'] = np.frombuffer(_marisa_read_data(fp), dtype=np.int32)[0]
		elif (tag == MARISAIO_TAGS.t):
			params['t'] = np.frombuffer(_marisa_read_data(fp), dtype=np.float64)[0]
		elif (tag == MARISAIO_TAGS.totalge):
			params['totalge'] = np.frombuffer(_marisa_read_data(fp), dtype=np.float64)[0]
		elif (tag == MARISAIO_TAGS.totalke):
			params['totalke'] = np.frombuffer(_marisa_read_data(fp), dtype=np.float64)[0]
		elif (tag == MARISAIO_TAGS.totalue):
			params['totalue'] = np.frombuffer(_marisa_read_data(fp), dtype=np.float64)[0]
		elif (tag == MARISAIO_TAGS.totalbe):
			params['totalbe'] = np.frombuffer(_marisa_read_data(fp), dtype=np.float64)[0]
		elif (tag == MARISAIO_TAGS.totalpsie):
			params['totalpsie'] = np.frombuffer(_marisa_read_data(fp), dtype=np.float64)[0]
		elif (tag == MARISAIO_TAGS.totalmomentum):
			params['totalmomentum'] = np.frombuffer(_marisa_read_data(fp), dtype=np.float64)[0]

		if (tag == MARISAIO_TAGS.rx):
			df[MARISAIO_TAGS.rx.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.ry):
			df[MARISAIO_TAGS.ry.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.rz):
			df[MARISAIO_TAGS.rz.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.vx):
			df[MARISAIO_TAGS.vx.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.vy):
			df[MARISAIO_TAGS.vy.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.vz):
			df[MARISAIO_TAGS.vz.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)

		if (tag == MARISAIO_TAGS.bx):
			df[MARISAIO_TAGS.bx.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.by):
			df[MARISAIO_TAGS.by.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.bz):
			df[MARISAIO_TAGS.bz.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.psi):
			df[MARISAIO_TAGS.psi.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)

		if (tag == MARISAIO_TAGS.euleralpha):
			df[MARISAIO_TAGS.rx.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.ax):
			df[MARISAIO_TAGS.ax.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.ay):
			df[MARISAIO_TAGS.ay.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.az):
			df[MARISAIO_TAGS.az.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)


		if (tag == MARISAIO_TAGS.m):
			df[MARISAIO_TAGS.m.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.h):
			df[MARISAIO_TAGS.h.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.rho):
			df[MARISAIO_TAGS.rho.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.P):
			df[MARISAIO_TAGS.P.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.ue):
			df[MARISAIO_TAGS.ue.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.ke):
			df[MARISAIO_TAGS.ke.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.s):
			df[MARISAIO_TAGS.s.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)

		if (tag == MARISAIO_TAGS.alpha):
			df[MARISAIO_TAGS.alpha.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.alphamag):
			df[MARISAIO_TAGS.alphamag.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.alphau):
			df[MARISAIO_TAGS.alphau.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)

		if (tag == MARISAIO_TAGS.divv):
			df[MARISAIO_TAGS.divv.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.divb):
			df[MARISAIO_TAGS.divb.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.divbsymm):
			df[MARISAIO_TAGS.divbsymm.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.curlb):
			df[MARISAIO_TAGS.curlb.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.dustfrac):
			df[MARISAIO_TAGS.dustfrac.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)
		if (tag == MARISAIO_TAGS.colour):
			df[MARISAIO_TAGS.colour.name] = np.frombuffer(_marisa_read_data(fp), dtype=np.double)

		i = i + 1

	if (not endsliceread):
		raise AssertionError("Did not find end of slice tag")

	return SarracenDataFrame(df, params=params)
