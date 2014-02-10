Introduction
============

A simple tool to generate depth streams from animated characters.
These can then be used for testing purposes.

Requires
- Visual Studio Express 2010 C#
- XNA 4.0
- XBox gamepad


Usage
=====

1. Start the app and use the gamepad's thumb sticks and triggers to
   move the virtual camera.
2. Press 'A' once to start and once stop recording.
3. A file dialog pops up. Choose where you want to save the recording to.
4. Quit the app with 'Back' or 'ESC'.


.depth file format version 1:
===============================

Bytes		Type		Default value		Meaning
---------------------------------------------------------------
15		string		KPPL raw depth\n	magic ascii header
4		int		1			version number
4		int					number of frames (#frames)

// the following two entries are repeated [#frames] times:

64		float3x4				view matrix (R*T) of the following frame, row major
648*480*2	short					depth in mm (0 means invalid measurement), row major


Notes
=====

The camera's projection matrix is defined by the Kinect's lens parameters:
- principal point: (319.5, 239.5)
- focal length: (585, 585)

Kinecting People (KPPL) uses throughout:
- row vectors and column matrices (i.e. positionView = positionWorld * matrixView)
- a right-handed coordinate system (i.e. smaller Z = further away in camera space)