#pragma once



namespace svcu {

void segmented_inclusive_scan
(
	unsigned const * data, unsigned size, 
	unsigned segmentSize,
	
	unsigned * out
);

void segmented_exclusive_scan
(
	unsigned const * data, unsigned size, 
	unsigned segmentSize,
	
	unsigned * out
);

}