#pragma once



namespace svcu {

void segmented_reduce
(
	unsigned const * data, unsigned size,
	unsigned segmentSize,

	unsigned * outSums
);

}