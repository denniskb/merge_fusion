#pragma once



namespace kifi {
namespace cuda {

void segmented_reduce
(
	unsigned const * data, unsigned size,
	unsigned segmentSize,

	unsigned * outSums
);

}} // namespace