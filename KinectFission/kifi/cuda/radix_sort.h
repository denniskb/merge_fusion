#pragma once



namespace kifi {
namespace cuda {

void radix_sort
(
	unsigned * data, unsigned size, 
	unsigned * tmp 
);

}} // namespace