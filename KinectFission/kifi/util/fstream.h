#pragma once

#include <cstdint>
#include <ostream>
#include <string>



namespace kifi {
namespace util {

std::uint_least64_t fsize( std::string const & fileName );

}} // namespace