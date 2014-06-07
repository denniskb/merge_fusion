#include <kifi/Brick.h>



namespace kifi {

// static 
void Brick::Index1Dto3D
(
	unsigned index,

	unsigned & outX,
	unsigned & outY,
	unsigned & outZ
)
{
	outX = index & 1u;
	outY = ( index >> 1 ) & 1u;
	outZ = index >> 2;
}

} // namespace