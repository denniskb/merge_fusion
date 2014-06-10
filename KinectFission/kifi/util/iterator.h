#pragma once

#include <cstddef>
#include <iterator>



namespace kifi {
namespace util {

template< class Iterator, class UnaryOperation >
class map_iterator : public std::iterator
<
	typename std::iterator_traits< Iterator >::iterator_category,
	typename std::iterator_traits< Iterator >::value_type,
	typename std::iterator_traits< Iterator >::difference_type,
	typename std::iterator_traits< Iterator >::pointer,
	typename std::iterator_traits< Iterator >::reference
>
{
public:
	map_iterator();
	map_iterator( Iterator it, UnaryOperation op );

	map_iterator & operator++();
	map_iterator operator++( int );
	map_iterator & operator--();
	map_iterator operator--( int );

	bool operator==( map_iterator const & rhs ) const;
	bool operator!=( map_iterator const & rhs ) const;

	value_type operator*() const;

	bool operator<( map_iterator const & rhs ) const;
	bool operator>( map_iterator const & rhs ) const;
	bool operator<=( map_iterator const & rhs ) const;
	bool operator>=( map_iterator const & rhs ) const;

	difference_type operator-( map_iterator const & rhs );

	friend map_iterator 
	operator+( map_iterator const &, difference_type );

	friend map_iterator 
	operator-( map_iterator const &, difference_type );

	friend map_iterator 
	operator+( difference_type, map_iterator const & );

	friend map_iterator 
	operator-( difference_type, map_iterator const & );

	map_iterator & operator+=( difference_type n );
	map_iterator & operator-=( difference_type n );

	value_type operator[]( std::size_t n ) const;

private:
	Iterator m_it;
	UnaryOperation m_op;
};



template< class Iterator, class UnaryOperation >
map_iterator< Iterator, UnaryOperation > make_map_iterator( Iterator it, UnaryOperation op );

}} // namespace



#pragma region Implementation

#include <kifi/util/functional.h>



namespace kifi {
namespace util {

template< class Iterator, class UnaryOperation >
map_iterator< Iterator, UnaryOperation >::map_iterator() :
	m_it( Iterator() ),
	m_op( id< value_type >() )
{}

template< class Iterator, class UnaryOperation >
map_iterator< Iterator, UnaryOperation >::map_iterator( Iterator it, UnaryOperation op ) :
	m_it( it ),
	m_op( op )
{}



template< class Iterator, class UnaryOperation >
map_iterator< Iterator, UnaryOperation > & map_iterator< Iterator, UnaryOperation >::operator++()
{
	++m_it;
	return * this;
}

template< class Iterator, class UnaryOperation >
map_iterator< Iterator, UnaryOperation > map_iterator< Iterator, UnaryOperation >::operator++( int )
{
	auto result = * this;
	++(* this);
	return result;
}

template< class Iterator, class UnaryOperation >
map_iterator< Iterator, UnaryOperation > & map_iterator< Iterator, UnaryOperation >::operator--()
{
	--m_it;
	return * this;
}

template< class Iterator, class UnaryOperation >
map_iterator< Iterator, UnaryOperation > map_iterator< Iterator, UnaryOperation >::operator--( int )
{
	auto result = * this;
	--(* this);
	return result;
}



template< class Iterator, class UnaryOperation >
bool map_iterator< Iterator, UnaryOperation >::operator==( map_iterator const & rhs ) const
{
	return m_it == rhs.m_it;
}

template< class Iterator, class UnaryOperation >
bool map_iterator< Iterator, UnaryOperation >::operator!=( map_iterator const & rhs ) const
{
	return m_it != rhs.m_it;
}



template< class Iterator, class UnaryOperation >
typename map_iterator< Iterator, UnaryOperation >::value_type map_iterator< Iterator, UnaryOperation >::operator*() const
{
	return m_op( * m_it );
}



template< class Iterator, class UnaryOperation >
bool map_iterator< Iterator, UnaryOperation >::operator<( map_iterator const & rhs ) const
{
	return m_it < rhs.m_it;
}

template< class Iterator, class UnaryOperation >
bool map_iterator< Iterator, UnaryOperation >::operator>( map_iterator const & rhs ) const
{
	return m_it > rhs.m_it;
}

template< class Iterator, class UnaryOperation >
bool map_iterator< Iterator, UnaryOperation >::operator<=( map_iterator const & rhs ) const
{
	return m_it <= rhs.m_it;
}

template< class Iterator, class UnaryOperation >
bool map_iterator< Iterator, UnaryOperation >::operator>=( map_iterator const & rhs ) const
{
	return m_it >= rhs.m_it;
}



template< class Iterator, class UnaryOperation >
typename map_iterator< Iterator, UnaryOperation >::difference_type map_iterator< Iterator, UnaryOperation >::operator-( map_iterator const & rhs )
{
	return m_it - rhs.m_it;
}



template< class Iterator, class UnaryOperation >
map_iterator< Iterator, UnaryOperation > & map_iterator< Iterator, UnaryOperation >::operator+=( difference_type n )
{
	m_it += n;
	return * this;
}

template< class Iterator, class UnaryOperation >
map_iterator< Iterator, UnaryOperation > & map_iterator< Iterator, UnaryOperation >::operator-=( difference_type n )
{
	m_it -= n;
	return * this;
}



template< class Iterator, class UnaryOperation >
typename map_iterator< Iterator, UnaryOperation >::value_type map_iterator< Iterator, UnaryOperation >::operator[]( std::size_t n ) const
{
	return m_op( m_it[ n ] );
}



template< class Iterator, class UnaryOperation >
map_iterator< Iterator, UnaryOperation > make_map_iterator( Iterator it, UnaryOperation op )
{
	return map_iterator< Iterator, UnaryOperation >( it, op );
}

}} // namespace



template< class Iterator, class UnaryOperation >
kifi::util::map_iterator< Iterator, UnaryOperation > 
operator+( kifi::util::map_iterator< Iterator, UnaryOperation > const & it, typename kifi::util::map_iterator< Iterator, UnaryOperation >::difference_type n )
{
	return kifi::util::map_iterator< Iterator, UnaryOperation >( it.m_it + n, it.m_op );
}

template< class Iterator, class UnaryOperation >
kifi::util::map_iterator< Iterator, UnaryOperation > 
operator-( kifi::util::map_iterator< Iterator, UnaryOperation > const & it, typename kifi::util::map_iterator< Iterator, UnaryOperation >::difference_type n )
{
	return kifi::util::map_iterator< Iterator, UnaryOperation >( it.m_it - n, it.m_op );
}

template< class Iterator, class UnaryOperation >
kifi::util::map_iterator< Iterator, UnaryOperation > 
operator+( typename kifi::util::map_iterator< Iterator, UnaryOperation >::difference_type n, kifi::util::map_iterator< Iterator, UnaryOperation > const & it )
{
	return it + n;
}

template< class Iterator, class UnaryOperation >
kifi::util::map_iterator< Iterator, UnaryOperation > 
operator-( typename kifi::util::map_iterator< Iterator, UnaryOperation >::difference_type n, kifi::util::map_iterator< Iterator, UnaryOperation > const & it )
{
	return it - n;
}

#pragma endregion