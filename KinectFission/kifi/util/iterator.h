#pragma once

#include <cstddef>
#include <iterator>



namespace kifi {
namespace util {

#pragma warning( push )
#pragma warning( disable : 4512 ) // assignment operator could not be generated for const_iterator due to member 'm_value'

template< typename T >
class const_iterator : public std::iterator< std::input_iterator_tag, T >
{
public:
	// all
	const_iterator( T value ) : m_value( value ) {}

	const_iterator & operator++() { return * this; }
	const_iterator operator++( int ) { return * this; }

	// input
	bool operator==( const_iterator const & rhs ) const;
	bool operator!=( const_iterator const & rhs ) const;

	T const & operator*() const { return m_value; }
	T const * operator->() const { return & m_value; }

private:
	T m_value;
};

#pragma warning( pop )

template< typename T >
const_iterator< T > make_const_iterator( T const & x ) { return const_iterator< T >( x ); }



template< class Iterator, class UnaryOperation >
class transform_iterator : public std::iterator
<
	typename std::iterator_traits< Iterator >::iterator_category,
	typename std::iterator_traits< Iterator >::value_type,
	typename std::iterator_traits< Iterator >::difference_type,
	typename std::iterator_traits< Iterator >::pointer,
	typename std::iterator_traits< Iterator >::reference
>
{
public:
	// all
	transform_iterator( Iterator it, UnaryOperation op );

	transform_iterator & operator++();
	transform_iterator operator++( int );

	// input
	bool operator==( transform_iterator const & rhs ) const;
	bool operator!=( transform_iterator const & rhs ) const;

	value_type operator*() const;

	// output

	// forward
	transform_iterator();

	// bidirectional
	transform_iterator & operator--();
	transform_iterator operator--( int );

	// random access
	       transform_iterator operator+( difference_type n );
	friend transform_iterator operator+( difference_type n, transform_iterator const & a );
	       transform_iterator operator-( difference_type n );
		   difference_type    operator-( transform_iterator const & rhs );

	bool operator<( transform_iterator const & rhs ) const;
	bool operator>( transform_iterator const & rhs ) const;
	bool operator<=( transform_iterator const & rhs ) const;
	bool operator>=( transform_iterator const & rhs ) const;

	transform_iterator & operator+=( difference_type n );
	transform_iterator & operator-=( difference_type n );

	value_type operator[]( std::size_t n ) const;

private:
	Iterator m_it;
	UnaryOperation m_op;
};

template< class Iterator, class UnaryOperation >
transform_iterator< Iterator, UnaryOperation > make_transform_iterator( Iterator it, UnaryOperation op );

}} // namespace



#pragma region Implementation

#include <kifi/util/functional.h>



namespace kifi {
namespace util {

template< class Iterator, class UnaryOperation >
transform_iterator< Iterator, UnaryOperation >::transform_iterator( Iterator it, UnaryOperation op ) :
	m_it( it ),
	m_op( op )
{}



template< class Iterator, class UnaryOperation >
transform_iterator< Iterator, UnaryOperation > & transform_iterator< Iterator, UnaryOperation >::operator++()
{
	++m_it;
	return * this;
}

template< class Iterator, class UnaryOperation >
transform_iterator< Iterator, UnaryOperation > transform_iterator< Iterator, UnaryOperation >::operator++( int )
{
	auto result = * this;
	++(* this);
	return result;
}



template< class Iterator, class UnaryOperation >
bool transform_iterator< Iterator, UnaryOperation >::operator==( transform_iterator const & rhs ) const
{
	return m_it == rhs.m_it;
}

template< class Iterator, class UnaryOperation >
bool transform_iterator< Iterator, UnaryOperation >::operator!=( transform_iterator const & rhs ) const
{
	return m_it != rhs.m_it;
}



template< class Iterator, class UnaryOperation >
typename transform_iterator< Iterator, UnaryOperation >::value_type transform_iterator< Iterator, UnaryOperation >::operator*() const
{
	return m_op( * m_it );
}



template< class Iterator, class UnaryOperation >
transform_iterator< Iterator, UnaryOperation >::transform_iterator() :
	m_it( Iterator() ),
	m_op( id< value_type >() )
{}



template< class Iterator, class UnaryOperation >
transform_iterator< Iterator, UnaryOperation > & transform_iterator< Iterator, UnaryOperation >::operator--()
{
	--m_it;
	return * this;
}

template< class Iterator, class UnaryOperation >
transform_iterator< Iterator, UnaryOperation > transform_iterator< Iterator, UnaryOperation >::operator--( int )
{
	auto result = * this;
	--(* this);
	return result;
}



template< class Iterator, class UnaryOperation >
transform_iterator< Iterator, UnaryOperation > transform_iterator< Iterator, UnaryOperation >::operator+( difference_type n )
{
	return transform_iterator( m_it + n, m_op );
}

template< class Iterator, class UnaryOperation >
transform_iterator< Iterator, UnaryOperation > transform_iterator< Iterator, UnaryOperation >::operator-( difference_type n )
{
	return transform_iterator( m_it - n, m_op );
}

template< class Iterator, class UnaryOperation >
typename transform_iterator< Iterator, UnaryOperation >::difference_type transform_iterator< Iterator, UnaryOperation >::operator-( transform_iterator const & rhs )
{
	return m_it - rhs.m_it;
}



template< class Iterator, class UnaryOperation >
bool transform_iterator< Iterator, UnaryOperation >::operator<( transform_iterator const & rhs ) const
{
	return m_it < rhs.m_it;
}

template< class Iterator, class UnaryOperation >
bool transform_iterator< Iterator, UnaryOperation >::operator>( transform_iterator const & rhs ) const
{
	return m_it > rhs.m_it;
}

template< class Iterator, class UnaryOperation >
bool transform_iterator< Iterator, UnaryOperation >::operator<=( transform_iterator const & rhs ) const
{
	return m_it <= rhs.m_it;
}

template< class Iterator, class UnaryOperation >
bool transform_iterator< Iterator, UnaryOperation >::operator>=( transform_iterator const & rhs ) const
{
	return m_it >= rhs.m_it;
}



template< class Iterator, class UnaryOperation >
transform_iterator< Iterator, UnaryOperation > & transform_iterator< Iterator, UnaryOperation >::operator+=( difference_type n )
{
	m_it += n;
	return * this;
}

template< class Iterator, class UnaryOperation >
transform_iterator< Iterator, UnaryOperation > & transform_iterator< Iterator, UnaryOperation >::operator-=( difference_type n )
{
	m_it -= n;
	return * this;
}



template< class Iterator, class UnaryOperation >
typename transform_iterator< Iterator, UnaryOperation >::value_type transform_iterator< Iterator, UnaryOperation >::operator[]( std::size_t n ) const
{
	return m_op( m_it[ n ] );
}



template< class Iterator, class UnaryOperation >
transform_iterator< Iterator, UnaryOperation > make_transform_iterator( Iterator it, UnaryOperation op )
{
	return transform_iterator< Iterator, UnaryOperation >( it, op );
}

}} // namespace



template< class Iterator, class UnaryOperation >
kifi::util::transform_iterator< Iterator, UnaryOperation > 
operator+( typename kifi::util::transform_iterator< Iterator, UnaryOperation >::difference_type n, kifi::util::transform_iterator< Iterator, UnaryOperation > const & it )
{
	return it + n;
}

#pragma endregion