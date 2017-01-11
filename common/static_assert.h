#ifndef __STATIC_ASSERT_H_INCLUDED
#define __STATIC_ASSERT_H_INCLUDED

template<bool value> 
struct static_assert_t {};

template<> 
struct static_assert_t<true>
{
	static void do_check() {}
};

#define STATIC_ASSERT(_expr_) static_assert_t<(_expr_)>::do_check()

#endif // __STATIC_ASSERT_H_INCLUDED
