#ifndef _ATOMIC_OPERATIONS_H__INCLUDED_
#define _ATOMIC_OPERATIONS_H__INCLUDED_

#if defined(LINUX)

#elif defined(SOLARIS)
#	include <atomic.h>
#elif defined(WIN32)

#endif // defined(LINUX)

namespace Atomic
{
	// These functions return the old value
	inline unsigned int Add32(volatile unsigned int * puiTarget, unsigned int uiValue);
	inline unsigned long long Add64(volatile unsigned long long * pullTarget, 
		unsigned long long ullValue);
	inline unsigned int And32(volatile unsigned int * puiTarget, unsigned int uiValue);
	inline unsigned long long And64(volatile unsigned long long * pullTarget, 
		unsigned long long ullValue);
	inline unsigned int Or32(volatile unsigned int * puiTarget, unsigned int uiValue);
	inline unsigned long long Or64(volatile unsigned long long * pullTarget, 
		unsigned long long ullValue);
	inline unsigned int CompareAndSwap32(volatile unsigned int * puiTarget, 
		unsigned int uiComparand, unsigned int uiNewValue);
	inline unsigned long long CompareAndSwap64(volatile unsigned long long * pullTarget, 
		unsigned long long ullComparand, unsigned long long ullNewValue);
	inline void * CompareAndSwapPtr(void * volatile * ppTarget, void * pComparand, 
		void * pNewValue);
	inline unsigned int Swap32(volatile unsigned int * puiTarget, unsigned int uiNewValue);
	inline unsigned long long Swap64(volatile unsigned long long * pullTarget, 
		unsigned long long ullNewValue);
	inline void * SwapPtr(void * volatile * ppTarget, void * pNewValue);
	inline unsigned int Increment32(volatile unsigned int * puiTarget);
	inline unsigned long long Increment64(volatile unsigned long long * pullTarget);
	inline unsigned int Decrement32(volatile unsigned int * puiTarget);
	inline unsigned long long Decrement64(volatile unsigned long long * pullTarget);

	/*
	 * Add functions
	 */
	inline unsigned int Add32(volatile unsigned int * puiTarget, unsigned int uiValue)
	{
#if defined(LINUX)
		return __sync_fetch_and_add(puiTarget, uiValue);
#elif defined(SOLARIS)
		return (atomic_add_32_nv(puiTarget, uiValue) - uiValue);
#elif defined(WIN32)
		return (unsigned int)InterlockedExchangeAdd((LONG volatile*)puiTarget, (LONG)uiValue);
#endif // defined(LINUX)
	}

	inline unsigned long long Add64(volatile unsigned long long * pullTarget, 
		unsigned long long ullValue)
	{
#if defined(LINUX)
		return __sync_fetch_and_add(pullTarget, ullValue);
#elif defined(SOLARIS)
		return (atomic_add_64_nv((volatile unsigned long *)pullTarget, ullValue) - ullValue);
#elif defined(WIN32)
		return (unsigned long long)InterlockedExchangeAdd64((LONGLONG volatile*)pullTarget, 
			(LONGLONG)ullValue);
#endif // defined(LINUX)
	}

	/*
	 * And functions
	 */
	inline unsigned int And32(volatile unsigned int * puiTarget, unsigned int uiValue)
	{
#if defined(LINUX)
		return __sync_fetch_and_and(puiTarget, uiValue);
#elif defined(SOLARIS)
		register unsigned int uiOldValue = 0U;
		do
		{
			uiOldValue = *puiTarget;
		} 
		while(CompareAndSwap32(puiTarget, uiOldValue, uiOldValue & uiValue) != uiOldValue);
		return uiOldValue;
#elif defined(WIN32)
		return (unsigned int)_InterlockedAnd((LONG volatile*)puiTarget, (LONG)uiValue);
#endif // defined(LINUX)
	}

	inline unsigned long long And64(volatile unsigned long long * pullTarget, 
		unsigned long long ullValue)
	{
#if defined(LINUX)
		return __sync_fetch_and_and(pullTarget, ullValue);
#elif defined(SOLARIS)
		register unsigned long long ullOldValue = 0U;
		do
		{
			ullOldValue = *pullTarget;
		} 
		while(CompareAndSwap64(pullTarget, ullOldValue, ullOldValue & ullValue) != ullOldValue);
		return ullOldValue;
#elif defined(WIN32)
		return (unsigned long long)InterlockedAnd64((LONGLONG volatile*)pullTarget, 
			(LONGLONG)ullValue);
#endif // defined(LINUX)
	}

	/*
	 * Or functions
	 */
	inline unsigned int Or32(volatile unsigned int * puiTarget, unsigned int uiValue)
	{
#if defined(LINUX)
		return __sync_fetch_and_or(puiTarget, uiValue);
#elif defined(SOLARIS)
		register unsigned int uiOldValue = 0U;
		do
		{
			uiOldValue = *puiTarget;
		} 
		while(CompareAndSwap32(puiTarget, uiOldValue, uiOldValue | uiValue) != uiOldValue);
		return uiOldValue;
#elif defined(WIN32)
		return (unsigned int)_InterlockedOr((LONG volatile*)puiTarget, (LONG)uiValue);
#endif // defined(LINUX)
	}

	inline unsigned long long Or64(volatile unsigned long long * pullTarget, 
		unsigned long long ullValue)
	{
#if defined(LINUX)
		return __sync_fetch_and_or(pullTarget, ullValue);
#elif defined(SOLARIS)
		register unsigned long long ullOldValue = 0U;
		do
		{
			ullOldValue = *pullTarget;
		} 
		while(CompareAndSwap64(pullTarget, ullOldValue, ullOldValue | ullValue) != ullOldValue);
		return ullOldValue;
#elif defined(WIN32)
		return (unsigned long long)InterlockedOr64((LONGLONG volatile*)pullTarget, 
			(LONGLONG)ullValue);
#endif // defined(LINUX)
	}

	/*
	 * Compare and Swap functions
	 */
	inline unsigned int CompareAndSwap32(volatile unsigned int * puiTarget, 
		unsigned int uiComparand, unsigned int uiNewValue)
	{
#if defined(LINUX)
		return __sync_val_compare_and_swap(puiTarget, uiComparand, uiNewValue);
#elif defined(SOLARIS)
		return atomic_cas_32(puiTarget, uiComparand, uiNewValue);
#elif defined(WIN32)
		return (unsigned int)InterlockedCompareExchange((volatile LONG*)puiTarget, (LONG)uiNewValue,
			(LONG)uiComparand);
#endif // defined(LINUX)
	}

	inline unsigned long long CompareAndSwap64(volatile unsigned long long * pullTarget, 
		unsigned long long ullComparand, unsigned long long ullNewValue)
	{
#if defined(LINUX)
		return __sync_val_compare_and_swap(pullTarget, ullComparand, ullNewValue);
#elif defined(SOLARIS)
		return atomic_cas_64((volatile unsigned long *)pullTarget, ullComparand, ullNewValue);
#elif defined(WIN32)
		return (unsigned long long)InterlockedCompareExchange64((LONGLONG volatile*)pullTarget, 
			(LONGLONG)ullNewValue, (LONGLONG)ullComparand);
#endif // defined(LINUX)
	}

	inline void * CompareAndSwapPtr(void * volatile * ppTarget, void * pComparand, 
		void * pNewValue)
	{
#if defined(LINUX)
		return __sync_val_compare_and_swap((void**)ppTarget, pComparand, pNewValue);
#elif defined(SOLARIS)
		return atomic_cas_ptr(ppTarget, pComparand, pNewValue);
#elif defined(WIN32)
		return InterlockedCompareExchangePointer((PVOID volatile *)ppTarget, pNewValue, pComparand);
#endif // defined(LINUX)
	}

	/*
	 * Swap functions
	 */
	inline unsigned int Swap32(volatile unsigned int * puiTarget, unsigned int uiNewValue)
	{
#if defined(LINUX)
		return __sync_lock_test_and_set (puiTarget, uiNewValue);
#elif defined(SOLARIS)
		return atomic_swap_32(puiTarget, uiNewValue);
#elif defined(WIN32)
		return (unsigned int)InterlockedExchange((volatile LONG *)puiTarget, (LONG)uiNewValue);
#endif // defined(LINUX)
	}

	inline unsigned long long Swap64(volatile unsigned long long * pullTarget, 
		unsigned long long ullNewValue)
	{
#if defined(LINUX)
		return __sync_lock_test_and_set(pullTarget, ullNewValue);
#elif defined(SOLARIS)
		return atomic_swap_64((volatile unsigned long *)pullTarget, ullNewValue);
#elif defined(WIN32)
		return (unsigned long long)InterlockedExchange64((LONGLONG volatile*)pullTarget, 
			(LONGLONG)ullNewValue);
#endif // defined(LINUX)
	}

	inline void * SwapPtr(void * volatile * ppTarget, void * pNewValue)
	{
#if defined(LINUX)
		return __sync_lock_test_and_set((void**)ppTarget, pNewValue);
#elif defined(SOLARIS)
		return atomic_swap_ptr(ppTarget, pNewValue);
#elif defined(WIN32)
		// volatile qualifier causes build errors 
		return InterlockedExchangePointer((PVOID *)ppTarget, pNewValue);
#endif // defined(LINUX)
	}

	/*
	 * Increment/Decrement functions
	 */
	inline unsigned int Increment32(volatile unsigned int * puiTarget)
	{
#if defined(LINUX)
		return Add32(puiTarget, 1U);
#elif defined(SOLARIS)
		return atomic_inc_32_nv(puiTarget) - 1U;
#elif defined(WIN32)
		return (unsigned int)InterlockedIncrement((LONG volatile*)puiTarget) - 1U;
#endif // defined(LINUX)
	}

	inline unsigned long long Increment64(volatile unsigned long long * pullTarget)
	{
#if defined(LINUX)
		return Add64(pullTarget, 1ULL);
#elif defined(SOLARIS)
		return atomic_inc_64_nv((volatile unsigned long *)pullTarget) - 1ULL;
#elif defined(WIN32)
		return (unsigned long long)InterlockedIncrement64((LONGLONG volatile*)(pullTarget)) - 1ULL;
#endif // defined(LINUX)
	}

	inline unsigned int Decrement32(volatile unsigned int * puiTarget)
	{
#if defined(LINUX)
		return __sync_sub_and_fetch(puiTarget, (unsigned int)1U) + 1U;
#elif defined(SOLARIS)
		return atomic_dec_32_nv(puiTarget) + 1U;
#elif defined(WIN32)
		return InterlockedDecrement((volatile LONG*)puiTarget) + 1U;
#endif // defined(LINUX)
	}

	inline unsigned long long Decrement64(volatile unsigned long long * pullTarget)
	{
#if defined(LINUX)
		return __sync_sub_and_fetch(pullTarget, (unsigned int)1ULL) + 1ULL;
#elif defined(SOLARIS)
		return atomic_dec_64_nv((volatile unsigned long *)pullTarget) + 1ULL;
#elif defined(WIN32)
		return InterlockedDecrement64((LONGLONG volatile *)pullTarget) + 1ULL;
#endif // defined(LINUX)
	}
};

#endif // _ATOMIC_OPERATIONS_H__INCLUDED_
