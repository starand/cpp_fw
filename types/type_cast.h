#ifndef __TYPE_CAST_H_INCLUDED
#define __TYPE_CAST_H_INCLUDED


template <typename fromT, typename toT>
union typecast
{
	fromT _from;
	toT value;

	typecast(fromT from) : _from(from) { }
};

#define TYPECAST(_FROM_, _TO_, _VALUE_) typecast<_FROM_, _TO_>(_VALUE_).value

#endif // __TYPE_CAST_H_INCLUDED
