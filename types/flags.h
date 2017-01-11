#ifndef __H_FLAGSCLASS__
#define __H_FLAGSCLASS__

class CFlags
{
public:
	CFlags( size_t nFlags = 0 ) : m_nFlags(nFlags) { }

	void SetFlags( size_t nFlags ) { m_nFlags = nFlags; }
	size_t GetFlags() { return m_nFlags; }

	void SetFlag( size_t nFlag, bool bSet ) { if(bSet) { m_nFlags |= nFlag; } else { m_nFlags &= ~nFlag; } }
	bool GetFlag( size_t nFlag ) { return ( (m_nFlags & nFlag) != 0 ); }

private:
	size_t	m_nFlags;
};

#endif // __H_FLAGSCLASS__
