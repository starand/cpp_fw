#ifndef __H_TOOLTIPNOTIFIER__
#define __H_TOOLTIPNOTIFIER__

#include <shellapi.h>

class CToolTipNotifier
{
public:
	CToolTipNotifier();
	~CToolTipNotifier();

	bool SetIcon( const char* szIconFile, const char* szIconTip = NULL );
	// Possible uiFlags values : NIIF_NONE, NIIF_INFO, NIIF_WARNING, NIIF_ERROR, NIIF_USER 
	bool SetMessage( const char* szMessage, const char* szTitle, uint uiFLags = NIIF_INFO );

	bool Create();
	bool Update();
	bool Destroy();

private:
	NOTIFYICONDATA	m_nidIconData;
};

#endif // __H_TOOLTIPNOTIFIER__
