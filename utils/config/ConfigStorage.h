#ifndef __CONFIGSTORAGE_H_INCLUDED
#define __CONFIGSTORAGE_H_INCLUDED


enum ECONFIGOPTION;


//////////////////////////////////////////////////

class CConfigStorage
{
public:
	CConfigStorage();
	~CConfigStorage();

public:
	void StoreOptionValue(ECONFIGOPTION coConfigOption, const string& sOptionValue);

	bool GetOptionString(ECONFIGOPTION coConfigOption, string &sOutOptionValue) const;
	
	bool GetOptionInt(ECONFIGOPTION coConfigOption, int &nOutOptionValue) const;
	bool GetOptionInt(ECONFIGOPTION coConfigOption, sizeint &siOutOptionValue) const;
	bool GetOptionInt(ECONFIGOPTION coConfigOption, ushort &usOutOptionValue) const;

public:
	void Dump();

private:
	typedef map<ECONFIGOPTION, string> COptionsStorage;

private:
	COptionsStorage &GetOptionsStorage() const { return const_cast<COptionsStorage &>(m_osOptionsStorage); }

private:
	COptionsStorage		m_osOptionsStorage;

};

#endif // __CONFIGSTORAGE_H_INCLUDED
