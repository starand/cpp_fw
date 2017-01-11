#ifndef __ENVIRONMENT_H_INCLUDED
#define __ENVIRONMENT_H_INCLUDED

//////////////////////////////////////////////////
// CEnvironment declaration

class CEnvironment
{
public:
	CEnvironment();
	~CEnvironment();

	static bool GetVariable(const string &sVariableName, string &sOutValue);

private:

};

#endif // __ENVIRONMENT_H_INCLUDED
