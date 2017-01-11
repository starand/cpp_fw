#ifndef __H_TASK_INTERFACE__
#define __H_TASK_INTERFACE__

class ITask
{
public:
	virtual bool Execute() = 0;
	virtual void GetDescription( string& sDescription ) const { }
};

#endif // __H_TASK_INTERFACE__
