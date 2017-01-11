#include "StdAfx.h"
#include "volume_mixer.h"

#include <Mmdeviceapi.h>
#include <Endpointvolume.h>


//////////////////////////////////////////////////
// CMixerVolume implementation

volume_mixer_t::volume_mixer_t()
	: m_pepEndpointVolume(NULL)
{
	initialize();
}

volume_mixer_t::~volume_mixer_t()
{
	IAudioEndpointVolume *pevEndpointVolume = get_endpoint_volume();

	if (pevEndpointVolume)
	{
		pevEndpointVolume->Release();
	}

	CoUninitialize();
}


bool volume_mixer_t::initialize()
{
	bool bResult = false;
	IMMDevice *defaultDevice = NULL;

	do 
	{
		if (S_OK != CoInitialize(NULL))
		{
			//LOG_ERROR("Unable to initialize COM library");
			break;
		}

		IMMDeviceEnumerator *deviceEnumerator = NULL;
		if (S_OK != CoCreateInstance(__uuidof(MMDeviceEnumerator), NULL, CLSCTX_INPROC_SERVER, __uuidof(IMMDeviceEnumerator), (LPVOID *)&deviceEnumerator))
		{
			//LOG_ERROR("Unable to create COM instance");
			break;
		}

		if (S_OK != deviceEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &defaultDevice))
		{
			//LOG_ERROR("Unable to retrieve default audio endpoint");
			break;
		}

		IAudioEndpointVolume *pepEndpointVolume = NULL;
		if (S_OK != defaultDevice->Activate(__uuidof(IAudioEndpointVolume), CLSCTX_INPROC_SERVER, NULL, (LPVOID *)&pepEndpointVolume))
		{
			//LOG_ERROR("Unable to activate audio endpoint");
			break;
		}

		ASSERTE(pepEndpointVolume);
		set_endpoint_volume(pepEndpointVolume);
		bResult = true;
	}
	while (false);

	defaultDevice->Release();
	defaultDevice = NULL;

	return bResult;
}


bool volume_mixer_t::get_volume_level(float& fOutLevel, bool bScalar /*= true*/) const
{
	IAudioEndpointVolume *pevEndpointVolume = get_endpoint_volume();
	ASSERTE(pevEndpointVolume);

	HRESULT hResult = bScalar 
		? pevEndpointVolume->GetMasterVolumeLevelScalar(&fOutLevel)
		: pevEndpointVolume->GetMasterVolumeLevel(&fOutLevel);

	bool bResult = S_OK == hResult;
	return bResult;
}

bool volume_mixer_t::set_volume_level(float fLevel, bool bScalar /*= true*/) const
{
	IAudioEndpointVolume *pevEndpointVolume = get_endpoint_volume();
	ASSERTE(pevEndpointVolume);

	HRESULT hResult = bScalar
		? pevEndpointVolume->SetMasterVolumeLevelScalar(fLevel, NULL) // range from 0.0 to 1.0
		: pevEndpointVolume->SetMasterVolumeLevel(fLevel, NULL); // leven in decibels; to optain call IAudioEndpointVolume::GetVolumeRange(float *min, float *max, float *increment)

	bool bResult = S_OK == hResult;
	return bResult;
}


bool volume_mixer_t::get_volume_range(float& fOutMinRange, float& fOutMaxRange, float& fOutIncrement) const
{
	IAudioEndpointVolume *pevEndpointVolume = get_endpoint_volume();
	ASSERTE(pevEndpointVolume);

	bool bResult = S_OK == pevEndpointVolume->GetVolumeRange(&fOutMinRange, &fOutMaxRange, &fOutIncrement);
	return bResult;	
}
