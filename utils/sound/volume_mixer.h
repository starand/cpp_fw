#ifndef __VOLUMEMIXER_H_INCLUDED
#define __VOLUMEMIXER_H_INCLUDED


struct IAudioEndpointVolume;

//////////////////////////////////////////////////
// CMixerVolume declaration

class volume_mixer_t
{
public:
	volume_mixer_t();
	~volume_mixer_t();

private:
	bool initialize();

public:
	bool is_initialized() const { return m_pepEndpointVolume != NULL; }

	bool get_volume_level(float& fOutLevel, bool bScalar = true) const;
	bool set_volume_level(float fLevel, bool bScalar = true) const;

	bool get_volume_range(float& fOutMinRange, float& fOutMaxRange, float& fOutIncrement) const;

private:
	IAudioEndpointVolume *get_endpoint_volume() const { return const_cast<IAudioEndpointVolume *>(m_pepEndpointVolume); }
	void set_endpoint_volume(IAudioEndpointVolume *pepEndpointVolume) { m_pepEndpointVolume = pepEndpointVolume; }

private:
	IAudioEndpointVolume *m_pepEndpointVolume;
};

#endif // __VOLUMEMIXER_H_INCLUDED
