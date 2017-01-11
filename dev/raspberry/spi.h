#ifndef __RASPI_SPI_H_INCLUDED
#define __RASPI_SPI_H_INCLUDED

#include "bcm2835.h"

//////////////////////////////////////////////////
// raspi_spi_t declaration

class spi_t
{
public:
	spi_t();
	~spi_t();

public:
	uint8_t read_write(uint8_t data) const;
	void read_write(const char *in_data, char *out_data, uint32_t length) const;

public:
	static void sleep_ms(int ms);

private:

};

#endif // __RASPI_SPI_H_INCLUDED
