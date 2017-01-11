#include "spi.h"

#include "bcm2835.h"
#include "asserts.h"

#include <time.h>

//////////////////////////////////////////////////
// raspi_spi_t implementation

spi_t::spi_t()
{
	if (!bcm2835_init())
	{
		ASSERT_FAIL("Unable to init bcm2835 library");
	}

	bcm2835_spi_begin();
	bcm2835_spi_setBitOrder(BCM2835_SPI_BIT_ORDER_MSBFIRST);
	bcm2835_spi_setDataMode(BCM2835_SPI_MODE0);
	bcm2835_spi_setClockDivider(BCM2835_SPI_CLOCK_DIVIDER_65536);
	bcm2835_spi_chipSelect(BCM2835_SPI_CS0);
	bcm2835_spi_setChipSelectPolarity(BCM2835_SPI_CS0, LOW);
}

spi_t::~spi_t()
{

}


uint8_t spi_t::read_write(uint8_t data) const
{
	return bcm2835_spi_transfer(data);
}

void spi_t::read_write(const char *in_data, char *out_data, uint32_t length) const
{
	bcm2835_spi_transfernb((char *)in_data, out_data, length);
}


/*static */
void spi_t::sleep_ms(int ms)
{
	struct timespec tim, tim2;
	tim.tv_sec = 0;
	tim.tv_nsec = ms * 1000000; // 10 ms
	nanosleep(&tim , &tim2);
}
