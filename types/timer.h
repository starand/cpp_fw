#pragma once

#include <common/asserts.h>
#include <threading/threading.h>
#include <ctime>
#include <set>

#ifdef WINDOWS
#   include <windows.h>
#endif


class time_observer_t
{
public:
    virtual ~time_observer_t( ) { }

    virtual void on_timer( time_t time ) = 0;
};


class timer_object_t : public thread_base_t
{
public:
    timer_object_t( uint interval )
        : m_interval( interval )
    {
    }

    void add_observer( time_observer_t& observer )
    {
        m_observers.insert(& observer );
    }

    void remove_observer( time_observer_t& observer )
    {
        m_observers.erase(& observer );
    }

private:
    virtual void do_run( )
    {
        while ( !is_stopping( ) )
        {
            notify( );

            sleep_ms( m_interval );
        }
    }

    void sleep_ms( int period )
    {
#ifdef LINUX
        struct timespec tim, tim2;
        tim.tv_sec = 0;
        tim.tv_nsec = period * 1000000; // 10 ms
        nanosleep(& tim,& tim2 );
#else
        Sleep( period );
#endif
    }

    void notify( )
    {
        for ( auto& observer : m_observers )
        {
            ASSERT( observer != nullptr );
            time_t current = std::time( nullptr );
            observer->on_timer( current );
        }
    }

private:
    uint m_interval;

    std::set< time_observer_t* > m_observers;
};

