#pragma once

#include <memory>
#include <json/value.h>


class config_t
{
public:
    config_t( );
    virtual ~config_t( ) { }

public:
    bool read_config( const std::string& config_file, bool short_name = false );

public:
    const Json::Value& operator[]( const std::string& name ) const;

protected:
    virtual bool do_parse( );

protected:
    std::unique_ptr< Json::Value > m_json;
};
