#include "StdAfx.h"
#include "config.h"

#include <common/asserts.h>
#include <files/fileutils.h>

#include <json/json.h>


//--------------------------------------------------------------------------------------------------

config_t::config_t( )
    : m_json( new Json::Value( ) )
{
    ASSERT( m_json.get( ) != nullptr );
}

//--------------------------------------------------------------------------------------------------

bool config_t::read_config( const std::string& config_file, bool short_name /*= false*/ )
{
    std::string config_file_name = config_file;
    if ( short_name )
    {
        std::string binary_folder;
        if ( !FileUtils::GetBinaryDir( binary_folder ) )
        {
            LOG_ERROR( "Cannot retreive binary foilder" );
            return false;
        }

        config_file_name = binary_folder + config_file;
    }

    std::string content;
    if ( !FileUtils::GetFileContent( config_file_name, content ) )
    {
        LOG_ERROR( "Cannot open config file %s", config_file.c_str( ) );
        return false;
    }

    Json::Reader reader;
    if ( !reader.parse( content, *m_json ) || m_json->isNull( ) )
    {
        LOG_ERROR( "Cannot parse config data:\n%s", content.c_str( ) );
        return false;
    }

    return do_parse( );
}

//--------------------------------------------------------------------------------------------------

const Json::Value& config_t::operator[]( const std::string& name ) const
{
    return (*m_json)[ name ];
}

//--------------------------------------------------------------------------------------------------

/*virtual */
bool config_t::do_parse( )
{
    return true;
}

//--------------------------------------------------------------------------------------------------
