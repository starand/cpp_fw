#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

#define STATIC_LENGTH(a) (sizeof(a) - 1)


const char	NODE_CDATA_STAG[]="<![CDATA[";
const char	NODE_CDATA_ETAG[]="]]>";
const char	NODE_COMMENT_STAG[]="<!--";
const char	NODE_COMMENT_ETAG[]="-->";
const char	NODE_DOCDEF_STAG[]="<?xml";
const char	NODE_DOCDEF_ETAG[]="?>";
const char	NODE_TOKEN_STAG[]="<";
const char	NODE_TOKEN_ETAG[]=">";
const char	NODE_EMPTY_ETAG[]="/>";
const char	NODE_FINALISE_STAG[]="</";
const char  NODE_ATTR_STAG[] = " ";
const char  NODE_ATTR_MTAG[] = "=\"";
const char  NODE_ATTR_ETAG[] = "\"";

#endif
