/* $Id: stl_types.cc 1161 2011-10-14 22:27:31Z mikewong899 $ */
/* Copyright (c) 2002 Mike Liang.  All rights reserved. */

#include "stl_types.h"
#ifndef USE_TEMPLATE

ostream & operator<<(ostream &outs, const Strings &list) {
    outs << toString( list );
    return outs;
}

const char *toString(const Strings &list) {
    string buffer;
	for( Strings::const_iterator i = list.begin(); i != list.end(); i++ ) {
		buffer += *i + ' ';
	}
	return buffer.c_str();
}
#endif
