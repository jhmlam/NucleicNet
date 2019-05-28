/* $Id: stl_types.h 1075 2011-08-25 01:14:45Z mikewong899 $ */
/* Copyright (c) 2002 Mike Liang.  All rights reserved. */

#ifndef STL_TYPES_H
#define STL_TYPES_H

#include <string>
#include <iostream>
#include <utility>
#include <algorithm>
#include <vector>
#include <map>
#include <sstream>

using std::map;
using std::ostream;
using std::string;
using std::stringstream;
using std::vector;

/**
 * @brief Specifies a vector of string objects
 **/
class Strings : public vector<string> {
public:

    /**
     * @brief Adds the given string to the Strings object
     * @param data A string token to add to the Strings object
     **/
    void append(const char *data) {
        this->push_back(string(data));
    }
    void append( string data) {
        this->push_back( data );
    }

    /**
     * @brief Adds a number of strings parsed from a character-separated string
     * @param data The character-separated string
     * @param sep The character that indicates a separation of tokens, e.g. a comma
     **/
    int extend(char *data, const char *sep) {
		stringstream buffer( data );
        int count = 0;
		char newline = '\n';
		while( buffer ) {
			string token;

			// ===== FIND THE DELIMITER THAT GIVES THE SMALLEST SUBSTRING
			size_t min = buffer.str().size();
			char *delimiter = NULL;
			for( char *i = (char *) sep; (*i); i++ ) {
				size_t pos = buffer.str().find( (*i ));
				if( pos == string::npos ) { delimiter = &newline; }
				if( pos < min )           { delimiter = i; min = pos; }
			}
			if( ! getline( buffer, token, (*delimiter))) break;
			this->push_back( token );
			count++;
		}
        return count;
    }
};

typedef vector<double> Doubles;

ostream &operator<<(ostream &outs, const Strings &list);
const char *toString(const Strings &strings);

#endif
