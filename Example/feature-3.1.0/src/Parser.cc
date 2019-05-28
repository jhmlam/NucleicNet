// FEATURE
// See COPYING and AUTHORS for copyright and license agreement terms.

// $Id: Parser.cc 1166 2011-10-21 05:33:24Z mikewong899 $

#include "Parser.h"

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

const bool Parser::isAllowed[256] = {
	// For each ASCII character (0-255), true if ASCII character matches POSIX
	// character class isalnum, false otherwise
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,
	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
	1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

Parser::Parser(const string& stream) {
    position   = 0;
	datastream = stream;
}

Parser::Parser() {
}

void Parser::assertToken(const string& wanted, string error) {
    string token;
	token.clear();
    getToken( token );

    if ( token != wanted ) {
        if (error != "")
            throw error;
        else throw string("unexpected token [Parser::assertToken] wanted '" + wanted + "' and got '" + token + "'" );
    }
}

bool Parser::getToken(string& token) {
	token.clear();

    if (datastream.length() == 0)
        return false;

    if (saved != "") {
        token = saved;
        saved.clear();
        return true;
    }

    // Advance to the next valid letter
    char letter = datastream[ position ];
    while ((!isgraph( letter )) && position < datastream.length() ) {
		position++;
		letter = datastream[ position ];
    }

	// Accept all valid letters as part of the token
	while (isAllowed[ (int) letter ] && position < datastream.length() ) {
		token += letter;
		position++;
		letter = datastream[ position ];
	}
    return true;
}

bool Parser::parseUntil( const string& token ) {
    if (datastream.length() == 0)
        return false;

    if ((saved != "") && (saved.find( token ) != string::npos)) {
    	saved.clear();
        return true;
	}

	string remaining = datastream.substr( position );
	size_t found     = remaining.find( token );
	if( found == string::npos ) {
		return false;
	}
    return true;
}

bool Parser::peekToken(string& token) {
	if( saved != "" ) {
		token = saved;
		return true;
	}

    long int previous = position;
    bool tokenFound = getToken(token);
    position = previous;
    return tokenFound;
}

void Parser::reset() {
    this->saved.clear();
    this->position = 0;
}

void Parser::ungetToken(string& token) {
    if (saved != "")
        throw "Parser::ungetToken failed: token already cached";
    saved = token;
}

Parser::FixedWidth::FixedWidth() : Parser() {
}

Parser::FixedWidth::FixedWidth( const string & stream ) : Parser( stream ) {
	datastream = stream;
}

void Parser::FixedWidth::parseString(string &dest, size_t start, size_t length) {
	dest   = datastream.substr(start, length);
	// Trim leading and trailing whitespace
	size_t lead  = dest.find_first_not_of( " \t" );
	size_t trail = dest.find_last_not_of(  " \t" );
	if((lead != string::npos) && (trail != string::npos)) {
		dest = dest.substr( lead, (trail - lead) +1);
	}
}

void Parser::FixedWidth::parseQuote(string& dest, size_t startPos) {
	size_t endPos = datastream.find( '\"', startPos);

	if ( endPos != string::npos )
		dest = datastream.substr( startPos, endPos );
	return;
}

void Parser::FixedWidth::parseChar( char& dest, size_t start ) {
	if (isprint(datastream[start])) {
		dest = datastream[start];
	} else {
		dest = ' ';
	}
}

void Parser::FixedWidth::parseDouble( double& dest, size_t start, size_t length ) {
	string substring = datastream.substr( start, length );
	istringstream stream( substring );
	if (!(stream >> dest))
		dest = 0;
	return;
}

void Parser::FixedWidth::parseFloat( float& dest, size_t start, size_t length ) {
	string substring = datastream.substr( start, length );
	istringstream stream( substring );
	if (!(stream >> dest))
		dest = 0;
	return;
}

void Parser::FixedWidth::parseLong( long int& dest, size_t start, size_t length ) {
	string substring = datastream.substr( start, length );
	istringstream stream( substring );
	if (!(stream >> dest))
		dest = 0;
	return;
}

void Parser::FixedWidth::parseShort( short& dest, size_t start, size_t length ) {
	string substring = datastream.substr( start, length );
	istringstream stream( substring );
	if (!(stream >> dest))
		dest = 0;
	return;
}

void Parser::FixedWidth::parseInt( int& dest, size_t start, size_t length ) {
	string substring = datastream.substr( start, length );
	istringstream stream( substring );
	if (!(stream >> dest))
		dest = 0;
	return;
}

bool isStringIn( string& set, string& query ) {
	vector<string> items;
	stringstream tokens( set );
	string token;
	while( tokens.good() ) {
		tokens >> token;
		items.push_back( token );
	} 
	vector<string>::iterator item;
	for( item = items.begin(); item != items.end(); item++ ) {
		if( query == *item ) return true;
	}
	return false;
}

