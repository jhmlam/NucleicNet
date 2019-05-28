// FEATURE
// See COPYING and AUTHORS for copyright and license agreement terms.

// $Id: Parser.h 1125 2011-09-20 23:08:40Z mikewong899 $

// Header file for Parser.cpp

#ifndef PARSER_H
#define PARSER_H

#include "Utilities.h"
#include "gzstream.h"
#include <string>
#include <iostream>
#include <sstream>

using std::istringstream;
using std::string;
using std::stringstream;

/**
 * @brief A generalized stream parser
 * @ingroup file_module
 **/
class Parser {
	private:
		static const bool isAllowed[256];

	protected:
		string            saved;
		string            datastream;
		unsigned long     position;

	public:
		Parser();
		Parser(const string& stream);

		void assertToken( const string& token, string Error = "" );
		bool getToken( string& token );
		bool parseUntil( const string& token );
		bool peekToken( string& token );
		void reset();
		void ungetToken( string& token );

	class FixedWidth;
};

/**
 * @brief A simple parser for fixed-width column formats or simply for
 * extracting values from strings.
 * @ingroup file_module
 **/
class Parser::FixedWidth : public Parser {
	public:
		FixedWidth();
		FixedWidth( const string & );

		void  parseString( string &   dest, size_t start = 0,    size_t length = string::npos );
		void  parseChar(   char &     dest, size_t start = 0 );
		void  parseDouble( double&    dest, size_t start = 0,    size_t length = string::npos );
		void  parseFloat(  float &    dest, size_t start = 0,    size_t length = string::npos );
		void  parseLong(   long int & dest, size_t start = 0,    size_t length = string::npos );
		void  parseShort(  short &    dest, size_t start = 0,    size_t length = string::npos );
		void  parseInt(    int &      dest, size_t start = 0,    size_t length = string::npos );
		void  parseQuote(  string &   dest, size_t start = 0 );

		void  parseString( string &   dest, string & source ) { datastream = source; parseString( dest ); };
		void  parseChar(   char &     dest, string & source ) { datastream = source; parseChar(   dest ); };
		void  parseDouble( double&    dest, string & source ) { datastream = source; parseDouble( dest ); };
		void  parseFloat(  float &    dest, string & source ) { datastream = source; parseFloat(  dest ); };
		void  parseLong(   long int & dest, string & source ) { datastream = source; parseLong(   dest ); };
		void  parseShort(  short &    dest, string & source ) { datastream = source; parseShort(  dest ); };
		void  parseInt(    int &      dest, string & source ) { datastream = source; parseInt(    dest ); };
		void  parseQuote(  string &   dest, string & source ) { datastream = source; parseQuote(  dest ); };
};
bool isStringIn( string& set, string &query );

#endif
