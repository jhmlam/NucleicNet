/* $Id: Utilities.h 1161 2011-10-14 22:27:31Z mikewong899 $ */
/* Copyright (c) 2002 Mike Liang.  All rights reserved. */

#ifndef UTILITIES_H
#define UTILITIES_H

#include <string>
#include <vector>
#include <algorithm>
#include <regex.h>
#include "const.h"
#include "logutils.h"
#include <fstream>
#include <sstream>

#ifndef NULL
#define NULL (0)
#endif

using std::fstream;
using std::string;
using std::stringstream;
using std::vector;

typedef vector<string>::iterator str_iter;

/**
 * @defgroup utilities_library File Utilities Library
 * @brief Utilities for file and filename manipulation and I/O
 * @ingroup file_module
 *
 * @{
 **/

string joinPath( const string &, const string & );
string getFilename( string pathfile );

string getPdbIdFromFilename( const string &filename);

// ========================================================================
bool   fileExists( const string &, const string & = "." );
string findPdbFile( const string &pdbid, const string & = "" );
string findDsspFile( const string &pdbid, const string & = "" );
bool   findProteinFile( const string &pdbid, const string &dir, const vector< string > &extensions, const string &search, string &result );

string strToUpper ( const string orig );
string strToLower ( const string orig );
void   trim( string &orig );
string getEnvironmentVariable( const char * );

/**
 * @}
 **/

#endif
