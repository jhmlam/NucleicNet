/* $Id: Utilities.cc 1535 2013-07-12 01:07:37Z teague $ */
/* Copyright (c) 2002 Mike Liang.  All rights reserved. */

#include "Utilities.h"

string joinPath( const string &path, const string &filename ) {
	string joined;
	size_t lastChar = (path.size() - 1);
	if( ! path.empty() ) {
		joined = path;
		if( joined[ lastChar ] != '/' ) joined += '/';
	}
	joined += filename;
    return joined;
}

/**
 * \brief Retrieves the filename from an absolute or relative path
 **/
string getFilename( string pathfile ) {
	string filename;
	size_t start = pathfile.find_last_of( '/' );
    if ( start == string::npos )
        filename = pathfile;
    else
        filename = pathfile.substr( start );
    return filename;
}

/**
 * \brief Extract pdbid from filename (best guess)
 **/
string getPdbIdFromFilename( const string &pathfile ) {
    string             filename      = getFilename( pathfile );
    static const char *pdbidPattern  = "[0-9][0-9a-zA-Z]{3}(\\.[0-9]+)?";
    static regex_t     pdbidRegex;
    static bool        isInitialized = false;

	// Compile regular expression pattern
    if (isInitialized == false) {
        if( regcomp( &pdbidRegex, pdbidPattern, REG_EXTENDED)!=0 ) {
            debug( 1, "Error creating pattern %s\n", pdbidPattern );
            return NULL;
        }
        isInitialized = true;
    }

    regmatch_t pdbid_match;
    size_t     start  = 0;
    size_t     length = 0;

	// Execute regular expression pattern to find a PDB ID match
    if( regexec( &pdbidRegex, filename.c_str(), 1, &pdbid_match, 0 ) == 0 ) {
        start  = pdbid_match.rm_so;
        length = pdbid_match.rm_eo - pdbid_match.rm_so;
    } else {
        fatal_error( "Utilities: Could not find a PDB ID in '%s'\n", filename.c_str() );
    }

	string pdbid = filename.substr( start, length );
    return pdbid;
}


// ========================================================================
// Find a good pdb file
string findPdbFile( const string &pdbid, const string &search ) {
	string  pdbDir = getEnvironmentVariable( PdbDirEnvStr );

	vector<string> extensions;
	extensions.push_back( "" );
	extensions.push_back( ".pdb" );
	extensions.push_back( ".ent" );
	extensions.push_back( ".FULL" );

	string found;
	if( ! findProteinFile( pdbid, pdbDir, extensions, search, found ))
		fatal_error( "PDB file not found for PDB ID %s\n", pdbid.c_str() );

	return found;
}

// ========================================================================
// Find a good dssp file
string findDsspFile( const string &dsspid, const string &search) {
	string  dsspDir = getEnvironmentVariable( DsspDirEnvStr );

	vector<string> extensions;
	extensions.push_back( "" );
	extensions.push_back( ".dssp" );
	extensions.push_back( ".DSSP" );

	string found;
	if( ! findProteinFile( dsspid, dsspDir, extensions, search, found )) 
		error( "DSSP file not found for DSSP ID %s; properties that depend on the DSSP will be inaccurate.\n", dsspid.c_str() );

	return found;
}

// ========================================================================
// Find a PDB or DSSP file
bool findProteinFile( const string &pdbid, const string &dir, const vector<string> &extensions, const string &search, string &found ) {
    vector<string> directories, directoryVariants, filenames, compressions, files;
    stringstream pathEnvironmentVariable( dir );
    
    // ===== DIRECTORIES
	string path;
	stringstream searchPath( search );
	while( getline( searchPath, path, ':' )) {
		directories.push_back( path );
	}
	directories.push_back( "." ); // Always search the current working directory
	while( getline( pathEnvironmentVariable, path, ':' )) {
		if( path == "." ) continue; // Already added the CWD
		directories.push_back( path );
	}

    // ===== DIRECTORY VARIANTS
	// RCSB PDB divided filesystem uses the middle two characters in the PDB ID
	// as an index for the PDB files
	string branch = pdbid.substr(1, 2);
	directoryVariants.push_back( "" );
	directoryVariants.push_back( "pdb" );
	directoryVariants.push_back( "dssp" );
	directoryVariants.push_back( branch );
	directoryVariants.push_back( "divided/" + branch );
	directoryVariants.push_back( "pdb/" + branch );
	directoryVariants.push_back( "dssp/" + branch );
	directoryVariants.push_back( "pdb/divided/" + branch );
	directoryVariants.push_back( "dssp/divided/" + branch );
    
	if(!isupper(branch[0]) && !isupper(branch[1])) {
		directoryVariants.push_back( strToUpper(branch));
		directoryVariants.push_back( "divided/" + strToUpper(pdbid));

	} else if(!islower(branch[0]) && !islower(branch[1])) {
		directoryVariants.push_back( strToLower(branch));
		directoryVariants.push_back( "divided/" + strToLower(branch));    
	}
    
	// ===== FILENAMES
	// filenames that RCSB has previously used
	filenames.push_back( pdbid );
	filenames.push_back("pdb"+pdbid);
	if(strToUpper(pdbid) != pdbid) {
		filenames.push_back(strToUpper(pdbid));
		filenames.push_back("PDB"+strToUpper(pdbid));

	} 
	if(strToLower(pdbid) != pdbid) {
		filenames.push_back(strToLower(pdbid));
		filenames.push_back("pdb"+strToLower(pdbid));    
	}
    
	// ===== ZLIB COMPRESSION EXTENSIONS
	compressions.push_back( "" );
	compressions.push_back( ".gz" );
	compressions.push_back( ".Z" );
    
	for(str_iter directory = directories.begin(); directory != directories.end(); directory++) {
		for(str_iter dirVar = directoryVariants.begin(); dirVar != directoryVariants.end(); dirVar++) {
			string i = joinPath( *directory, *dirVar );
			for(str_iter filename = filenames.begin(); filename != filenames.end(); filename++) {
				string j = joinPath( i, *filename );
				for(vector<string>::const_iterator ext = extensions.begin(); ext != extensions.end(); ext++) {
					string k = j + *ext;
					for(str_iter comp = compressions.begin(); comp != compressions.end(); comp++) {
						string l = k + *comp;
						files.push_back( l );
					}
				}
			}
		}
	}
    
	string file, tried;
    for(str_iter it = files.begin(); it != files.end(); it++) {
		file = (*it);
		tried += file + '\n';
        if( fileExists( file )) {
            debug(2, "Found:%s\n", it->c_str());
			found = file;
            return true;
        }
    }
	error( "Unable to locate file for %s!\n", pdbid.c_str() );
	return false;
}

bool fileExists(const string& filename, const string& pathname) {
	fstream filetest;
    string  fullName;
	if( filename[ 0 ] == '/' ) {
		fullName = filename;
	} else {
		fullName = pathname + '/' + filename;
	}
    debug(2, "Checking For File::%s\n", fullName.c_str());
	filetest.open( fullName.c_str(), fstream::in );
    if (filetest.is_open()) {
		filetest.close();
        return true;
    } else {
        return false;
    }
}

string strToUpper (const string orig) {
    string str = orig;
    for(int c = 0, l = str.length(); c < l; c++) {
        str[c] = toupper(str[c]);
    }
    return str;
}

string strToLower (const string orig) {
    string str = orig;
    for(int c = 0, l = str.length(); c < l; c++) {
        str[c] = tolower(str[c]);
    }
    return str;
}

/**
 * \brief Trims the leading whitespace and trailing whitespace from the string.
 **/
void trim( string &orig ) {
	size_t lead = orig.find_first_not_of( " \t" );
	if( lead != string::npos ) orig = orig.substr( lead );
	size_t tail = orig.find_last_not_of( " \t" );
	if( tail != string::npos ) orig = orig.substr( 0, (tail + 1));
}

string getEnvironmentVariable( const char *name ) { 
	char   *variable = getenv( name );
	string variableString = "";
	if( variable == NULL ) {
		warning( "Environment variable %s is not set.\n", name );
	} else {
		variableString = variable;
	}
	return variableString;
}
