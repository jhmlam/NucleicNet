// $Id: Points.cc 1534 2013-07-12 01:06:26Z teague $
// Copyright (c) 2004 Mike Liang. All rights reserved

#include "Points.h"
#include <string.h>
#define BUF_SIZE 1024


Point::Point() : coord(0,0,0) {
}

Point::Point(const Point3D &coord) : coord(coord) {
}

Point::Point(double x, double y, double z) {
    coord.x = x;
    coord.y = y;
    coord.z = z;
}

/**
 * @brief Parses a coordinate entry from a PDB
 **/
string Point::parse( string &data ) {
	string pdbid;
	stringstream buffer( data );

	buffer >> pdbid;
	buffer >> coord.x;
	buffer >> coord.y;
	buffer >> coord.z;
	getline( buffer, description );

	// Trim leading whitespace
	size_t start = description.find_first_not_of( " \t" );
	if( start != string::npos ) description = description.substr( start );

	return pdbid;
}

Point::~Point() {}

/**
 * @brief Prints the x, y, and z coordinates and optionally, description
 **/
void Point::write(FILE *outs) {
    fprintf( outs, "%g",   this->coord.x );
    fprintf( outs, "\t%g", this->coord.y );
    fprintf( outs, "\t%g", this->coord.z );
    if( ! this->description.empty() )
        fprintf( outs, "\t%s", this->description.c_str() );
}

Point3D
Point::getCoordinates() {
	return coord;
}

void
Point::setDescription( string &description ) {
	this->description = description;
}

Points::Points() {
}

Points::~Points() {
    vector<Point *>::iterator iter;
    // deallocate elements of points
    for (iter = points.begin(); iter != points.end(); iter++) {
        delete (*iter);
    }
}

/**
 * @brief Returns the number of points in the list
 **/
int Points::numPoints() {
    return points.size();
}

/**
 * @brief Returns Point in Points group at a given index
 **/
Point *Points::getPoint(int index) {
    return points[index];
}

/**
 * @brief Adds a Point to the end of a Points group
 **/
void Points::append(Point *value) {
    return points.push_back(value);
}

PointFile::PointFile(const char *filename) {
    if ( filename ) 
        read( filename );
}

/**
 * @brief Gets the Points group for a given PDB ID
 **/
Points *PointFile::getPoints(const string &pdbid) {
    return lookup(pdbid);
}

/**
 * @brief Reads in a PointFile from a file
 **/
void PointFile::read( const char *filename ) {
    this->filename = string( filename );

    igzstream infile;

    if ( this->filename == "-" ) {
        infile.open( stdin );
    } else {
        infile.open( filename );
    }

    if ( ! infile ) {
        warning("Point file '%s' not found\n", filename);
        return;
    }

	string buffer;
	while( getline( infile, buffer )) {
		if( buffer.empty() ) continue;

		Point *point = new Point();
		string pdbid = point->parse( buffer );

		Points *point_list;
		point_list = lookup( pdbid, true );
		point_list->append( point );
	}
}

/**
 * @brief Retrieves or creates a Points group given a hash key
 * @param key Hash key
 * @param create If true, create a new Points and entry in hash table; do nothing otherwise
 **/
Points *PointFile::lookup( const string key, bool create ) {
    PdbPoints::iterator iter;
    iter = map.find(key);
    if (iter == map.end()) {
        if (create) {
            Points *new_value = new Points;
            map[key] = new_value;
            return new_value;
        } else {
            return NULL;
        }
    }
    return (*iter).second;
}

PointFile::~PointFile() {
    PdbPoints::iterator iter;
    // deallocate elements of map
    for (iter = map.begin(); iter != map.end(); ) {
        PdbPoints::iterator cur_iter = iter++;
        delete (*cur_iter).second;
    }
}

/**
 * @brief Retrieves keys for the PointFile
 * @returns A vector of PDB ID keys (which are strings)
 **/
vector<string> *PointFile::getPdbids() {
    vector<string> *pdbids = new vector<string>;

    PdbPoints::iterator iter;
    for (iter = map.begin(); iter != map.end(); iter++) {
        pdbids->push_back((*iter).first);
    }

    sort(pdbids->begin(), pdbids->end());
    return pdbids;
}
