// $Id: Points.h 1285 2012-03-22 01:12:42Z mikewong899 $
// Copyright (c) 2004 Mike Liang. All rights reserved
#ifndef POINT_LIST_H
#define POINT_LIST_H

#if HAVE_CONFIG_H
#include "config.h"
#endif

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <zlib.h>
#include "gzstream.h"
#include "Point3D.h"
#include "Utilities.h"
#include "logutils.h"

using std::map;
using std::vector;
using std::string;
using std::stringstream;

/**
 * @brief A point in 3D Cartesian coordinates that also has a description
 * @ingroup euclidean_module
 **/
class Point {
	protected:
		Point3D coord;
		string  description;

	public:
		Point();
		Point(const Point3D & coord);
		Point(double x, double y, double z);
		~Point();

		Point3D getCoordinates();
		string  parse( string & );
		void    write( FILE * outs );
		void    setDescription( string & );
};

/**
 * @brief A list of points
 * @ingroup euclidean_module
 * @note this class is redundant to Point3DList; the latter has less information and should be removed
 **/
class Points {
	protected:
		vector<Point *> points;
	public:
		Points();
		~Points();

		void   append(Point *value);
		int    numPoints();
		Point *getPoint(int index);
};

typedef map<string, Points *> PdbPoints;

/**
 * @brief Contains a list of points for each PDB ID, parsed from a point file for site or non-site characterization
 * @ingroup euclidean_module
 * 
 * Parses a point file (extension .ptf) which locates an active site across a number of
 * PDB files. 
 **/
class PointFile {
	protected:
		string    filename;
		PdbPoints map;

	public:
		PointFile(const char *filename);
		~PointFile();

		void            getPdbids(const char ***_copy_keys, int *_num_keys);
		vector<string> *getPdbids();
		Points         *getPoints(const string &pdbid);
		Points         *lookup(const string key, bool create = false);
		void            read(const char *filename);
};


#endif
