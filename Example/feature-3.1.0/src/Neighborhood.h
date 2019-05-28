/* $Id: Neighborhood.h 1122 2011-09-19 22:53:12Z mikewong899 $ */
/* Copyright (c) 2002 Mike Liang.  All rights reserved. */

#ifndef NEIGHBORHOOD_H
#define NEIGHBORHOOD_H

#include <iostream>
#include <cmath>
#include <cstdio>
#include "stl_types.h"
#include "Point3D.h"
#include "Atoms.h"

using std::ostream;
using std::endl;

/**
 * @brief Finds nearby Atom objects to a Cartesian point; neighbors are referenced by their order of appearance in a PDB
 * @ingroup euclidean_module
 * 
 * Creates a sphere around the given points. The sphere is composed of a set 
 * of voxels (volumetric pixels). The coordinate system for the sphere is 
 * normalized to a given spacing parameter. Each voxel is associated 
 * with a list of zero or more points.
 **/
class Neighborhood {
	protected:

		/**
		 * @brief A localized region of the Neighborhood (actually a box in 3D
		 * space, i.e. a voxel)
		 * @ingroup euclidean_module
		 **/
		struct Sector {
			int x;
			int y;
			int z;

			Sector() {}
			Sector(int x,int y,int z) : x(x), y(y), z(z) {}

			/**
			 * @brief compares two blocks to provide an ordering from lower, left,
			 * near to upper, right, far
			 **/
			struct compare {
				bool operator()(const Sector& a, const Sector& b) const {
					if (a.x < b.x) return true;
					if (a.x > b.x) return false;
					if (a.y < b.y) return true;
					if (a.y > b.y) return false;
					if (a.z < b.z) return true;
					return false;
				}
			};
		};

		typedef map<Sector, Atoms *, Sector::compare> NearbyAtoms;
		NearbyAtoms nearbyAtoms;
		float spacing;

	public:
		Neighborhood( Atoms *atoms, float spacing );
		~Neighborhood();

		Atoms* getNeighbors( const Point3D& coord );
		Atoms* getNeighbors( Atom *atom );

    friend ostream & operator << (ostream &outs, const Neighborhood &nbr);
    friend ostream & operator << (ostream &outs, const Neighborhood::Sector &sector);
};

// ========================================================================
ostream & operator << (ostream &outs, const Neighborhood::Sector &sector);

// ========================================================================
ostream & operator << (ostream &outs, const Neighborhood &nbr);

#endif
