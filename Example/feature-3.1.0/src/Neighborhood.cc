/* $Id: Neighborhood.cc 1122 2011-09-19 22:53:12Z mikewong899 $ */
/* Copyright (c) 2002 Mike Liang.  All rights reserved. */

#include "Neighborhood.h"

// ========================================================================
ostream & operator <<(ostream &outs, const Neighborhood::Sector &sector) {
    return outs << "(" << sector.x << ", " << sector.y << ", " << sector.z << ")";
}

// ========================================================================
ostream & operator <<(ostream &outs, const Neighborhood &nbr) {
    outs << "Spacing: " << nbr.spacing << endl;
    outs << "Sectors: " << endl;
    for (Neighborhood::NearbyAtoms::const_iterator iter = nbr.nearbyAtoms.begin(); iter != nbr.nearbyAtoms.end(); iter++) {
        Neighborhood::NearbyAtoms::value_type item = *iter;
        outs << item.first << endl;
    }
    return outs;
}

Neighborhood::Neighborhood( Atoms *atoms, float spacing ) : spacing( spacing ) {
	Atom        *atom = NULL;
	Point3D     *v    = NULL;
	for( Atoms::iterator i = atoms->begin(); i != atoms->end(); i++ ) {
		atom = (*i);
		v = atom->getCoordinates();
		int  x = (int) floor(v->x / spacing);
		int  y = (int) floor(v->y / spacing);
		int  z = (int) floor(v->z / spacing);
        Sector k = Sector( x, y, z );

        if (nearbyAtoms.find( k ) == nearbyAtoms.end()) {
            nearbyAtoms[ k ] = new Atoms();
        }
        nearbyAtoms[ k ]->push_back( atom );
	}
}

// ========================================================================
Neighborhood::~Neighborhood() {
    for(NearbyAtoms::iterator it = nearbyAtoms.begin(); it != nearbyAtoms.end(); it++) {
        delete it->second; // Delete the atom list
        it->second = NULL;
    }
    nearbyAtoms.clear();
}

// ========================================================================
// @brief Given a 3-D coordinate, find all neighbors within one to that point
Atoms* Neighborhood::getNeighbors(const Point3D& coord) {
	int x0 = (int) floor(coord.x / spacing);
	int y0 = (int) floor(coord.y / spacing);
	int z0 = (int) floor(coord.z / spacing);

    Sector  center     = Sector( x0, y0, z0 );
    Atoms  *list       = new Atoms();
    Atoms  *neighbors;

	// ===== ITERATE OVER ALL VOXELS WITHIN ONE UNIT DISTANCE (NORMALIZED BY OUTERMOST SHELL RADIUS)
    for (int x = center.x - 1; x <= center.x + 1; x++) {
        for (int y = center.y - 1; y <= center.y + 1; y++) {
            for (int z = center.z - 1; z <= center.z + 1; z++) {

				// ===== IF A NEIGHBOR EXISTS, ADD IT TO THE LIST
                Sector sector = Sector(x, y, z);
                if (nearbyAtoms.find( sector ) != nearbyAtoms.end()) {
                    neighbors = nearbyAtoms[ sector ];
                    list->insert( list->end(), neighbors->begin(), neighbors->end() );
                }
            }
        }
    }
    return list;
}

Atoms* Neighborhood::getNeighbors( Atom * atom ) {
	Point3D *coord = atom->getCoordinates();
	return getNeighbors( *coord );
}
