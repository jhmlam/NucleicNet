/* $Id: ForceField.cc 1161 2011-10-14 22:27:31Z mikewong899 $ */
/* Copyright (c) 2002 Mike Liang.  All rights reserved. */

#include "ForceField.h"

// ========================================================================
ostream& operator<< (ostream& outs, const ForceField::AtomID & atomID) {
    return outs << atomID.residueName << " " << atomID.atomName;
}

// ========================================================================
ostream& operator<< (ostream& outs, const ForceField::Parameters & parameter) {
    return outs << "type: " << parameter.type << " charge: " << parameter.charge;
}

// ========================================================================
ostream& operator<< (ostream& outs, const ForceField::Terminus& terminus) {
	ForceField::Terminus::const_iterator i;
    for ( i = terminus.begin(); i != terminus.end(); i++) {
        outs << "    " << i->first << " " << *(i->second) << endl;
    }
    return outs;
}

// ========================================================================
ostream& operator<< (ostream& outs, const ForceField & set) {
    outs << "set: normal"     << endl << set.normal     << endl;
    outs << "set: n_terminal" << endl << set.n_terminal << endl;
    outs << "set: c_terminal" << endl << set.c_terminal << endl;
    return outs;
}

// ========================================================================
ForceField::ForceField( string filename ) {
    string path = getEnvironmentVariable( FeatureDirEnvStr );
    string fullFileName = joinPath( path, filename);
    this->load( fullFileName );
}

// ========================================================================
void ForceField::load( string filename) {
    debug(3, "Loading ForceField Parameters %s...\n", filename.c_str());
    ifstream ins( filename.c_str() );
    if (!ins) fatal_error( "ForceField: Could not find AMBER force field file '%s'; check environment variable %s.",filename.c_str(), FeatureDirEnvStr );

	string token;
    enum { INITIAL, GET_SET_NAME, GET_RESIDUE_NAME, GET_ATOM_NAME, GET_TYPE, GET_CHARGE } state = INITIAL;
    ForceField::Terminus* terminus = NULL;
    ForceField::AtomID atomID;
    ForceField::Parameters parameters;

    while (ins >> token) {
        if ( token == "set:" ) {
            state = GET_SET_NAME;
        } else if ( token == "type:" ) {
            state = GET_TYPE;
        } else if ( token == "charge:" ) {
            state = GET_CHARGE;
        } else {
            switch(state) {
            case INITIAL:
                fatal_error( "ForceField: Error in parsing file '%s'. Unwanted initial token: [%s]\n", filename.c_str(), token.c_str() );
                break;
            case GET_SET_NAME:
                if ( token == "normal" ) {
                    terminus = &this->normal;
                } else if ( token == "n_terminal" ) {
                    terminus = &this->n_terminal;
                } else if ( token == "c_terminal" ) {
                    terminus = &this->c_terminal;
                } else {
                    fatal_error( "ForceField: Error in parsing file '%s'. Unknown set name: %s\n", filename.c_str(), token.c_str() );
                    terminus = NULL;
                    state = INITIAL;
                    break;
                }
                state = GET_RESIDUE_NAME;
                break;
            case GET_RESIDUE_NAME:
                atomID.residueName = token;
                state = GET_ATOM_NAME;
                break;
            case GET_ATOM_NAME:
                atomID.atomName = token;
                break;
            case GET_TYPE:
                parameters.type = token;
                break;
            case GET_CHARGE:
				stringstream doubleValue( token );
				doubleValue >> parameters.charge;
                (*terminus)[atomID] = new ForceField::Parameters( parameters );
                state = GET_RESIDUE_NAME;
                break;
            }
        }
    }
    ins.close();
}

ForceField::Terminus::~Terminus() {
	Terminus::iterator i;
	for( i = begin(); i != end(); i++ ) {
		delete i->second; // Free the Parameters object
	}
	clear();
}
