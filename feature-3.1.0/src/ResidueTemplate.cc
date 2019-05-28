/* $Id: ResidueTemplate.cc 1161 2011-10-14 22:27:31Z mikewong899 $ */
/* Copyright (c) 2002 Mike Liang.  All rights reserved. */

#include "ResidueTemplate.h"

ResidueTemplate::ResidueTemplate( string filename ) {
    string path = getEnvironmentVariable( FeatureDirEnvStr );
    string fullFileName = joinPath( path, filename);
    load(fullFileName);
}

void ResidueTemplate::load( string filename ) {
    debug(3, "Loading Residue Templates %s...\n",filename.c_str());
    ifstream ins( filename.c_str() );
    if (!ins) fatal_error("ResidueTemplate: Could not find AMBER residue template file '%s'. Check environment variable '%s'\n",filename.c_str(), FeatureDirEnvStr);

    string token;
    enum { INITIAL, GET_SET_NAME, GET_RESIDUE_NAME, GET_PDB_ATOM_NAME, GET_AMBER_ATOM_NAME } state = INITIAL;
    Terminus    *terminus = NULL;
    AtomAliases *alias    = NULL;
    string       atomAlias;

    while (ins >> token) {
        if ( token == "set:" ) {
            state = GET_SET_NAME;
        } else if ( token == "residue:" ) {
            state = GET_RESIDUE_NAME;
        } else if ( token == "aliases:" ) {
            state = GET_PDB_ATOM_NAME;
        } else {
            switch (state) {
            case INITIAL:
                fatal_error( "ResideTemplate: Error in parsing file '%s'. Unwanted initial token: [%s]\n", filename.c_str(), token.c_str() );
                break;
            case GET_SET_NAME:
                if ( token == "normal" ) {
                    terminus = &this->normal;
                } else if ( token == "n_terminal" ) {
                    terminus = &this->n_terminal;
                } else if ( token == "c_terminal" ) {
                    terminus = &this->c_terminal;
                } else {
                    fatal_error( "ResidueTemplate: Error in parsing file '%s'. Unknown set name: %s\n", filename.c_str(), token.c_str() );
                    terminus = NULL;
                    state = INITIAL;
                    break;
                }
                break;
            case GET_RESIDUE_NAME:
                if (terminus == NULL) {
                    warning("curMap == NULL\n");
                    state = INITIAL;
                    break;
                }
                alias = new ResidueTemplate::AtomAliases();
                (*terminus)[ token ] = alias;
                break;
            case GET_PDB_ATOM_NAME:
                if (terminus == NULL) {
                    warning("curTmpl == NULL\n");
                    state = INITIAL;
                    break;
                }
                atomAlias = token;
                state = GET_AMBER_ATOM_NAME;
                break;
            case GET_AMBER_ATOM_NAME:
                if (terminus == NULL) {
                    warning("curTmpl == NULL\n");
                    state = INITIAL;
                    break;
                }
                (*alias)[ atomAlias ] = token;
                state = GET_PDB_ATOM_NAME;
                break;
            }
        }
    }
    ins.close();
}

ResidueTemplate::Terminus::~Terminus() {
	Terminus::iterator i;
	for( i = begin(); i != end(); i++ ) {
		delete i->second;
	}
	clear();
}

// ========================================================================
ostream& operator<< (ostream& outs, const ResidueTemplate::AtomAliases& aam) {
    ResidueTemplate::AtomAliases::const_iterator iter;
    for (iter = aam.begin(); iter != aam.end(); iter++) {
        outs << "            " << (*iter).first << " " << (*iter).second << endl;
    }
    return outs;
}

// ========================================================================
ostream& operator<< (ostream& outs, const ResidueTemplate::Terminus& rtm) {
    ResidueTemplate::Terminus::const_iterator iter;
    for (iter = rtm.begin(); iter != rtm.end(); iter++) {
        outs << "    " << "residue: " << (*iter).first << endl;
    	outs << "        " << "aliases:" << endl;
        outs << *(*iter).second;
    }
    return outs;
}

// ========================================================================
ostream& operator<< (ostream& outs, const ResidueTemplate& set) {
    outs << "set: normal" << endl << set.normal << endl;
    outs << "set: n_terminal" << endl << set.n_terminal << endl;
    outs << "set: c_terminal" << endl << set.c_terminal << endl;
    return outs;
}
