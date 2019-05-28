// FEATURE
// See COPYING and AUTHORS for copyright and license agreement terms.

// $Id: Atoms.cc 1128 2011-09-20 23:43:19Z mikewong899 $

//Atoms.cpp

#include "Atoms.h"

/**
 * @brief Filters atoms that are Heteroatoms (non-residue atoms)
 **/
bool MarkCriteria::Hetatm::matches( Atom &atom ) {
    return (atom.getIsHetatm());
}

/**
 * @brief Filters atoms based on the given residue name
 **/
bool MarkCriteria::ResidueName::matches( Atom &atom ) {
    return ( atom.getResidueName() == residueName );
}

Atoms::~Atoms() {
    clear();
}

void
Atoms::addAtom(Atom * atom ) {
	push_back( atom );
}

void
Atoms::addAtoms(Atoms *pAtoms ) {
    int num = pAtoms->size();
    for (int i = 0; i < num; i++)
        addAtom(pAtoms->at(i) );
}

void
Atoms::clearAtomMarks() {
	Atom *atom;
    for( Atoms::iterator i = begin(); i != end(); i++ ) {
		atom = *i;
		atom->clearMark();
    }
}

void
Atoms::invertAtomMarks() {
	Atom *atom;
    for( Atoms::iterator i = begin(); i != end(); i++ ) {
		atom = *i;
		atom->toggleMark();
    }
}

Atoms *
Atoms::extractMarkedAtoms() {
    Atoms *markedAtoms = new Atoms;
	Atom *atom;

    for (Atoms::iterator i = begin(); i != end(); i++ ) {
		atom = *i;
        if( atom->isMarked() ) {
            markedAtoms->addAtom( atom );
        }
    }
    return (markedAtoms);
}

int
Atoms::markAtoms( MarkCriteria &criteria ) {
    int touched = 0;
	Atom *atom;

    for (Atoms::iterator i = begin(); i != end(); i++ ) {
		atom = *i;
        if( criteria.matches( *atom )) {
			atom->setMark();
            touched++;
        }
    }

    return touched;
}

ostream &operator<<( ostream & stream, Atoms & atoms ) {
	Atom *atom;
	for( Atoms::iterator i = atoms.begin(); i != atoms.end(); i++ ) {
		atom = *i;
		stream << (*atom) << endl;
	}
    
	return stream;
}
