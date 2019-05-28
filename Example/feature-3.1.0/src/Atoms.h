// FEATURE
// See COPYING and AUTHORS for copyright and license agreement terms.

// $Id: Atoms.h 1128 2011-09-20 23:43:19Z mikewong899 $

// Header file for Atoms.cpp

#ifndef ATOMS_H
#define ATOMS_H

#include "Atom.h"
#include "const.h" // MaxProteinLength
#include <iostream>
#include <vector>
#include <string>

using std::ostream;
using std::string;
using std::vector;

/**
 * @brief Virtual delegate superclass that requires an implementation of the matches() method
 *
 * The purpose of this class is to require an implementation of the matches() method.
 * The matches() method returns true if the given atom matches the criteria defined
 * by the method, false otherwise.
 **/
class MarkCriteria {
	public:
		/**
		 * @returns True if the given atom matches the criteria, false otherwise
		 **/
		virtual bool matches( Atom &atom ) = 0;
		class ResidueName;
		class Hetatm;
};

/**
 * @brief Delegate class that determines if a given atom's residue name matches
 * the given residue name.
 *
 * To use this delegate class, create an instance of the class with a residue
 * name parameter (e.g. "HIS"). The delegate returns true for each atom that
 * matches the given residue name.
 **/ 
class MarkCriteria::ResidueName : public MarkCriteria {
	protected:
		string residueName;
	public:
		ResidueName( char *name )   { residueName = string( name ); };
		ResidueName( string &name ) { residueName = name; };
		bool matches( Atom &atom );
};

/**
 * @brief Delegate class that determines if a given atom is a hetatm
 * (heteroatom, i.e. non-residue atom).
 *
 * To use this delegate class, create an instance of the class. The delegate
 * returns true for each atom that is a hetatm.
 **/ 
class MarkCriteria::Hetatm : public MarkCriteria {
	public:
		bool matches( Atom &atom );
};

/**
 * @brief An array of Atom instances
 * @ingroup protein_module
 *
 * The purpose of the Atoms class is to store and filter atoms by selection
 * criteria.
 *
 * The Atoms class simply points to atoms instantiated by other classes;
 * therefore there ought to be no reason to delete an instance of the Atoms
 * class unless the instance is a component of another class that instantiates
 * Atom objects (e.g. Protein). 
 *
 * Other classes simply use Atoms to refer to data structures managed by other
 * classes. An example of a class that has Atom instances that shouldn't be
 * deallocated is Neighborhood (a Neighborhood describes spatial relationships
 * between Atom instances in the context of a Protein; the Protein handles the
 * instantiation and deletion of the Atom instances, the Neighborhood just
 * refers to them.
 *
 * @note Formerly known as CAtomArray
 **/
class Atoms : public vector<Atom *> {
	public:
		~Atoms();

	friend ostream &operator<<( ostream &, Atoms & );
	void   addAtom( Atom *pAtom );
   	void   addAtoms( Atoms *pAtoms );
	void   clearAtomMarks();
	Atoms *extractMarkedAtoms();
	void   invertAtomMarks();
	int    markAtoms( MarkCriteria & );
	Atoms &operator=(Atoms &ref);
};

#endif
