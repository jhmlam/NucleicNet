/* $Id: ForceField.h 1122 2011-09-19 22:53:12Z mikewong899 $ */
/* Copyright (c) 2002 Mike Liang.  All rights reserved. */

#ifndef FORCE_FIELD_PARAMETERS_H
#define FORCE_FIELD_PARAMETERS_H

#include <cstdlib>
#include "stl_types.h"
#include <fstream>
#include <sstream>
#include <string>
#include "Utilities.h"

using std::ifstream;
using std::ostream;
using std::stringstream;
using std::endl;

/**
 * @brief Datastructure which correlates the AMBER force field atom types and
 * partial charges for each atom of each residue.
 *
 * @ingroup protein_module
 *
 * @note This class uses the Singleton design pattern
 *
 * Force field equations are approximations to quantum mechanical equations to
 * calculate potential energy of a molecular system. AMBER Force Field is a
 * public database of parameters for force field equations. These parameters
 * are appied to force field equations (some of which are similar to
 * mechanical spring equations) to rather accurately predict kinetic energy
 * and structural states for large molecules.
 *
 * The purpose of the ForceField class is to parse AMBER Force Field database
 * files and allow Protein instances to use the <b>atom type</b> and 
 * <b>partial charge</b> information contained in the files.
 *
 * Force field parameters also depend on the residue's position within the
 * protein. The three different positions are: N-terminus, C-terminus, and
 * and in the middle .
 **/
class ForceField {
	public:

		/**
		 * @brief Generally identifies an atom in a residue (e.g. all CYS have
		 * an SG atom) as opposed to a particular SG atom in a particular CYS
		 * residue.
		 **/
		class AtomID {
			public:
				string residueName;
				string atomName;

				AtomID() {}
				AtomID(string residueName, string atomName) : residueName(residueName), atomName(atomName) {}

				/**
				 * @brief compares two AtomID objects; if they are equal, the
				 * comparator returns true, false otherwise.
				 **/
				struct compare {
					bool operator() (const AtomID & a, const AtomID & b) const {
						int diff = a.residueName.compare( b.residueName );
						if (diff < 0) return true;
						if (diff > 0) return false;
						if (a.atomName.compare( b.atomName ) < 0) return true;
						return false;
					}
				};
		};

		/**
		 * @brief Holds partial charge and atom type information
		 *
		 * @ingroup protein_module
		 *
		 * Force field atom types depend on element and hybridization, among
		 * other factors. 
		 *
		 * Force field charge represents the partial charge for each atom.
		 **/
		class Parameters {
			public:
				string type;
				float charge;
		};

		/**
		 * @brief Stores a set of ForceField::Parameters for each position in
		 * the peptide chain: C-terminus, N-terminus, and middle.
		 **/
		class Terminus : public map<AtomID, Parameters*, AtomID::compare> {
			public:
				~Terminus();
		};

		public:
			Terminus normal;
			Terminus n_terminal;
			Terminus c_terminal;

			ForceField( string filename );
			~ForceField();

		private:
			void load( string filename );
};


ostream & operator<<(ostream& outs, const ForceField::AtomID & );
ostream & operator<<(ostream& outs, const ForceField::Parameters & );
ostream & operator<<(ostream& outs, const ForceField::Terminus & );
ostream & operator<<(ostream& outs, const ForceField & );

#endif
