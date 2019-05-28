// FEATURE
// See COPYING and AUTHORS for copyright and license agreement terms.

// $Id: Protein.h 1285 2012-03-22 01:12:42Z mikewong899 $

// Header file for Protein.cpp

#ifndef PROTEIN_H
#define PROTEIN_H

#include "Atoms.h"
#include "Parser.h"
#include "Utilities.h"
#include "Neighborhood.h"
#include "ForceField.h"
#include "ResidueTemplate.h"
#include <zlib.h>
#include "gzstream.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>

using std::string;
using std::vector;
using std::map;
using std::set;

/**
 * @ingroup protein_module
 * @brief A multisource data parser that stores the molecular physicochemical
 * characteristics and protein structure.
 *
 * The purpose of the Protein class is to integrate entries from multiple
 * protein/molecular databases (such as PDB, DSSP, and AMBER force field
 * files), extract relevant information, and stores the information into Atom
 * objects. 
 *
 * The Protein object achieves this by first parsing each PDB ATOM entry and
 * creates one Atom object per entry. 
 *
 * The Protein object parses the DSSP secondary structure information and
 * applies the secondary structure information for each atom for each residue.
 * Secondary structure information includes: secondary structure (e.g. alpha
 * helix or beta sheet) and solvent accessibility.
 *
 * Finally, the Protein object uses an instance of the Protein::Forcefield
 * class to read AMBER force field files provided with FEATURE. These
 * force field files provide partial charge and atom type information for each
 * atom.
 *
 * In the case of nucleic acids or proteins with nucleic acids, MCAnnotate
 * is software that looks for secondary structure motifs similar to DSSPcmbi.
 * MCAnnotate (MCA) files are converted to DSSP files and the biomolecule
 * structure is read from the PDB just as if it were a normal protein.
 *
 * Pursuant to the purpose of the class's role, developers should avoid using
 * the Protein class to perform anything more than parsing and storing data to
 * Atom instances. The Protein class must be responsible for managing the
 * memory allocation and deallocation of its component object instances (e.g.
 * Atoms).
 **/
class Protein {
	public:
		/**
		 * @ingroup protein_module
		 * @brief Datastructure with supports the idea that proteins are made
		 * of chains of amino acid residues
		 *
		 * Both the PDB and DSSP protein description files have the view that
		 * proteins are chains of amino acids. This class makes it possible to
		 * iterate over each residue in a Protein and to iterate over every
		 * atom in each residue.
		 **/
		class Residues : public map<Atom::ResidueID, Atoms *> {};

	protected:
		string name;
		Residues           uniqueResidues;
		Atoms              atoms;
		Atoms              hetAtoms;

		int modelNum;

		bool addAtomToUniqueResidues( Atom * );
		void parseAtom(     string & line );
		void parseHetatm(   string & line );
		void parseHeader(   string & line );
		void parseSSBond(   string & line );
		void parsePDBLine(  string & line );
		void parseDSSPLine( string & line );

		/**
		 * @brief Adapter class that applies the AMBER force fields to the FEATURE protein structure model
		 * @ingroup protein_module
		 *
		 * Integrates PDB information with the AMBER force field information from ResidueTemplate and ForceField classes.
		 **/
		class ForceFields {
			private:
				ForceFields();

			protected:
				static const string        defaultForceFieldParametersFilename;
				static const string        defaultResidueTemplateFilename;
				static const float         disulfideBondLength;
				static bool                initialized;
				static ForceFields        *instance;

				ForceField                *forceField;
				ForceField::Terminus      *forceFieldTerminus;
				ResidueTemplate           *residueTemplate;
				ResidueTemplate::Terminus *residueTemplateTerminus;
				set<string>               *nTerminalAtoms;
				set<string>               *cTerminalAtoms;

				void determinePosition( Atoms * );
				bool terminalAtomsMatchAny( set<string> *, set<string> * );
				void setTypeAndCharge( string, Atoms * );
				void setTypeAndCharge( string, Atom * );
				void setTypeAndChargeForDisulfideBridge( Residues *, Atoms *, Neighborhood * );
				Atom *findSulfurA( Atoms * );
				Atom *findSulfurB( Atoms *, Atom * );

			public:
				static ForceFields *getInstance();
				void setTypeAndCharge( Protein * );
		};

	public:
		Protein();
		Protein( const string &filename);
		~Protein();

		void readPDBFile( const string &filename );
		void readDSSPFile( const string &filename );

		const char      *getName();
		Atom            *operator[]( int i ) { return atoms[ i ]; };
		Atoms::iterator  begin()             { return atoms.begin(); };
		Atoms::iterator  end()               { return atoms.end(); };
		int              size()              { return atoms.size(); };
		Atoms           *getAtoms()          { return &atoms; };
		Residues        *getUniqueResidues() { return &uniqueResidues; };
};

#endif
