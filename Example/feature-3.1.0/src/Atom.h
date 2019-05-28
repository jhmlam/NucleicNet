// FEATURE
// See COPYING and AUTHORS for copyright and license agreement terms.

// $Id: Atom.h 1609 2013-12-10 08:03:27Z mikewong899 $

// Header file for Atom.cpp

#ifndef SATOM_H
#define SATOM_H

#include <cctype>       // tolower
#include <cmath>        // pow
#include <iostream>
#include <sstream>
#include <iomanip>
#include "Point3D.h"
#include "Parser.h"     // IsStringIn
#include "Utilities.h"
#include "const.h"      // PI

using std::endl;
using std::left;
using std::ostream;
using std::setfill;
using std::setprecision;
using std::setw;
using std::stringstream;

/**
 * @class Atom
 * @brief A parser and data structure to store atom information
 * @ingroup protein_module
 * 
 * The purpose of the Atom class is to parse ATOM entries in a PDB file and to
 * store information provided by the Protein class. A Protein has one or more
 * Atom instances. Currently the Atom class stores PDB structure, DSSP
 * secondary structure, and AMBER force field information. 
 * 
 * Some of the information from these databases infer other information. For
 * example, by knowing the residue name, one can determine if the residue is
 * polar, non-polar, acidic, basic, or neutral. If you know the residue and
 * the atom name, you can find a table from a biochemistry text that will tell
 * you whether or not the atom participates in an s-bond or pi-bond, or if the
 * atom is part of an aromatic ring or an aliphatic chain.
 *  
 * The primary data from the different databases plus the inferred data make
 * up some of the possibilities of physicochemical properties that a
 * researcher might want to investigate for modeling protein function.
 *
 * Given the purpose and role of this class, developers should resist the
 * temptation to have the Atom class participate in any calculations other
 * than parsing and storing data. Some of the data that is OK to store are
 * trivial combinations of primary data (such as concatenating several
 * fields). Other classes can collaborate with the Atom class to perform
 * meaningful computations. A good example of this is the AtomProperties
 * class.
 **/
class Atom {
public:
	/**
	 * @brief Uniquely identifies a residue for a given atom.
	 * @note Consider replacing this with a string such as ResidueLabel
	 * @note The residue name is not part of this data structure; DSSP
	 * files use a single-character code for residue name. For the most part,
	 * the single-character code is the standard amino acid residue single
	 * character code, however, there are additional letters and uppercase and
	 * lowercase. Trying to map the residue name from the single character code
	 * using the standard amino acid codes causes the overall FEATURE behavior
	 * to differ from the Golden Revision.
	 **/
    struct ResidueID {
        long int residueSequence;
        char chainID;
        char insCode;
		ResidueID();
        ResidueID( long int, char, char );
		ResidueID( Atom * );
        
        bool operator== (const ResidueID& other) const {
            return residueSequence == other.residueSequence
                && chainID         == other.chainID
                && insCode         == other.insCode;
        }
        
       bool operator!= (const ResidueID& other) const {
            return residueSequence != other.residueSequence
                || chainID         != other.chainID
                || insCode         != other.insCode;
        }

        bool operator< (const ResidueID& other) const {
			if     ( chainID         < other.chainID )         return true;
			else if( chainID         > other.chainID )         return false;
			else if( residueSequence < other.residueSequence ) return true;
			else if( residueSequence > other.residueSequence ) return false;
			else if( insCode         < other.insCode )         return true;
			else                                               return false;
        }

		friend ostream &operator<<( ostream &, Atom::ResidueID & );
    };

    typedef enum {
        Coil, Strand, Bend, Turn, Bridge, Helix3, Helix4, Helix5, Het
    } SecondaryStructure;

	protected:
	// PDB Data
    long int           serialNumber;
    string             name;                // N, C, O, etc.
    string             fullName;            // N, CA, CB, CG, etc.
    char               altLocation;
    string             residueName;         // ASP, GLU, LYS, etc.
    char               chainID;
    long int           residueSequence;     // residue #
    char               insCode;
    string             residueFullSequence; // chain+res+ins
    Point3D            coord;
    double             occupancy;
    double             bFactor;
	ResidueID          residueID;

    // DSSP Data
    SecondaryStructure secondaryStructure;
    long int           solventAcc;
    bool               isHetatm;
    string             atomType;
    double             partialCharge;

	// Atom selection
	bool               mark;

    // Public Functions
	public:
	Atom();
	Atom( Atom * );
	~Atom();
    static Atom *parsePDBAtom(const string & line);
    static void setResidueFullSequence( Atom * );
    
    bool operator== (const Atom &b) const;
    bool operator< (const Atom &b) const;
	friend ostream &operator<<( ostream &, Atom & );

    string getResidueLabel();

	// Atom getters and setters
	string    getAtomType()                         { return atomType;           };
	char      getChainID()                          { return chainID;            };
	Point3D	* getCoordinates()                      { return &coord;             };
	string    getFullName()                         { return fullName;           };
	bool      isMarked()                            { return mark;               };
	string    getName()                             { return name;               };
	char      getInsCode()                          { return insCode;            };
	bool      getIsHetatm()                         { return isHetatm;           };
	double    getPartialCharge()                    { return partialCharge;      };
	string    getResidueName()                      { return residueName;        };
	long int  getResidueSequence()                  { return residueSequence;    };
	long int  getSerialNumber()                     { return serialNumber;       };
	long int  getSolventAccessibility()             { return solventAcc;         };
	SecondaryStructure getSecondaryStructure()      { return secondaryStructure; };
	ResidueID getResidueID()                        { return residueID;          };
	double    getDistance( Atom *b )                { return coord.getDistance( b->getCoordinates() ); };

	bool      nameIsIn( const char * value )        { string set( value ); return isStringIn( set, name );        };
	bool      fullNameIsIn( const char * value )    { string set( value ); return isStringIn( set, fullName );    };
	bool      residueNameIsIn( const char * value ) { string set( value ); return isStringIn( set, residueName ); };

	void      clearMark()                           { mark               = false;              };
	void      setAtomType( const char *value )      { atomType           = value;              };
	void      setAtomType( const string value )     { atomType           = value;              };
	void      setFullName( const char *value )      { fullName           = value;              };
	void      setFullName( string value )           { fullName           = value;              };
	void      setIsHetatm( bool value )             { isHetatm           = value;              };
	void      setMark()                             { mark               = true;               };
	void      setPartialCharge( double value )      { partialCharge      = value;              };
	void      setResidueName( const char *value )   { residueName        = value;              };
    void      setSecondaryStructure(char);
	void      setSecondaryStructureToHetatm()       { secondaryStructure = Het;                };
	void      setSolventAcc( long int value )       { solventAcc         = value;              };
	void      toggleMark()                          { mark               = ! mark;             };
};

#endif
