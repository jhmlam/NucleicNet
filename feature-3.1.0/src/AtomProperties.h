// FEATURE
// See COPYING and AUTHORS for copyright and license agreement terms.

// $Id: AtomProperties.h 1615 2013-12-16 23:05:43Z mikewong899 $

//AtomProperties.h
#ifndef ATOMPROPERTIES_H
#define ATOMPROPERTIES_H
#define _USE_MATH_DEFINES

#include "unordered_map.h"
#include <cassert>
#include <cmath> // fabs, M_PI (Math constant PI)
#include <string>
#include "Atom.h"
#include "PropertyList.h"

using std::string;

/**
 * @brief Derives Property information from an Atom (typically a Neigbor atom)
 * to store in a EnvironmentShell (i.e. an Environment shell)
 *
 * @ingroup properties_module
 * 
 * An adapter class that collects Property information from Atom objects and
 * stores it to the EnvironmentShell. All the property values are converted to
 * integers; significant digit information depends on each Property.
 *
 * @note Also note that for Bayesian Classification, the properties MUST be
 * discrete (enumerable); for other classification methods, this limitation is
 * unnecessary
 *
 * @note C++ STL map accessor [] operator searches for a key; if the key
 * doesn't exist, it is added, and the default initializer for the value is
 * invoked. Therefore all non-declared properties will automatically have a
 * default value of 0.
 **/
class AtomProperties {
	protected:
		/**
		 * @brief Provides AtomProperties information such as Van der Waals
		 * radii, whether or not an atom is part of an aromatic structure
		 *
		 * A good deal of information is available that is can be inferred from
		 * PDB and DSSP data. This class provides a method to integrate this
		 * additional data by hard-coding lookup tables based on residue, atom
		 * name, or any other primary data from the PDB. 
		 **/
		class LookupTable : public unordered_map< string, double > {
			public:
				bool has( string );
		};

		/**
		 * @brief Manages LookupTable instances
		 *
		 * The LookupTables assist the AtomProperties class by using primary
		 * data from the PDB, DSSP, and AMBER force field files and finding
		 * secondary data in the LookupTable instances. Many atom properties
		 * are directly taken from the lookup results based on residue or atom
		 * name.
		 *
		 * This class follows the Singleton design pattern and is globally
		 * accessible. 
		 **/
		class LookupTables {
			protected:
				static bool isInitialized;
				static LookupTables *singleton;
				unordered_map< string, LookupTable > tables;
				LookupTables();

			public:
				static LookupTables *getInstance();
				LookupTable &get( string );
				void add( string, LookupTable & );
		};

		Atom                        *atom;
		string                       residueName;
		string                       fullName;
		string                       name;
		LookupTables                *lookupTables;

		double                       getCharge();
		double                       getHydrophobicity();
		double                       getRadius();
		bool                         isBackbone();

		void                         setAmide();
		void                         setAmine();
		void                         setCarbonyl();
		void                         setCharge();
		void                         setChargeWithHis();
		void                         setFunctionalGroup();
		void                         setHydrophobicity();
		void                         setHydroxyl();
		void                         setMobility();
		void                         setName();
		void                         setPosCharge();
		void                         setRingSystem();
		void                         setNegCharge();
		void                         setPartialCharge();
		void                         setPeptide();
		void                         setResidueName();
		void                         setResidueClass1();
		void                         setResidueClass2();
		void                         setSolventAccessibility();
		void                         setSecondaryStructure1();
		void                         setSecondaryStructure2();
		void                         setType();
		void                         setVDWVolume();

	public:
		Point3D                      coord;
		unordered_map< string, int > propertyValue;

		AtomProperties( Atom * );

};

#endif
