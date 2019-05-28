// FEATURE
// See COPYING and AUTHORS for copyright and license agreement terms.

// $Id: PropertyList.h 1615 2013-12-16 23:05:43Z mikewong899 $

#ifndef PROPERTYLIST_H
#define PROPERTYLIST_H

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include "const.h"
#include "Utilities.h"
#include "logutils.h"
#include "MetaData.h"
#include "unordered_map.h"

using std::string;
using std::vector;
using std::ifstream;
using std::stringstream;

/**
 * @brief A single property, described by a string, identified by a unique index, and the number of significant digits for values
 * @ingroup properties_module
 *
 * The significant digits are listed in terms of powers of 10 (10 for 1 significant figures, 100 for 2, 1000 for 3, etc.)
 **/
class Property {
	public:
		enum Source { PDB, DSSP, FORCE_FIELD };

	protected:
		string  name;
		int     index;
		double  divider;
		Source  source;

	public:
		Property(string name, int index, Source source, double divider);
		
		string&      getName();
		const int    getIndex() const;
		const double getDivider() const;
		const Source getSource() const;
		void         setIndex( int i ) { index = i; };
};

/**
 * @brief A collection of Property objects whose values are mutually exclusive.
 * @ingroup properties_module
 *
 * Property Groups are groups of properties that have mutually exclusive
 * options.  For example, an atom can be elemental carbon or elemental oxygen,
 * but not simultaneously both. When evaluating Shells, knowing
 * that properties are grouped together is helpful to reducing the total
 * number of calculations to evaluate.
 *
 **/
class PropertyGroup : public vector<string> {
	protected:
		string defaultName;
		int    defaultValue;
		string counterName;
		int    counterStep;

	public: 
		PropertyGroup();
		void   include( string );
		string getCounterName()  { return counterName; };
		int    getCounterStep()  { return counterStep; };
		string getDefaultName()  { return defaultName; };
		int    getDefaultValue() { return defaultValue; };
		void   setDefault( string, int = 1 );
		void   setCounter( string, int = 1 );
};

/**
 * @brief A global list of properties that relevant for the characterization of microenvironments.
 * @ingroup properties_module
 * 
 * This class enumerates the list of properties which are of interest for a particular
 * classification of proteins. For example, partial charge properties have been shown to 
 * be effective for characterizing proteins which have a serine protease function. On 
 * the other hand, functional R group properties have been shown to be effective for
 * characterizing metal ion chelators. This data structure differs from EnvironmentShell,
 * which focuses on the values for each of the properties in the PropertyList.
 **/

class PropertyList {
	private:
		PropertyList();
		
		// Check if the list of dividers has been initialized
		static bool initialized;
		static bool isLoaded;

		// A global name to double map of dividers.
		// One day this could be moved to a data file
		static unordered_map<string, Property *> definedProperties;

		// Populate the dividers map
		static void initialize();
		static void defineProperty( string, Property::Source, double );
		static void defineProperty( const char *, Property::Source, double divider );

	protected:
		static vector<Property *> list;
		static unordered_map<string, unsigned int> indexLookup;

	public:
		static vector<Property *>::iterator begin();
		static vector<Property *>::iterator end();
		static vector<Property *>& getList();
		static unordered_map< string, PropertyGroup > propertyGroup;
		static Property *get(const int index);
		static Property *get(string);
		static Property *get(const char *name);
		static double    getDivider( string );
		static void      enable( const char * );
		static void      enable( string );
		static bool      has(string);
		static bool      has(const char *name);
		static bool      isDefined( string );
		static void      load( const string& file );
		static void      load( Metadata * );
		static void      load();
		static void      loadDefaults();
		static bool      loaded ();
		static void      set( Metadata *);
		static int 		 size();
		
};
         
#endif
