// $Id: Environments.h 1615 2013-12-16 23:05:43Z mikewong899 $
// Copyright (c) 2004 Mike Liang. All rights reserved
#ifndef ENVIRONMENT_LIST_H
#define ENVIRONMENT_LIST_H

#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <string>
#include <zlib.h>
#include "gzstream.h"
#include "PropertyList.h"
#include "AtomProperties.h"
#include "stl_types.h"
#include "logutils.h"
#include "MetaData.h"
#include "unordered_map.h"

using std::string;
using std::stringstream;

/**
 * @brief Characterizes a Point in or near a protein. Contains an array of
 * shells (one Shell object per shell level) and descriptive information. An
 * Environments group has one Environment object per protein.
 *
 * @ingroup properties_module
 * @note formerly called EnvironmentList
 *
 * The purpose of the Environment class is to manage an array of Shell
 * instances and to manage and accumulate physicochemical properties from the
 * Shells. The Environment class uses the 
 **/
class Environment {
	public:
		/**
		 * @brief A collection of Property instances. An environment has one EnvironmentShell
		 * object per shell level. Each EnvironmentShell object has sums of AtomProperties.
		 * @ingroup properties_module
		 * @note Formerly called Properties and CPropertiesArray before that.
		 **/
		class Shell {
		protected:
			bool hasAtLeastOneAtom;
			int *property;
			void addGroup( AtomProperties *, PropertyGroup );
			void add( AtomProperties *, string  );
			void add(int valueToAdd, string propertyName);
			void get(int *propValue, const int propertyIndex);
			void get(int *propValue, string propertyName);

		public:
			Shell();
			~Shell();

			int &at(int index) {
				return property[index];
			}
			int      get( const int property );
			int      get( string propertyName );
			double   getValue( const int property );
			bool     hasAtoms() { return hasAtLeastOneAtom; };
			void     include( AtomProperties * atomPropertyValues );
			void     parse( stringstream & );
			void     write( FILE * );
		};

	protected:
		Shell **shells;
		string label;
		string description;
		int numShells;

	public:
		Environment(int numShells);
		~Environment();
		string           &getDescription()  { return description; };
		string           &getLabel()        { return label; };
		int               getNumShells()    { return numShells; };
		Shell            *getShell( int i ) { return shells[ i ]; };
		void              write(FILE * outs);
		void              parse( string );
};

/**
 * @brief Quantitatively characterizes the microenvironment of one or more proteins at one or more points. A list of Environment objects, read from a @ref feature_vector "Feature Vector file". 
 * @ingroup properties_module
 * 
 * The purpose of the Environments class is to manage an array of Environment
 * instances. Each Environments instance is intended to describe a class of
 * similar points across multiple proteins, where the classes are typically:
 * known functional, known non-functional, or unknown.
 * 
 * Environments are used in one of three ways:
 * @li Quantitatively characterizes the microenvironment of a protein at a number of points; or
 * @li Characterizes functionally equivalent protein sites across a group of proteins (i.e. site); or 
 * @li Characterizes functionally non-equivalent points across an unrelated group of proteins (i.e. non-site). 
 * @note featurize produces an Environments group but the output method is not a class method of Environments; consider changing this
 **/
class Environments : public vector< Environment * > {
	public:
		Environments(const char *filename, int numShells);
		~Environments();
		int          numEnvironments();
		Environment *getEnvironment(int index);
		int          getSize();
		Metadata    *getMetadata();
		int          getNumShells();
		int          getNumProperties();
		Doubles     *getValues(int prop_idx, int shell_number);
		void         read(const char *filename, int numShells);
		static void  enableMetadataKeywords( Metadata * );
	protected:
		Metadata* metadata;
};

#endif
