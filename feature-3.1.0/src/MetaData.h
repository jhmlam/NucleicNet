// FEATURE
// See COPYING and AUTHORS for copyright and license agreement terms.

// $Id: PropertyList.h 994 2011-08-06 20:29:39Z teague $

#ifndef METADATA_H
#define METADATA_H

#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <set>
#include "Utilities.h"
#include "logutils.h"
#include "stl_types.h"

using std::cout;
using std::endl;
using std::map;
using std::string;
using std::vector;
using std::istream;
using std::ostream;
using std::stringstream;
// @note Since there's a Metadata::set() method, we use std::set to distinguish the two

class UsageMessage {
	virtual void show() = 0;
};

/**
 * @brief Parses and stores file metadata between the various FEATURE file
 * formats, and confirms compatibility between files
 *
 * The purpose of the Metadata class is to maintain data integrity between the
 * different FEATURE file formats. Because many FEATURE files are simply
 * columns of data, it's simple to confuse or transpose columns during an
 * experiment. Therefore it's all too easy to get seemingly meaningful results
 * from mixed-up data.
 *
 * The Metadata class performs two distinct functions. First it provides a
 * globally-accessible list of supported keywords. Second it allows instances
 * of the Metadata class, one per file being parsed. 
 *
 **/
class Metadata {
	public:
			class Keyword {
			public:
				enum    Type { INT, FLOAT, VALUE, LIST, BOOL, DEBUG };

				Keyword( string, string = "", string = "", Type = VALUE );

				Strings *asStrings();
				void      clear()                   { value.clear(); userSpecified = false; started = false; }
				string    getDescription()          { return description; }
				double    getDoubleValue();
				int       getIntValue();
				string    getName()                 { return name;           }
				string   &getValue()                { return value;          }
				Type      getType()                 { return type;           }
				bool      isBool()                  { return type == BOOL;   }
				bool      isDebug()                 { return type == DEBUG;  }
				bool      isFloat()                 { return type == FLOAT;  }
				bool      isInt()                   { return type == INT;    }
				bool      isList()                  { return type == LIST;   }
				bool      isValue()                 { return type == VALUE;  }
				bool      isStarted()               { return started;        }
				bool      isUserSpecified()         { return userSpecified;  }
				bool      isInputSpecified()        { return inputSpecified; }
				void      setStarted()              { started = true;        }
				void      setUserSpecified()        { userSpecified = true;  }
				void      setInputSpecified()       { inputSpecified = true; }
				void      setValue( string _value ) { value = _value;        }

			protected:
				Type      type;
				string    name;
				string	  value;
				string    description;
				bool      userSpecified;  // True if the user provides a parameter, false otherwise (use defaults)
				bool      inputSpecified; // True if the input file metadata provides a parameter, false otherwise (use defaults)
				bool      started;        // True if this keyword is type LIST and has parsed some data
		};

		Metadata( string );

		bool              hasProperties();
		bool              parse( string & );
		void              copy( string, Metadata * );
		void              enable( string name, string defaultValue = "" );
		void              set( string, int );
		void              set( string, double );
		void              set( string, string );
		void              set( string, char * );
		void              setName( string _entityName ) { entityName = _entityName; };
		string            asString( string name ); 
		int               asInt( string name ); 
		double            asDouble( string name ); 
		Strings          *asStrings( string name ); 
		string            getSupportError();
		void              checkSupport( string, Metadata * );
		void              checkSupport( string, int );
		void              checkSupport( string, double );
		void              checkSupport( string, string );
		void              write();
		Keyword          *operator[]( string );
		friend ostream   &operator<<( ostream &, Metadata & );

		typedef map< string, Keyword * > Keywords;

	protected:
		string                        entityName;
		Keywords                      keywords;
		vector< string >              missing;
		bool                          seenProperties;
		string                        options;

		Keyword                      *getKeyword( string &name );
		string                        parseList( string & );
		void                          writeList( string );

		static bool                   initialized;
		static map< string, string >  definedAliases;
		static Keywords               definedKeywords;

		static void                   initialize();
		static void                   defineAlias(   string alias, string name );
		static void                   defineKeyword( string name, Keyword::Type type = Keyword::VALUE, string description = "", string defaultValue = "" );
		static inline void            standardize( string &name ) { if( definedAliases.find( name ) != definedAliases.end() ) name = definedAliases[ name ]; if( definedKeywords.find( name ) == definedKeywords.end() ) fatal_error( "Enabled metadata keyword '%s' not supported.\n", name.c_str() ); };
};

#endif
