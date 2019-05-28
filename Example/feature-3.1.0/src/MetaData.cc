#include "MetaData.h"

bool Metadata::initialized = false;
map< string, string > Metadata::definedAliases  = map< string, string >();
Metadata::Keywords    Metadata::definedKeywords = Metadata::Keywords();

void Metadata::initialize() {
	const char *defaultPropertiesList =
		"ATOM_TYPE_IS_C, ATOM_TYPE_IS_CT, ATOM_TYPE_IS_CA, ATOM_TYPE_IS_N, "
		"ATOM_TYPE_IS_N2, ATOM_TYPE_IS_N3, ATOM_TYPE_IS_NA, ATOM_TYPE_IS_O, "
		"ATOM_TYPE_IS_O2, ATOM_TYPE_IS_OH, ATOM_TYPE_IS_S, ATOM_TYPE_IS_SH, "
		"ATOM_TYPE_IS_OTHER, PARTIAL_CHARGE, ELEMENT_IS_ANY, ELEMENT_IS_C, "
		"ELEMENT_IS_N, ELEMENT_IS_O, ELEMENT_IS_S, ELEMENT_IS_OTHER, "
		"HYDROXYL, AMIDE, AMINE, CARBONYL, RING_SYSTEM, PEPTIDE, "
		"VDW_VOLUME, CHARGE, NEG_CHARGE, POS_CHARGE, CHARGE_WITH_HIS, "
		"HYDROPHOBICITY, MOBILITY, SOLVENT_ACCESSIBILITY, "
		"RESIDUE_NAME_IS_ALA, RESIDUE_NAME_IS_ARG, RESIDUE_NAME_IS_ASN, "
		"RESIDUE_NAME_IS_ASP, RESIDUE_NAME_IS_CYS, RESIDUE_NAME_IS_GLN, "
		"RESIDUE_NAME_IS_GLU, RESIDUE_NAME_IS_GLY, RESIDUE_NAME_IS_HIS, "
		"RESIDUE_NAME_IS_ILE, RESIDUE_NAME_IS_LEU, RESIDUE_NAME_IS_LYS, "
		"RESIDUE_NAME_IS_MET, RESIDUE_NAME_IS_PHE, RESIDUE_NAME_IS_PRO, "
		"RESIDUE_NAME_IS_SER, RESIDUE_NAME_IS_THR, RESIDUE_NAME_IS_TRP, "
		"RESIDUE_NAME_IS_TYR, RESIDUE_NAME_IS_VAL, RESIDUE_NAME_IS_HOH, "
		"RESIDUE_NAME_IS_OTHER, RESIDUE_CLASS1_IS_HYDROPHOBIC, "
		"RESIDUE_CLASS1_IS_CHARGED, RESIDUE_CLASS1_IS_POLAR, "
		"RESIDUE_CLASS1_IS_UNKNOWN, RESIDUE_CLASS2_IS_NONPOLAR, "
		"RESIDUE_CLASS2_IS_POLAR, RESIDUE_CLASS2_IS_BASIC, "
		"RESIDUE_CLASS2_IS_ACIDIC, RESIDUE_CLASS2_IS_UNKNOWN, "
		"SECONDARY_STRUCTURE1_IS_3HELIX, SECONDARY_STRUCTURE1_IS_4HELIX, "
		"SECONDARY_STRUCTURE1_IS_5HELIX, SECONDARY_STRUCTURE1_IS_BRIDGE, "
		"SECONDARY_STRUCTURE1_IS_STRAND, SECONDARY_STRUCTURE1_IS_TURN, "
		"SECONDARY_STRUCTURE1_IS_BEND, SECONDARY_STRUCTURE1_IS_COIL, "
		"SECONDARY_STRUCTURE1_IS_HET, SECONDARY_STRUCTURE1_IS_UNKNOWN, "
		"SECONDARY_STRUCTURE2_IS_HELIX, SECONDARY_STRUCTURE2_IS_BETA, "
		"SECONDARY_STRUCTURE2_IS_COIL, SECONDARY_STRUCTURE2_IS_HET, "
		"SECONDARY_STRUCTURE2_IS_UNKNOWN";

	string defaultProperties = string( defaultPropertiesList );

	defineAlias(   "NUM_BINS",           "BINS"            );
	defineAlias(   "NUM_SHELLS",         "SHELLS"          );
	defineAlias(   "AFFILIATION",        "INSTITUTION"     );

	defineKeyword( "ALL_SCORES",          Keyword::BOOL,  "Show all scores (ignore cutoff)",                    "0"                  );
	defineKeyword( "BINS",                Keyword::INT,   "Set the number of NB bins",                          "5"                  );
	defineKeyword( "BOUNDS_FILE",         Keyword::VALUE, "Specify the min/max bounds in shell major order"                          );
	defineKeyword( "CUTOFF",              Keyword::VALUE, "Set the minimum score to print out",                 "0.0"                );
	defineKeyword( "EXCLUDED_RESIDUES",   Keyword::LIST,  "Exclude the given comma-separated list of residues", "HETATM"             );
	defineKeyword( "EXTENDED_OUTPUT",     Keyword::BOOL,  "Display additional MWW Rank Sum information",        "0"                  );
	defineKeyword( "FEATURE_VECTOR_FILE", Keyword::VALUE                                                                             );
	defineKeyword( "IGNORE_EMPTY_SHELLS", Keyword::BOOL,  "Ignore empty shells",                                "0"                  );
	defineKeyword( "INSTITUTION",         Keyword::VALUE, "Affiliated research institution"                                          );
	defineKeyword( "MODEL_FILE",          Keyword::VALUE                                                                             );
	defineKeyword( "NUM_PROPERTIES",      Keyword::INT,   "",                                                   "80"                 );
	defineKeyword( "PDBID_LIST",          Keyword::LIST                                                                              );
	defineKeyword( "PDBID_FILE",          Keyword::VALUE                                                                             );
	defineKeyword( "POINT_FILE",          Keyword::VALUE                                                                             );
	defineKeyword( "PROPERTIES",          Keyword::LIST,  "",                                                   defaultProperties    );
	defineKeyword( "PROPERTIES_FILE",     Keyword::VALUE                                                                             );
	defineKeyword( "P_SITE",              Keyword::INT,   "Set the prior probability for site classification",  "1"                  );
	defineKeyword( "P_LEVEL",             Keyword::INT,   "Set the maximum threshold for MWW p-value",          "1"                  );
	defineKeyword( "SEARCH_PATH",         Keyword::VALUE                                                                             );
	defineKeyword( "SHELLS",              Keyword::INT,   "Set the number of micorenvironment shells",          "6"                  );
	defineKeyword( "SHELL_WIDTH",         Keyword::FLOAT, "Set the radius of the microenvironment shells",      "1.25"               );
	defineKeyword( "TRAIN_NEG_FILE",      Keyword::VALUE                                                                             );
	defineKeyword( "TRAIN_POS_FILE",      Keyword::VALUE                                                                             );
	defineKeyword( "VERBOSITY",           Keyword::DEBUG, "Increase the verbosity of information messages",     "0"                  );

	initialized = true;
}

void Metadata::defineAlias( string alias, string keyword ) {
	definedAliases[ alias ] = keyword;
}

void Metadata::defineKeyword( string name, Keyword::Type type, string description, string defaultValue ) {
	definedKeywords[ name ] = new Keyword( name, description, defaultValue, type );
}

/**
 * \brief Retrieves the keyword or throw a fatal error if the keyword isn't enabled
 * \note  Modifies the name argument by replacing aliases with the standard name
 **/
Metadata::Keyword *Metadata::getKeyword( string &name ) {
	standardize( name );
	if( keywords.find( name ) == keywords.end() )
		fatal_error( "Metadata: The metadata keyword '%s' is not enabled for this file type '%s'.\n", name.c_str(), entityName.c_str() );

	return keywords[ name ];
}

string Metadata::asString( string name ) {
	Keyword *keyword = getKeyword( name );
	return keyword->getValue();
}

int Metadata::asInt( string name ) {
	Keyword *keyword = getKeyword( name );
	stringstream buffer( keyword->getValue() );
	int intValue;
	buffer >> intValue;
	return intValue;
}

double Metadata::asDouble( string name ) {
	Keyword *keyword = getKeyword( name );
	stringstream buffer( keyword->getValue() );
	double doubleValue;
	buffer >> doubleValue;
	return doubleValue;
}

Strings *Metadata::asStrings( string name ) {
	Keyword *keyword = getKeyword( name );
	return keyword->asStrings();
}

Metadata::Metadata( string _entityName ) {
	if( ! initialized ) initialize();
	entityName = _entityName;
}

string Metadata::getSupportError() {
	string error;
	vector< string >::iterator i;
	for( i = missing.begin(); i != missing.end(); i++ ) {
		error += "    " + (*i) + "\n";
	}
	return error;
}

/**
 * \brief Takes the given keyword name and copies the keyword value from the
 * given metadata to the current metadata
 **/
void Metadata::copy( string name, Metadata *other ) {
	Keyword *keyword = (*this)[ name ];
	string value = (*other)[ name ]->getValue();
	keyword->setValue( value );
}

bool Metadata::parse( string &line ) {
	stringstream buffer( line );

	// ===== CONFIRM THAT THE LINE IS A METADATA LINE
	string hash;
	buffer >> hash;
	trim( hash );
	if( hash != "#" ) return false;

	// ===== PARSE KEYWORD
	string name;
	buffer >> name;
	Keyword *keyword = getKeyword( name );
	
	// ===== SEE IF THERE'S A GIVEN VALUE FROM THE FILE
	string valueString;
	if( getline( buffer, valueString ) ) {
		trim( valueString );
	}

	// ===== ASSIGN THE METADATA VALUE
	if ( keyword->isList() ) {
		if( keyword->isStarted() ) {
			string currentValue = keyword->getValue();
			keyword->setValue( currentValue + valueString );
		} else {
			keyword->clear();
			keyword->setStarted();
			keyword->setValue( valueString );
		}

	} else  {
		keyword->setValue( valueString );
	}
	keyword->setInputSpecified();
	return true;
}

void Metadata::set( string name, int value ) {
	Keyword *keyword = getKeyword( name );
	stringstream buffer;
	buffer << value;
	keyword->setValue( buffer.str() );
}

void Metadata::set( string name, double value ) {
	Keyword *keyword = getKeyword( name );
	stringstream buffer;
	buffer << value;
	keyword->setValue( buffer.str() );
}

void Metadata::set( string name, string value ) {
	Keyword *keyword = getKeyword( name );
	keyword->setValue( value );
}

void Metadata::set( string name, char *value ) {
	Keyword *keyword = getKeyword( name );
	keyword->setValue( string( value ));
}

void Metadata::checkSupport( string name, Metadata *other ) {
	Keyword *mine     = getKeyword( name );
	Keyword *theirs   = other->getKeyword( name );
	string myValue    = mine->getValue();
	string theirValue = theirs->getValue();
	if( mine->isList() ) {
		Strings *myList    = mine->asStrings();
		Strings *theirList = theirs->asStrings();
		bool matches = true;
		for( unsigned int i = 0; i < myList->size(); i++ ) {
			string a = (*myList)[ i ];
			string b = (*theirList)[ i ];
			matches &= (a == b);
		}
		delete myList;
		delete theirList;
		if( ! matches )
			error( "Metadata: %s metadata for %s (%s) differs from %s (%s).\n", name.c_str(), this->entityName.c_str(), myValue.c_str(), other->entityName.c_str(), theirValue.c_str() );

	} else {
		if( myValue != theirValue )
			error( "Metadata: %s metadata for %s (%s) differs from %s (%s).\n", name.c_str(), this->entityName.c_str(), myValue.c_str(), other->entityName.c_str(), theirValue.c_str() );
	}
}

void Metadata::checkSupport( const string name, int theirs ) {
	if( asInt( name ) != theirs )
		error( "Metadata: %s metadata for %s differs from the user-specified value %d.\n", name.c_str(), this->entityName.c_str(), theirs );
}

void Metadata::checkSupport( const string name, double theirs ) {
	if( asDouble( name ) != theirs )
		error( "Metadata: %s metadata for %s differs from the user-specified value %f.\n", name.c_str(), this->entityName.c_str(), theirs );
}

void Metadata::checkSupport( const string name, string theirs ) {
	if( asString( name ) != theirs )
		error( "Metadata: %s metadata for %s differs from the user-specified value %s.\n", name.c_str(), this->entityName.c_str(), theirs.c_str() );
}

/**
 * @brief Directs the Metadata to use default values
 **/
void Metadata::enable( string name, string defaultValue ) {
	standardize( name );

	Keyword *defaultKeyword = definedKeywords[ name ];
	if( defaultKeyword == NULL ) 
		fatal_error( "Metadata:: The metadata keyword '%s' is not defined.\n", name.c_str() );
	if( defaultValue.empty() ) defaultValue = defaultKeyword->getValue();

	keywords[ name ]        = new Keyword( name, defaultKeyword->getDescription(), defaultValue, defaultKeyword->getType() );
}

Metadata::Keyword *Metadata::operator[]( string name ) {
	return getKeyword( name );;
}

void Metadata::writeList( string name ) {
	Strings *list = asStrings( name );
	if( list->size() == 0 ) return;

	vector< string >::iterator i = list->begin();
	string current, line;
	size_t lineLength = 62;
	while( i != (list->end() - 1)) {
		current = (*i);
		if( line.size() + current.size() > lineLength ) {
			cout << "#\t" << name << "\t" << line << endl;
			line.clear();
		}
		line += current + ", ";
		i++;
	}
	current = (*i);
	if( line.size() + current.size() > lineLength ) {
		cout << "#\t" << name << "\t" << line << endl;
		cout << "#\t" << name << "\t" << current << endl;
	} else {
		line += current;
		cout << "#\t" << name << "\t" << line << endl;
	}
	delete list;
}

void Metadata::write() {
	cout << (*this);
}

ostream &operator<<( ostream &out, Metadata &metadata ) {
	for( Metadata::Keywords::iterator i = metadata.keywords.begin(); i != metadata.keywords.end(); i++ ) {
		string             name    = i->first;
		Metadata::Keyword *keyword = i->second;
		if( keyword->isList() ) {
			metadata.writeList( name );
		} else {
			string value = keyword->getValue();
			if( ! value.empty() )
				out << "#\t" << keyword->getName() << "\t" << value << endl;
		}
	}
	return out;
}

Metadata::Keyword::Keyword( string _name, string _description, string _value, Metadata::Keyword::Type _type ) {
	name          = _name;
	description   = _description;
	value         = _value;
	type          = _type;
	userSpecified = false;
	started       = false;
}

int Metadata::Keyword::getIntValue() {
	stringstream buffer( value );
	int intValue;
	buffer >> intValue;
	return intValue;
}

double Metadata::Keyword::getDoubleValue() {
	stringstream buffer( value );
	double doubleValue;
	buffer >> doubleValue;
	return doubleValue;
}

bool Metadata::hasProperties() {
	string   name( "PROPERTIES" );
	Keyword *propertiesKeyword = getKeyword( name );
	Strings *properties        = propertiesKeyword->asStrings();
	return( properties->size() > 0 );
}

Strings *Metadata::Keyword::asStrings() {
	Strings *list = new Strings();
	stringstream buffer( value );
	while( buffer ) {
		string entry;
		getline( buffer, entry, ',' );
		trim( entry );
		if( entry.size() > 0 ) list->push_back( entry );
	}
	return list;
}

