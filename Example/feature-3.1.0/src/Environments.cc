// $Id: Environments.cc 1609 2013-12-10 08:03:27Z mikewong899 $
// Copyright (c) 2004 Mike Liang. All rights reserved

#include "Environments.h"

#define BUF_SIZE 8192
#define ENVIRONMENT_LIST_FORMAT_VERSION 1

Environment::Environment( int numShells ) {
    this->numShells = numShells;
    int r;
    shells = new Shell *[numShells];
    for (r = 0; r < numShells; r++) {
        shells[r] = new Shell();
    }
}

void Environments::enableMetadataKeywords( Metadata *metadata ) {
	metadata->enable( "EXCLUDED_RESIDUES" );
	metadata->enable( "INSTITUTION" );
	metadata->enable( "PDBID_FILE" );
	metadata->enable( "PDBID_LIST" );
	metadata->enable( "POINT_FILE" );
	metadata->enable( "PROPERTIES" );
	metadata->enable( "PROPERTIES_FILE" );
	metadata->enable( "SEARCH_PATH" );
	metadata->enable( "SHELLS" );
	metadata->enable( "SHELL_WIDTH" );
	metadata->enable( "VERBOSITY" );
}

void Environment::write(FILE *outs) {
    int i, r;
    if( ! label.empty() ) {
        fprintf(outs, "%s\t", label.c_str());
    }
    for (r = 0; r < numShells; r++) {
        for (i = 0; i < PropertyList::size(); i++) {
            double divider = PropertyList::get(i)->getDivider();
            fprintf(outs, "%g\t", shells[r]->at(i) / divider);
        }
    }
    if( ! description.empty() ) {
        fprintf(outs, "#\t%s", description.c_str());
    }
    fprintf(outs, "\n");
}

void Environment::parse( string buffer ) {
	size_t i, j;
	stringstream bufferStream( stringstream::in | stringstream::out );

	// Remove newline characters
	i = buffer.find( '\n' );
	if( i != string::npos ) buffer[ i ] = '\0';

	// Search for comment, which starts after the hash mark
	// i is the end of the data stream
	// j is the start of the comment stream
	i = j = buffer.find( '#' );
	if( i != string::npos ) {
		// Don't include the hash mark
		i--;
		j++;

		// Don't include leading tabs in the comment
		if( buffer[ j ] == '\t' ) j++;
		description = buffer.substr( j );
		bufferStream << buffer.substr( 0, i );
	}  else {
		bufferStream << buffer;
	}

	bufferStream >> label;
	if( ! bufferStream )
		fatal_error( "Environment: Failed parsing label (e.g. Env_1afr_0) from buffer\n  %s\n", buffer.c_str() );

    // Get properties
	for( int k = 0; k < numShells; k++ ) {
		Shell *shell = shells[ k ];
		shell->parse( bufferStream );
		if( ! bufferStream )
			fatal_error( "Environment: Failed parsing a property value (shell %d) for label: '%s' from buffer\n  %s", k, label.c_str(), buffer.c_str() );
	}
}

Environment::~Environment() {
    int r;
    for (r = 0; r < numShells; r++) {
        delete shells[r];
    }
    delete [] shells;
}

Environments::Environments(const char *filename, int numShells) {
    metadata = new Metadata( string( filename ) );
	enableMetadataKeywords( metadata );
	read(filename, numShells);
}

int Environments::numEnvironments() {
    return size();
}

Environment * Environments::getEnvironment(int index) {
    return (*this)[index];
}

void Environments::read(const char *filename, int numShells) {
	igzstream file( filename );
	string buffer;

	// ===== PARSE THE METADATA
	while ( getline( file, buffer ) ) {
		if( ! metadata->parse( buffer ) ) break;
	}
	if( ! PropertyList::loaded() ) {
		PropertyList::load( metadata );
		PropertyList::set( metadata );
	}

	Metadata::Keyword *shellsMetadata = (*metadata)[ "SHELLS" ];
	if( shellsMetadata->isInputSpecified() || shellsMetadata->isUserSpecified() ) {
		numShells = metadata->asInt( "SHELLS" );
	}
	
	// ===== PARSE THE ENVIRONMENT DATA
	do {
		if( buffer.empty() ) continue;

		Environment *environment = new Environment( numShells );
		environment->parse( buffer );
		push_back( environment );

    } while ( getline( file, buffer ) );
}

Environments::~Environments() {
    delete metadata;
    Environments::iterator iter;
    for (iter = begin(); iter != end(); iter++) {
        delete (*iter);
    }
}

int Environments::getSize() {
    return size();
}

Metadata *Environments::getMetadata() {
    return metadata;
}

int Environments::getNumShells() {
    if (size() > 0) {
        return (*this)[0]->getNumShells();
    }
    return 0;
}

int Environments::getNumProperties() {
    if (size() > 0) {
        return PropertyList::size();
    }
    return 0;
}

Doubles *Environments::getValues(int prop_idx, int shell_number) {
    Doubles *values = new Doubles;
    Environments::iterator iter;
    for (iter = begin(); iter != end(); iter++) {
		Environment        *environment = (*iter);
		Environment::Shell *shell       = environment->getShell( shell_number );

        values->push_back( shell->getValue( prop_idx ));
    }
    return values;
}


// Creates an array of integers which will store the properties
// of each shell in the scanning grid.
// PROPERTIES_ARRAY_SIZE is a constant in PropertyConstants.h

Environment::Shell::Shell() {
	hasAtLeastOneAtom = false;
	int size = (int) PropertyList::size();
    property = new int[ size ];
    for( int i = 0; i < size; i++ ) {
        property[i] = 0;
	}
}

Environment::Shell::~Shell() {
    delete []property;
}

void Environment::Shell::add(int valueToAdd, string name ) {
	if( ! PropertyList::has( name )) return;
	Property      *p = PropertyList::get( name );
	int            i = p->getIndex();
    property[ i ]   += valueToAdd;
}

void Environment::Shell::add( AtomProperties *atomProperty, string name ) {
	if( ! PropertyList::has( name )) return;
	if( atomProperty->propertyValue[ name ] != 0 ) {
		add( atomProperty->propertyValue[ name ], name );
	}
}

/**
 * @brief Adds the properties that are defined as belonging to a group of
 * mutually exclusive options. 
 *
 * @details Groups of properties are collections of mutually exclusive options.
 * For example, an atom cannot simultaneously be both of type 'C' and 'N'.
 * This method iterates over each item of the group and exits the loop once a
 * it finds the one relevant option. If a default option is provided and no
 * other options match, then the default value is used.
 *
 * Note that the options must have non-negative values; currently all options
 * are category counts (this [atom|residue] is a \em xyz).
 *
 * If any of the properties in the group do not exist in the PropertyList, then
 * the property is silently ignored. 
 **/
void Environment::Shell::addGroup( AtomProperties *atomProperty, PropertyGroup group ) {
	PropertyGroup::iterator i;
	string counterName  = group.getCounterName();
	int    counterStep  = group.getCounterStep();
	string defaultName  = group.getDefaultName();
	int    defaultValue = group.getDefaultValue();

	// ===== ALWAYS INCREMENT THE COUNTER (IF THERE IS A COUNTER)
	if( ! counterName.empty() && PropertyList::has( counterName )) {
		add( counterStep, counterName );
	}

	// ===== ITERATE OVER MUTUALLY EXCLUSIVE OPTIONS
	for( i = group.begin(); i != group.end(); i++ ) {
		int propertyValue = atomProperty->propertyValue[ (*i) ];
		if( propertyValue > 0 && PropertyList::has( (*i) )) {
			add( propertyValue, (*i) );
			return;
		}
	}

	// ===== APPLY DEFAULT VALUE
	if( ! defaultName.empty() && PropertyList::has( defaultName )) {
		add( defaultValue, defaultName );
	}
}

/**
 * @brief Retrieves the scaled property value for the given property name
 * @return the scaled integer value for the given property name
 **/
void Environment::Shell::get(int *propValue, const int i) {
    *propValue = property[ i ];
}

/** 
 * @brief Retrieves the property value for the given property name
 * @return the scaled integer value of desired property
 **/
int Environment::Shell::get( string propertyName) {
	Property *p = PropertyList::get( propertyName );
	int       i = p->getIndex();
    return property[ i ];
}

/** 
 * @brief Retrieves the scaled property value for the given property index i
 * @return the scaled integer value of desired property
 **/
int Environment::Shell::get(const int i) {
    int value;
    get(&value, i);
    return value;
}

/**
 * @brief Retrieves the property value for the given property index i
 * @return the real value of the desired property
 **/
double Environment::Shell::getValue( const int i ) {
	int value;
	get( &value, i );
	Property *property  = PropertyList::get( i );
	double    divider   = property->getDivider();
	double    realValue = ((double) value)/divider;

	return realValue;
}

/**
 * @brief Updates the property array specified as the this pointer.
 * atomProperty contains the values of the current atom's properties
 * which need to be added to the current property array.
 **/
void Environment::Shell::include( AtomProperties *atomProperty ) {
	hasAtLeastOneAtom = true;
	add( atomProperty, "PARTIAL_CHARGE" );
	add( atomProperty, "HYDROXYL" );
	add( atomProperty, "AMIDE" );
	add( atomProperty, "AMINE" );
	add( atomProperty, "CARBONYL" );
	add( atomProperty, "RING_SYSTEM" );
	add( atomProperty, "PEPTIDE" );
	add( atomProperty, "VDW_VOLUME" );
	add( atomProperty, "CHARGE" );
	add( atomProperty, "NEG_CHARGE" );
	add( atomProperty, "POS_CHARGE" );
	add( atomProperty, "CHARGE_WITH_HIS" );
	add( atomProperty, "HYDROPHOBICITY" );
	add( atomProperty, "MOBILITY" );
	add( atomProperty, "SOLVENT_ACCESSIBILITY" );

	unordered_map< string, PropertyGroup >::iterator group;
	for( group = PropertyList::propertyGroup.begin(); group != PropertyList::propertyGroup.end(); group++) {
		addGroup( atomProperty, group->second );
	}
}

/**
 * @brief Parses the buffer stream
 **/
void Environment::Shell::parse( stringstream &bufferStream ) {
	for (int i = 0; i < PropertyList::size(); i++) {
		double value;
		bufferStream >> value;
		if( ! bufferStream )
			fatal_error( "Environment::Shell: Stopped parsing at value: %g. Missing properties: wanted %d, got %d.\n\n%s\n\n", value, PropertyList::size(), i, bufferStream.str().c_str() );
		double divider = PropertyList::get(i)->getDivider();
		property[ i ]  = (int) round( (value * divider));
		if( value ) hasAtLeastOneAtom = true;
	}
}

/**
 * @brief Prints the shell to the given filehandle
 **/
void Environment::Shell::write( FILE *outs ) {
	for( int i = 0; i < PropertyList::size(); i++ ) {
		Property *property = PropertyList::get( i );
        int    value       = get( i );
		double divider     = property->getDivider();
		if( divider == 1.0 ) {
			fprintf(outs, "\t%d", value);
		} else {
			fprintf(outs, "\t%g", ((double) value) / divider );
		}
	}
}

/**
 * @brief Static method to print out the property values of an environment to a file
 *
 **/
void dumpEnvironment(FILE *ofile, Environment::Shell **environment, int numShells) {
    for (int r = 0; r < numShells; r++) {
        Environment::Shell *shell = environment[ r ];
        for (int property = 0; property < PropertyList::size(); property++) {
            if (environment && shell) {
                fprintf(ofile, "\t%d", shell->get(property));
            } else {
                fprintf(ofile, "\t%d", 0);
            }
        }
    }
}
