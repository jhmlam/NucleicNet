#include "PropertyList.h"

/** 
 * @note The static variables and initializations are getting out of hand;
 * when there is time, we should make this a proper Singleton. 
 **/
unordered_map< string, Property * >    PropertyList::definedProperties;
unordered_map< string, PropertyGroup > PropertyList::propertyGroup;
unordered_map< string, unsigned int >  PropertyList::indexLookup;
vector< Property * >                   PropertyList::list;
bool PropertyList::isLoaded    = false;
bool PropertyList::initialized = false;

Property::Property( string _name, int _index, Source _source, double _divider ) {
    name    = _name;
    index   = _index;
	source  = _source;
    divider = _divider;
}

string& Property::getName () {
    return name;
}

const int Property::getIndex() const {
    return index;
}

const double Property::getDivider() const {
    return divider;
}

const Property::Source Property::getSource() const {
	return source;
}

PropertyGroup::PropertyGroup() {
	defaultName.clear();
	defaultValue = 0;
	counterName.clear();
	counterStep  = 0;
}

/**
 * @brief Associates the given property name with the group.
 * @param name A property name
 * @note This method is used to partially define a property group.
 **/
void PropertyGroup::include( string name ) {
	if( ! PropertyList::isDefined( name )) {
		error( "Cannot include property '%s'; property is not defined.\n", name.c_str());
		return;
	}

	push_back( name );
}

/**
 * @brief Associates the given property name as the counter of the group.
 * @param name A property name
 * @param value Counter step value
 * @note This method is used to partially define a property group.
 **/
void PropertyGroup::setCounter( string name, int value ) {
	if( ! PropertyList::isDefined( name )) {
		error( "Cannot set property '%s' as group counter; property is not defined.\n", name.c_str());
		return;
	}

	counterName = name;
	counterStep = value;
}

/**
 * @brief Associates the given property name as the default of the group.
 * @param name A property name
 * @param value Default step value
 * @note This method is used to partially define a property group.
 **/
void PropertyGroup::setDefault( string name, int value ) {
	if( ! PropertyList::isDefined( name )) {
		error( "Cannot set property '%s' as group default; property is not defined.\n", name.c_str());
		return;
	}

	defaultName  = name;
	defaultValue = value;
}

PropertyList::PropertyList() {}

void PropertyList::enable( const char * name ) {
    enable( string( name ));
}


/**
 * @brief Adds the property to the PropertyList to use for analysis. Only
 * defined properties can be enabled. Defined properties are explicitly named
 * using the defineProperty() method. This mechanism prevents user errors (e.g.
 * typos).
 **/
void PropertyList::enable( string name ) {
    int i       = list.size();
    debug( 2, "Enabling property %d: %s\n", i, name.c_str());

    unordered_map<string, Property *>::iterator it = definedProperties.find( name );
    if( it == definedProperties.end() ) {
        fatal_error( "PropertyList: Undefined property: %s\n", name.c_str());
    }
    Property *property = it->second;

    indexLookup[ name ] = i;
	property->setIndex( i );
    list.push_back( property );
}

vector<Property *>::iterator PropertyList::begin() {
	return list.begin();
}

void PropertyList::defineProperty(const char * name, Property::Source source, double divider) {
    defineProperty(string(name), source, divider);
}

/**
 * @brief Defines the property as acceptable for FEATURE analysis. By default,
 * defined properties are disabled. To enable the property to use in an
 * analysis, it must be defined and enabled.
 **/
void PropertyList::defineProperty(string name, Property::Source source, double divider) {
    debug(2, "Defining divider for %s as %f\n", name.c_str(), divider);
	if( definedProperties.find( name ) != definedProperties.end() ) {
		error( "Cannot redefine property '%s'\n", name.c_str() );
	}
	definedProperties[ name ] = new Property( name, -1, source, divider );
}

vector<Property *>::iterator PropertyList::end() {
	return list.end();
}

Property *PropertyList::get(const int index) {
    return list[ index ];
}

Property *PropertyList::get(const char *name) {
    return get( string( name ));
}

/**
 * @brief Returns the property object
 * @note This function assumes that the property is defined and exists in the
 * list of properties; one should always check by using the PropertyList::has()
 * function to verify that the property is defined and exists in the list of
 * properties.
 **/
Property *PropertyList::get(string name) {
	if( indexLookup.find( name ) == indexLookup.end() )
		fatal_error( "PropertyList::get(): Property %s is not yet defined.\n", name.c_str() );
    int index = indexLookup[ name ];
    return list[ index ];
}

/**
 * @brief Returns the divider (i.e. significant figure precision) for the property.
 * @note This function assumes that the property is defined and exists in the
 * list of properties; one should always check by using the PropertyList::has()
 * function to verify that the property is defined and exists in the list of
 * properties.
 **/
double PropertyList::getDivider( string name ) {
	if( indexLookup.find( name ) == indexLookup.end() )
		fatal_error( "PropertyList::getDivider(): Property %s is not yet defined.\n", name.c_str() );
	int index = indexLookup[ name ];
	return list[ index ]->getDivider();
}

bool PropertyList::has(const char *name) {
    return has(string(name));
}

bool PropertyList::has(string name) {
	return ( indexLookup.find( name ) != indexLookup.end() );
}

bool PropertyList::isDefined( string name  ) {
	return ( definedProperties.find( name ) != definedProperties.end() );
}

int PropertyList::size() {
    return list.size();
}

/**
 * @brief Defines all supported properties; users will typically want a subset of this list
 **/
void PropertyList::initialize() {

	if (initialized == true) return;

	defineProperty( "ATOM_TYPE_IS_C",                  Property::FORCE_FIELD,      1.0 );
	defineProperty( "ATOM_TYPE_IS_CT",                 Property::FORCE_FIELD,      1.0 );
	defineProperty( "ATOM_TYPE_IS_CA",                 Property::FORCE_FIELD,      1.0 );
	defineProperty( "ATOM_TYPE_IS_N",                  Property::FORCE_FIELD,      1.0 );
	defineProperty( "ATOM_TYPE_IS_N2",                 Property::FORCE_FIELD,      1.0 );
	defineProperty( "ATOM_TYPE_IS_N3",                 Property::FORCE_FIELD,      1.0 );
	defineProperty( "ATOM_TYPE_IS_NA",                 Property::FORCE_FIELD,      1.0 );
	defineProperty( "ATOM_TYPE_IS_O",                  Property::FORCE_FIELD,      1.0 );
	defineProperty( "ATOM_TYPE_IS_O2",                 Property::FORCE_FIELD,      1.0 );
	defineProperty( "ATOM_TYPE_IS_OH",                 Property::FORCE_FIELD,      1.0 );
	defineProperty( "ATOM_TYPE_IS_S",                  Property::FORCE_FIELD,      1.0 );
	defineProperty( "ATOM_TYPE_IS_SH",                 Property::FORCE_FIELD,      1.0 );
	defineProperty( "ATOM_TYPE_IS_OTHER",              Property::FORCE_FIELD,      1.0 );
	defineProperty( "PARTIAL_CHARGE",                  Property::FORCE_FIELD,  10000.0 );
	defineProperty( "ELEMENT_IS_ANY",                  Property::PDB,              1   );
	defineProperty( "ELEMENT_IS_C",                    Property::PDB,              1   );
	defineProperty( "ELEMENT_IS_N",                    Property::PDB,              1   );
	defineProperty( "ELEMENT_IS_O",                    Property::PDB,              1   );
	defineProperty( "ELEMENT_IS_S",                    Property::PDB,              1   );
	defineProperty( "ELEMENT_IS_OTHER",                Property::PDB,              1   );
	defineProperty( "HYDROXYL",                        Property::PDB,             10.0 );
	defineProperty( "AMIDE",                           Property::PDB,            100.0 );
	defineProperty( "AMINE",                           Property::PDB,            100.0 );
	defineProperty( "CARBONYL",                        Property::PDB,             10.0 );
	defineProperty( "RING_SYSTEM",                     Property::PDB,              1   );
	defineProperty( "PEPTIDE",                         Property::PDB,              1   );
	defineProperty( "VDW_VOLUME",                      Property::PDB,           1000.0 );
	defineProperty( "CHARGE",                          Property::PDB,           1000.0 );
	defineProperty( "NEG_CHARGE",                      Property::PDB,           1000.0 );
	defineProperty( "POS_CHARGE",                      Property::PDB,           1000.0 );
	defineProperty( "CHARGE_WITH_HIS",                 Property::PDB,           1000.0 );
	defineProperty( "HYDROPHOBICITY",                  Property::PDB,           1000.0 );
	defineProperty( "MOBILITY",                        Property::PDB,              1   );
	defineProperty( "SOLVENT_ACCESSIBILITY",           Property::DSSP,             1   );
	defineProperty( "RESIDUE_NAME_IS_ALA",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_ARG",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_ASN",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_ASP",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_CYS",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_GLN",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_GLU",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_GLY",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_HIS",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_ILE",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_LEU",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_LYS",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_MET",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_PHE",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_PRO",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_SER",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_THR",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_TRP",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_TYR",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_VAL",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_HOH",             Property::PDB,              1   );
	defineProperty( "RESIDUE_NAME_IS_OTHER",           Property::PDB,              1   );
	defineProperty( "RESIDUE_CLASS1_IS_HYDROPHOBIC",   Property::PDB,              1   );
	defineProperty( "RESIDUE_CLASS1_IS_CHARGED",       Property::PDB,              1   );
	defineProperty( "RESIDUE_CLASS1_IS_POLAR",         Property::PDB,              1   );
	defineProperty( "RESIDUE_CLASS1_IS_UNKNOWN",       Property::PDB,              1   );
	defineProperty( "RESIDUE_CLASS2_IS_NONPOLAR",      Property::PDB,              1   );
	defineProperty( "RESIDUE_CLASS2_IS_POLAR",         Property::PDB,              1   );
	defineProperty( "RESIDUE_CLASS2_IS_BASIC",         Property::PDB,              1   );
	defineProperty( "RESIDUE_CLASS2_IS_ACIDIC",        Property::PDB,              1   );
	defineProperty( "RESIDUE_CLASS2_IS_UNKNOWN",       Property::PDB,              1   );
	defineProperty( "SECONDARY_STRUCTURE1_IS_3HELIX",  Property::DSSP,             1   );
	defineProperty( "SECONDARY_STRUCTURE1_IS_4HELIX",  Property::DSSP,             1   );
	defineProperty( "SECONDARY_STRUCTURE1_IS_5HELIX",  Property::DSSP,             1   );
	defineProperty( "SECONDARY_STRUCTURE1_IS_BRIDGE",  Property::DSSP,             1   );
	defineProperty( "SECONDARY_STRUCTURE1_IS_STRAND",  Property::DSSP,             1   );
	defineProperty( "SECONDARY_STRUCTURE1_IS_TURN",    Property::DSSP,             1   );
	defineProperty( "SECONDARY_STRUCTURE1_IS_BEND",    Property::DSSP,             1   );
	defineProperty( "SECONDARY_STRUCTURE1_IS_COIL",    Property::DSSP,             1   );
	defineProperty( "SECONDARY_STRUCTURE1_IS_HET",     Property::DSSP,             1   );
	defineProperty( "SECONDARY_STRUCTURE1_IS_UNKNOWN", Property::DSSP,             1   );
	defineProperty( "SECONDARY_STRUCTURE2_IS_HELIX",   Property::DSSP,             1   );
	defineProperty( "SECONDARY_STRUCTURE2_IS_BETA",    Property::DSSP,             1   );
	defineProperty( "SECONDARY_STRUCTURE2_IS_COIL",    Property::DSSP,             1   );
	defineProperty( "SECONDARY_STRUCTURE2_IS_HET",     Property::DSSP,             1   );
	defineProperty( "SECONDARY_STRUCTURE2_IS_UNKNOWN", Property::DSSP,             1   );
	defineProperty( "ALIPHATIC_CARBON",                Property::PDB,              1.0 );
	defineProperty( "AROMATIC_CARBON",                 Property::PDB,              1.0 );
	defineProperty( "PARTIAL_POSITIVE_CARBON",         Property::PDB,              1.0 );
	defineProperty( "ALIPHATIC_CARBON_NEXT_TO_POLAR",  Property::PDB,              1.0 );
	defineProperty( "AMIDE_CARBON",                    Property::PDB,              1.0 );
	defineProperty( "CARBOXYL_CARBON",                 Property::PDB,              1.0 );
	defineProperty( "AMIDE_NITROGEN",                  Property::PDB,              1.0 );
	defineProperty( "POSITIVE_NITROGEN",               Property::PDB,              1.0 );
	defineProperty( "AROMATIC_NITROGEN",               Property::PDB,              1.0 );
	defineProperty( "AMIDE_OXYGEN",                    Property::PDB,              1.0 );
	defineProperty( "CARBOXYL_OXYGEN",                 Property::PDB,              1.0 );
    defineProperty( "HYDROXYL_OXYGEN",                 Property::PDB,              1.0 );
	defineProperty( "SULFUR",                          Property::PDB,              1.0 );

	// ===== PROPERTY GROUPS WITH MUTUALLY EXCLUSIVE OPTIONS
	propertyGroup[ "ATOM_TYPE" ] = PropertyGroup();
	propertyGroup[ "ATOM_TYPE" ].include(    "ATOM_TYPE_IS_C" );
	propertyGroup[ "ATOM_TYPE" ].include(    "ATOM_TYPE_IS_CT" );
	propertyGroup[ "ATOM_TYPE" ].include(    "ATOM_TYPE_IS_CA" );
	propertyGroup[ "ATOM_TYPE" ].include(    "ATOM_TYPE_IS_N" );
	propertyGroup[ "ATOM_TYPE" ].include(    "ATOM_TYPE_IS_N2" );
	propertyGroup[ "ATOM_TYPE" ].include(    "ATOM_TYPE_IS_N3" );
	propertyGroup[ "ATOM_TYPE" ].include(    "ATOM_TYPE_IS_NA" );
	propertyGroup[ "ATOM_TYPE" ].include(    "ATOM_TYPE_IS_O" );
	propertyGroup[ "ATOM_TYPE" ].include(    "ATOM_TYPE_IS_O2" );
	propertyGroup[ "ATOM_TYPE" ].include(    "ATOM_TYPE_IS_OH" );
	propertyGroup[ "ATOM_TYPE" ].include(    "ATOM_TYPE_IS_S" );
	propertyGroup[ "ATOM_TYPE" ].include(    "ATOM_TYPE_IS_SH" );
	propertyGroup[ "ATOM_TYPE" ].setDefault( "ATOM_TYPE_IS_OTHER" );

	propertyGroup[ "ELEMENT" ] = PropertyGroup();
	propertyGroup[ "ELEMENT" ].include(    "ELEMENT_IS_C" );
	propertyGroup[ "ELEMENT" ].include(    "ELEMENT_IS_N" );
	propertyGroup[ "ELEMENT" ].include(    "ELEMENT_IS_O" );
	propertyGroup[ "ELEMENT" ].include(    "ELEMENT_IS_S" );
	propertyGroup[ "ELEMENT" ].include(    "ELEMENT_IS_OTHER" );
	propertyGroup[ "ELEMENT" ].setCounter( "ELEMENT_IS_ANY" );

	propertyGroup[ "RESIDUE_NAME" ] = PropertyGroup();
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_ALA" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_ARG" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_ASN" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_ASP" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_CYS" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_GLN" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_GLU" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_GLY" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_HIS" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_ILE" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_LEU" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_LYS" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_MET" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_PHE" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_PRO" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_SER" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_THR" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_TRP" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_TYR" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_VAL" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_HOH" );
	propertyGroup[ "RESIDUE_NAME" ].include( "RESIDUE_NAME_IS_OTHER" );

	propertyGroup[ "RESIDUE_CLASS1" ] = PropertyGroup();
	propertyGroup[ "RESIDUE_CLASS1" ].include( "RESIDUE_CLASS1_IS_HYDROPHOBIC" );
	propertyGroup[ "RESIDUE_CLASS1" ].include( "RESIDUE_CLASS1_IS_CHARGED" );
	propertyGroup[ "RESIDUE_CLASS1" ].include( "RESIDUE_CLASS1_IS_POLAR" );
	propertyGroup[ "RESIDUE_CLASS1" ].include( "RESIDUE_CLASS1_IS_UNKNOWN" );

	propertyGroup[ "RESIDUE_CLASS2" ] = PropertyGroup();
	propertyGroup[ "RESIDUE_CLASS2" ].include( "RESIDUE_CLASS2_IS_NONPOLAR" );
	propertyGroup[ "RESIDUE_CLASS2" ].include( "RESIDUE_CLASS2_IS_POLAR" );
	propertyGroup[ "RESIDUE_CLASS2" ].include( "RESIDUE_CLASS2_IS_BASIC" );
	propertyGroup[ "RESIDUE_CLASS2" ].include( "RESIDUE_CLASS2_IS_ACIDIC" );
	propertyGroup[ "RESIDUE_CLASS2" ].include( "RESIDUE_CLASS2_IS_UNKNOWN" );

	propertyGroup[ "SECONDARY_STRUCTURE1" ] = PropertyGroup();
	propertyGroup[ "SECONDARY_STRUCTURE1" ].include( "SECONDARY_STRUCTURE1_IS_3HELIX" );
	propertyGroup[ "SECONDARY_STRUCTURE1" ].include( "SECONDARY_STRUCTURE1_IS_4HELIX" );
	propertyGroup[ "SECONDARY_STRUCTURE1" ].include( "SECONDARY_STRUCTURE1_IS_5HELIX" );
	propertyGroup[ "SECONDARY_STRUCTURE1" ].include( "SECONDARY_STRUCTURE1_IS_BRIDGE" );
	propertyGroup[ "SECONDARY_STRUCTURE1" ].include( "SECONDARY_STRUCTURE1_IS_STRAND" );
	propertyGroup[ "SECONDARY_STRUCTURE1" ].include( "SECONDARY_STRUCTURE1_IS_TURN" );
	propertyGroup[ "SECONDARY_STRUCTURE1" ].include( "SECONDARY_STRUCTURE1_IS_BEND" );
	propertyGroup[ "SECONDARY_STRUCTURE1" ].include( "SECONDARY_STRUCTURE1_IS_COIL" );
	propertyGroup[ "SECONDARY_STRUCTURE1" ].include( "SECONDARY_STRUCTURE1_IS_HET" );
	propertyGroup[ "SECONDARY_STRUCTURE1" ].include( "SECONDARY_STRUCTURE1_IS_UNKNOWN" );

	propertyGroup[ "SECONDARY_STRUCTURE2" ] = PropertyGroup();
	propertyGroup[ "SECONDARY_STRUCTURE2" ].include( "SECONDARY_STRUCTURE2_IS_HELIX" );
	propertyGroup[ "SECONDARY_STRUCTURE2" ].include( "SECONDARY_STRUCTURE2_IS_BETA" );
	propertyGroup[ "SECONDARY_STRUCTURE2" ].include( "SECONDARY_STRUCTURE2_IS_COIL" );
	propertyGroup[ "SECONDARY_STRUCTURE2" ].include( "SECONDARY_STRUCTURE2_IS_HET" );
	propertyGroup[ "SECONDARY_STRUCTURE2" ].include( "SECONDARY_STRUCTURE2_IS_UNKNOWN" );

	propertyGroup[ "FUNCTIONAL_GROUP" ] = PropertyGroup();
	propertyGroup[ "FUNCTIONAL_GROUP" ].include( "ALIPHATIC_CARBON" );
	propertyGroup[ "FUNCTIONAL_GROUP" ].include( "AROMATIC_CARBON" );
	propertyGroup[ "FUNCTIONAL_GROUP" ].include( "PARTIAL_POSITIVE_CARBON" );
	propertyGroup[ "FUNCTIONAL_GROUP" ].include( "ALIPHATIC_CARBON_NEXT_TO_POLAR" );
	propertyGroup[ "FUNCTIONAL_GROUP" ].include( "AMIDE_CARBON" );
	propertyGroup[ "FUNCTIONAL_GROUP" ].include( "CARBOXYL_CARBON" );
	propertyGroup[ "FUNCTIONAL_GROUP" ].include( "AMIDE_NITROGEN" );
	propertyGroup[ "FUNCTIONAL_GROUP" ].include( "POSITIVE_NITROGEN" );
	propertyGroup[ "FUNCTIONAL_GROUP" ].include( "AROMATIC_NITROGEN" );
	propertyGroup[ "FUNCTIONAL_GROUP" ].include( "AMIDE_OXYGEN" );
	propertyGroup[ "FUNCTIONAL_GROUP" ].include( "CARBOXYL_OXYGEN" );
	propertyGroup[ "FUNCTIONAL_GROUP" ].include( "HYDROXYL_OXYGEN" );
	propertyGroup[ "FUNCTIONAL_GROUP" ].include( "SULFUR" );

    initialized = true;
}

vector<Property *>& PropertyList::getList() {
    return list;
}

void PropertyList::load() {
    initialize();
    debug( 0, "No properties file provided. Attempting to load from metadata.\n" );
}

void PropertyList::load( const string &file ) {
    initialize();
	if( isLoaded ) return;
    ifstream propertiesFile;
    
    debug(1, "Opening properties file: %s\n", file.c_str() );
    propertiesFile.open( file.c_str() , ifstream::in );
    
    // If not in current directory, try $FEATURE_DIR
    if( ! propertiesFile.is_open() ) {
        string  path = getEnvironmentVariable( FeatureDirEnvStr );
        string fullPath = joinPath( path, file );
        debug( 1, "Checking for properties file at: %s\n", fullPath.c_str() );
        propertiesFile.open( fullPath.c_str(), ifstream::in );
    }

    if( ! propertiesFile.is_open()) {
        fatal_error( "PropertyList: Unable to open properties file '%s'.\n", file.c_str());
    }
    
    string buffer, property;
    while( getline( propertiesFile, buffer )) {
        if( buffer.empty() ) continue;
    
		stringstream line( buffer );
        if( line.peek() == '#' ) continue; // Skip comments

		line >> property;
		if( PropertyList::isDefined( property ) ) {
			enable( property );
		} else {
			error( "PropertyList: Undefined property: '%s'\n", property.c_str() );
		}
    }
    
    isLoaded = true;
}

/**
 * @brief Loads the list of properties from the file metadata
 *
 * This is especially intended for Feature Vector files, but this could also
 * apply to Model files. Model files already have the properties listed in the
 * first column (as well as the shell index).
 **/
void PropertyList::load( Metadata *metadata ) {
    initialize();
	if( isLoaded ) return;
	Metadata::Keyword *propertiesMetadata = (*metadata)[ "PROPERTIES" ];
	stringstream propertiesList( propertiesMetadata->getValue() );

	string buffer, property;
    while( getline( propertiesList, buffer, ',' )) {
        if( buffer.empty() ) continue;
    
        // Read a line into a stream for parsing
		stringstream line( buffer );
		line >> property;
		size_t lead = property.find_first_not_of( " \t" );
		if( lead != string::npos ) property = property.substr( lead );          // Trim leading space
		size_t tail = property.find_last_not_of( " \t" );
		if( tail != string::npos ) property = property.substr( 0, (tail + 1) ); // Trim trailing space

		if( PropertyList::isDefined( property ) ) {
			enable( property );
		} else {
			error( "Undefined property: '%s'\n", property.c_str() );
		}
    }
    
    isLoaded = true;
}

bool PropertyList::loaded() {
    return isLoaded;
}

void PropertyList::set( Metadata *metadata ) {
	vector<Property *>::iterator i;

	string value;
	Metadata::Keyword *propertiesMetadata = (*metadata)[ "PROPERTIES" ];
	for( i = PropertyList::begin(); i != PropertyList::end(); i++ ) {
		value += (*i)->getName() + ", ";
	}
	value = value.substr( 0, (value.size() - 2)); // Trim trailing ", "
	propertiesMetadata->clear();
	propertiesMetadata->setValue( value );
}
