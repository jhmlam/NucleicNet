// FEATURE
// See COPYING and AUTHORS for copyright and license agreement terms.

// $Id: AtomProperties.cc 1609 2013-12-10 08:03:27Z mikewong899 $

//AtomProperties.cpp
//Most of how to calculate the property values are simulated off of PrebuiltProperties.cpp
#include "AtomProperties.h"

bool AtomProperties::LookupTables::isInitialized = false;
AtomProperties::LookupTables *AtomProperties::LookupTables::singleton = NULL;

/**
 * @brief Takes PDB and DSSP information from the Atom class and populates
 * property values either directly from the PDB/DSSP data or from lookup
 * tables based on biochemical knowledge.
 * 
 * Since many fields in Atom are doubles, we must convert to integers by
 * multiplying the number of significant digits for that property. This is
 * reversed when we use the values of the Properties to determine score.
 **/
AtomProperties::AtomProperties( Atom *_atom) {
	atom         = _atom;
	residueName  = atom->getResidueName();
	fullName     = atom->getFullName();
	name         = atom->getName();
	lookupTables = LookupTables::getInstance();

    coord = Point3D( *(atom->getCoordinates()));
    propertyValue[ "ELEMENT_IS_ANY" ] = 1; // always 1 for any atom
    setName();
	setFunctionalGroup();
	setHydroxyl();
	setAmide();
	setAmine();
	setCarbonyl();
    setRingSystem();
    setPeptide();
    setType();
    setPartialCharge();
    setVDWVolume();
	setCharge();
    setPosCharge();
    setNegCharge();
    setChargeWithHis();
	setHydrophobicity();
	setMobility();
	setSolventAccessibility();
    setResidueName();
    setResidueClass1();
    setResidueClass2();
    setSecondaryStructure1();
    setSecondaryStructure2();
}

double AtomProperties::getCharge() {
	string lookup = residueName + " " + fullName;
	LookupTable &table = lookupTables->get( "CHARGE" );
	if( table.has( lookup ) )      return table[ lookup ];

	LookupTable ions = lookupTables->get( "CHARGE_IONS" );
	if( ions.has( residueName ) ) return ions[ residueName ];
	return 0.0;
}

double AtomProperties::getHydrophobicity() {
    if (isBackbone()) {
        if      (atom->nameIsIn( "N O" )) return (-1.0);
        else if (name == "C")             return (0.0);
    } else if (atom->residueNameIsIn( "ALA VAL PHE MET ILE LEU" ))
        return (1.0);

	string lookup = residueName + " " + fullName;
	LookupTable &table = lookupTables->get( "HYDROPHOBICITY" );

	if( table.has( lookup ))      return table[ lookup ];
	if( table.has( residueName )) return table[ residueName ];

    if      ( name == "O"  ) return -1.0;
    else if ( name == "C"  ) return  0.5;
    else if ( name == "N"  ) return -1.0;
    else if ( name == "S"  ) return  0.0;
    else if ( name == "P"  ) return  0.0;
    else if ( name == "CA" ) return -2.0;
    else if ( name == "CU" ) return -2.0;
    else if ( name == "FE" ) return -2.0;
    else if ( name == "ZN" ) return -2.0;
    else if ( name == "MN" ) return -2.0;
    else if ( name == "MG" ) return -2.0;
    else if ( name == "CL" ) return -2.0;

    debug(3, "Hydrophobicity unknown: %s\n", fullName.c_str()); 
    return (0.0);
}


/**
 * @brief Returns the Van der Waals radius of the residue (Angstroms)
 **/
double AtomProperties::getRadius() {
	string lookup = residueName + " " + fullName;

	LookupTable &radius = lookupTables->get( "RADIUS_CONSTANTS" );
	if      (fullName == "N")  return (1.5);
    else if (fullName == "CA") return (2.0);
    else if (fullName == "C")  return (2.0);
    else if (fullName == "O")  return (1.4);

   if (residueName == "ZN") {
        return ( radius[ "Z2ION" ]);

    } else if (residueName == "O") {
        return ( radius[ "HOH" ]);
	}

	LookupTable &table = lookupTables->get( "RADIUS" );
	if( table.has( lookup )) return table[ lookup ];

    if      (fullName == "OT")  return (radius[ "O1O2H" ]);
    else if (fullName == "OXT") return (radius[ "O1O2H" ]);

	LookupTable &element = lookupTables->get( "RADIUS_ELEMENT" );
	if( element.has( name )) return element[ name ];

    debug(3, "Using default Van derWaals radius: %s\n", fullName.c_str());
    return (1.70);
}

bool AtomProperties::isBackbone() {
    // OXT marks the carboxy terminal oxygen. Also marks terminal oxygen on ASP, GLU
    if (atom->fullNameIsIn( "N C O OXT" )) {
        return true;

    // find C(alpha) but not calcium
    } else if ((fullName == "CA") && (residueName != "CA")) {
        return true;
	}

    return false;
}

void AtomProperties::setAmide() {
	if( ! PropertyList::has( "AMIDE" )) return;
	string lookup = residueName + " " + fullName;
	LookupTable &table = lookupTables->get( "AMIDE" );
	double precision = PropertyList::getDivider( "AMIDE" );
    if (isBackbone() && fullName == "N" ) propertyValue[ "AMIDE" ] = (int) (1.0 * precision);
	else                                  propertyValue[ "AMIDE" ] = (int) (table[ lookup ] * precision);
}

void AtomProperties::setAmine() {
	if( ! PropertyList::has( "AMINE" )) return;
	string lookup = residueName + " " + fullName;
	LookupTable &table = lookupTables->get( "AMINE" );
	double precision = PropertyList::getDivider( "AMINE" );
	propertyValue[ "AMINE" ] = (int) (table[ lookup ] * precision);
}

void AtomProperties::setCarbonyl() {
	if( ! PropertyList::has( "CARBONYL" )) return;
	string lookup = residueName + " " + fullName;
	LookupTable &table = lookupTables->get( "CARBONYL" );
	double precision = PropertyList::getDivider( "CARBONYL" );
    if (fullName == "O") propertyValue[ "CARBONYL" ] = (int) (1.0 * precision);
	else                 propertyValue[ "CARBONYL" ] = (int) (table[ lookup ] * precision);
}

void AtomProperties::setCharge() {
	if( ! PropertyList::has( "CHARGE" )) return;
	double precision = PropertyList::getDivider( "CHARGE" );
	double charge    = getCharge();
	propertyValue[ "CHARGE" ] = (int) fabs(charge * precision);
}

void AtomProperties::setHydrophobicity() {
	if( ! PropertyList::has( "HYDROPHOBICITY" )) return;
	double precision      = PropertyList::getDivider( "HYDROPHOBICITY" );
	double hydrophobicity = getHydrophobicity();
    propertyValue[ "HYDROPHOBICITY" ] = (int) (hydrophobicity * precision);
}

void AtomProperties::setHydroxyl() {
	if( ! PropertyList::has( "HYDROXYL" )) return;
	string lookup = residueName + " " + fullName;
	LookupTable &table = lookupTables->get( "HYDROXYL" );
	double precision   = PropertyList::getDivider( "HYDROXYL" );
    propertyValue[ "HYDROXYL" ] = (int) (table[ lookup ] * precision);
}

/**
 * @brief Returns the mobility of the atom; mobility is defined as the number
 * of bonds from the backbone. Therefore mobility is proportional to degrees
 * of freedom.
 *
 * @note Mobility is also known as "chain distance."
 **/
void AtomProperties::setMobility() {
	if( ! PropertyList::has( "MOBILITY" )) return;
	string lookup = residueName + " " + fullName;
	LookupTable &table = lookupTables->get( "MOBILITY" );

    if (isBackbone()) {
		propertyValue[ "MOBILITY" ] = 0;
	} else {
		double precision = PropertyList::getDivider( "MOBILITY" );
		propertyValue[ "MOBILITY" ] = (int) (table[ lookup ] * precision);
	}
}

/**
 * @brief sets the atomNameIs* to true or false depending on atom->name.
 * @note This function sets the atomNameIs* to true or false depending on atom->name.  True or false is actually the numerical value of 0 or 1 (i.e., this atom has 1 atomNameIs* and no other types of atoms)
 **/
void AtomProperties::setName() {
	if( atom->nameIsIn( "C N O S" )) {
		string element = "ELEMENT_IS_" + name;
		propertyValue[ element ] = 1;
	} else {
		propertyValue[ "ELEMENT_IS_OTHER" ] = 1;
	}
}

/**
 * @brief This function sets the property PEPTIDE to true or false depending on if the atom is part the residue backbone.
 * 
 **/
void AtomProperties::setPeptide() {
	propertyValue[ "PEPTIDE" ] = isBackbone();
}

/**
 * @brief This function sets the property RING_SYSTEM to true or false depending on if the atom is part of a ring system.
 **/
void AtomProperties::setRingSystem() {
	string lookup = residueName + " " + fullName;
	LookupTable &table = lookupTables->get( "RING_SYSTEM" );
	if( table.has( lookup )) propertyValue[ "RING_SYSTEM" ] = (int) table[ lookup ];
	else                     propertyValue[ "RING_SYSTEM" ] = 0;
}

void AtomProperties::setSolventAccessibility() {
    propertyValue[ "SOLVENT_ACCESSIBILITY" ] = atom->getSolventAccessibility();
}

/**
 * @brief This function sets the "atom type" field, depending on the AMBER forcefield atom type
 **/
void AtomProperties::setType() {
	string type = atom->getAtomType();
    if      (type == "C"  ) propertyValue[ "ATOM_TYPE_IS_C" ]     = 1;
    else if (type == "Ca" ) propertyValue[ "ATOM_TYPE_IS_CA" ]    = 1;
    else if (type == "CT" ) propertyValue[ "ATOM_TYPE_IS_CT" ]    = 1;
    else if (type == "N"  ) propertyValue[ "ATOM_TYPE_IS_N" ]     = 1;
    else if (type == "N2" ) propertyValue[ "ATOM_TYPE_IS_N2" ]    = 1;
    else if (type == "N3" ) propertyValue[ "ATOM_TYPE_IS_N3" ]    = 1;
    else if (type == "Na" ) propertyValue[ "ATOM_TYPE_IS_NA" ]    = 1;
    else if (type == "O"  ) propertyValue[ "ATOM_TYPE_IS_O" ]     = 1;
    else if (type == "O2" ) propertyValue[ "ATOM_TYPE_IS_O2" ]    = 1;
    else if (type == "OH" ) propertyValue[ "ATOM_TYPE_IS_OH" ]    = 1;
    else if (type == "S"  ) propertyValue[ "ATOM_TYPE_IS_S" ]     = 1;
    else if (type == "SH" ) propertyValue[ "ATOM_TYPE_IS_SH" ]    = 1;
    else                    propertyValue[ "ATOM_TYPE_IS_OTHER" ] = 1;
}

void 
AtomProperties::setPartialCharge() {
	if( ! PropertyList::has( "PARTIAL_CHARGE" )) return;
	double precision     = PropertyList::getDivider( "PARTIAL_CHARGE" );
	double partialCharge = atom->getPartialCharge();
    propertyValue[ "PARTIAL_CHARGE" ] = (int) (partialCharge * precision);
}

/**
 * @brief took the radius and used it to calculate the volume of a sphere
 * 
 **/
void 
AtomProperties::setVDWVolume() {
	if( ! PropertyList::has( "VDW_VOLUME" )) return;
    double radius    = getRadius();
	double precision = PropertyList::getDivider( "VDW_VOLUME" );
    propertyValue[ "VDW_VOLUME" ]     = (int) (precision * ((double) 4 * M_PI / 3 * radius * radius * radius)); //keeps 3 sig digits
}

/**
 * @brief sets the Negative Charge depending on the charge of the atom
 * 
 **/
void AtomProperties::setNegCharge() {
	if( ! PropertyList::has( "NEG_CHARGE" )) return;
    double charge    = getCharge();
	double precision = PropertyList::getDivider( "NEG_CHARGE" );
    if (charge < 0) propertyValue[ "NEG_CHARGE" ] = (int) (-precision * charge); //keeps 3 sig digits and makes pos
    else            propertyValue[ "NEG_CHARGE" ] = 0;
}

/**
 * @brief Same as positive Charge depending on the charge of the atom.
 * 
 **/
void AtomProperties::setPosCharge() {
	if( ! PropertyList::has( "POS_CHARGE" )) return;
    double charge = getCharge();
	double precision = PropertyList::getDivider( "POS_CHARGE" );
    if (charge > 0) propertyValue[ "POS_CHARGE" ] = (int) (precision * charge); //keeps 3 sig digits
    else            propertyValue[ "POS_CHARGE" ] = 0;
}

/**
 * @brief Sets the charge for nitrogens and protonated nitrogens in the histidine imidazole ring
 **/
void AtomProperties::setChargeWithHis() {
	if( ! PropertyList::has( "CHARGE_WITH_HIS" )) return;
    double charge;
    if (atom->getResidueName() == "HIS") {
        if (atom->fullNameIsIn( "ND1 NE2" ))
            charge = 0.5;
        else if (atom->fullNameIsIn( "AD1 AD2 AE1 AE2" ))
            charge = 0.25;
        else charge = 0.0;
    } else charge = getCharge();

    propertyValue[ "CHARGE_WITH_HIS" ] = (int) (1000 * fabs(charge)); //keeps 3 sig digits
}

/**
 * @brief Sets the Functional Group values
 *
 **/
void AtomProperties::setFunctionalGroup() {
	string residueName  = atom->getResidueName();
	string fullName     = atom->getFullName();
	string name         = atom->getName();

	if     ( fullName == "CA" )                           { propertyValue[ "PARTIAL_POSITIVE_CARBON" ]        = 1; }
	else if( fullName == "C" )                            { propertyValue[ "AMIDE_CARBON" ]                   = 1; }
	else if( fullName == "N" )                            { propertyValue[ "AMIDE_NITROGEN" ]                 = 1; }
	else if( fullName == "O" )                            { propertyValue[ "AMIDE_OXYGEN" ]                   = 1; }
	else if( atom->residueNameIsIn( "ALA VAL LEU ILE" ))  { propertyValue[ "ALIPHATIC_CARBON" ]               = 1; }
	else if( residueName == "SER" ) {
		if     ( name == "O" )                            { propertyValue[ "HYDROXYL_OXYGEN" ]                = 1; }
		else                                              { propertyValue[ "ALIPHATIC_CARBON_NEXT_TO_POLAR" ] = 1; }}
	else if( residueName == "THR")  {
		if     ( name     == "O" )                        { propertyValue[ "HYDROXYL_OXYGEN" ]                = 1; }
		else if( fullName == "CB" )                       { propertyValue[ "ALIPHATIC_CARBON_NEXT_TO_POLAR" ] = 1; }
		else                                              { propertyValue[ "ALIPHATIC_CARBON" ]               = 1; }}
	else if( atom->residueNameIsIn( "CYS CYX MET" )) {
		if     ( name == "S")                             { propertyValue[ "SULFUR" ]                         = 1; }
		else                                              { propertyValue[ "ALIPHATIC_CARBON" ]               = 1; }}
	else if( residueName == "ASP" ) {
		if     ( fullName == "CB" )                       { propertyValue[ "PARTIAL_POSITIVE_CARBON" ]        = 1; }
		else if( fullName == "CG" )                       { propertyValue[ "CARBOXYL_CARBON" ]                = 1; }
		else if( name     == "O" )                        { propertyValue[ "CARBOXYL_OXYGEN" ]                = 1; }}
	else if( residueName == "GLU" ) {
		if     ( fullName == "CG" )                       { propertyValue[ "PARTIAL_POSITIVE_CARBON" ]        = 1; }
		else if( fullName == "CD" )                       { propertyValue[ "CARBOXYL_CARBON" ]                = 1; }
		else if( name     == "O" )                        { propertyValue[ "CARBOXYL_OXYGEN" ]                = 1; }
		else                                              { propertyValue[ "ALIPHATIC_CARBON" ]               = 1; }}
	else if( residueName == "ASN") { //AD1 and AD2 handled correctly
		if     ( fullName == "CB" )                       { propertyValue[ "ALIPHATIC_CARBON_NEXT_TO_POLAR" ] = 1; }
		else if( fullName == "CG" )                       { propertyValue[ "AMIDE_CARBON" ]                   = 1; }
		else if( name     == "N" )                        { propertyValue[ "AMIDE_NITROGEN" ]                 = 1; }
		else                                              { propertyValue[ "AMIDE_OXYGEN" ]                   = 1; }}
	else if( residueName == "GLN")  { //AE1 and AE2 handled correctly
		if     ( fullName == "CB" )                       { propertyValue[ "ALIPHATIC_CARBON" ]               = 1; }
		else if( fullName == "CG" )                       { propertyValue[ "PARTIAL_POSITIVE_CARBON" ]        = 1; }
		else if( fullName == "CD" )                       { propertyValue[ "AMIDE_CARBON" ]                   = 1; }
		else if( name     == "N" )                        { propertyValue[ "AMIDE_NITROGEN" ]                 = 1; }
		else                                              { propertyValue[ "AMIDE_OXYGEN" ]                   = 1; }}
	else if( residueName == "LYS" ) {
		if     ( fullName == "CE" )                       { propertyValue[ "PARTIAL_POSITIVE_CARBON" ]        = 1; }
		else if( name     == "N" )                        { propertyValue[ "POSITIVE_NITROGEN" ]              = 1; }
		else                                              { propertyValue[ "ALIPHATIC_CARBON" ]               = 1; }}
	else if( residueName == "ARG") {
		if     ( fullName == "CZ" )                       { propertyValue[ "PARTIAL_POSITIVE_CARBON" ]        = 1; }
		else if( fullName == "CD" )                       { propertyValue[ "ALIPHATIC_CARBON_NEXT_TO_POLAR" ] = 1; }
		else if( name     == "N" )                        { propertyValue[ "POSITIVE_NITROGEN" ]              = 1; }
		else                                              { propertyValue[ "ALIPHATIC_CARBON" ]               = 1; }}
	else if( residueName == "HIS") {
		if     ( atom->fullNameIsIn( "CG CD2 CE1" ))      { propertyValue[ "AROMATIC_CARBON" ]                = 1; }
		else if( atom->fullNameIsIn( "ND1 NE2" ))         { propertyValue[ "AROMATIC_NITROGEN" ]              = 1; }
		else if( atom->fullNameIsIn( "AD1 AD2 AE1 AE2" )) { propertyValue[ "AROMATIC_NITROGEN" ]              = 1; }
		else                                              { propertyValue[ "ALIPHATIC_CARBON" ]               = 1; }}
	else if( residueName == "PHE") {
		if       ( fullName == "CB" )                     { propertyValue[ "ALIPHATIC_CARBON" ]               = 1; }
		else                                              { propertyValue[ "AROMATIC_CARBON" ]                = 1; }}
	else if( residueName == "TYR" ) {
		if       ( name == "O" )                          { propertyValue[ "HYDROXYL_OXYGEN" ]                = 1; }
		else                                              { propertyValue[ "AROMATIC_CARBON" ]                = 1; }}
	else if( residueName == "TRP") {
		if     ( fullName == "CB" )                       { propertyValue[ "ALIPHATIC_CARBON" ]               = 1; }
		else if( name     == "N" )                        { propertyValue[ "AROMATIC_NITROGEN" ]              = 1; }
		else                                              { propertyValue[ "AROMATIC_CARBON" ]                = 1; }}
}

/**
 * @brief converts the string value of the property field in Atom to an integer value of 0 or 1 
 **/
void AtomProperties::setResidueName() {
	const char * residues = 
		"ALA ARG ASN ASP CYS CYX GLN GLU GLY HIS ILE "
		"LEU LYS MET PHE PRO SER THR TRP TYR VAL HOH";
	if( atom->residueNameIsIn( residues ) ) {
		string residue = "RESIDUE_NAME_IS_" + atom->getResidueName();
		if( residueName == "CYX" ) residue = "RESIDUE_NAME_IS_CYS";
		propertyValue[ residue ] = 1;
	} else {
		propertyValue[ "RESIDUE_NAME_IS_OTHER" ] = 1;
	}
}

void AtomProperties::setResidueClass1() {
	const char *hydrophobic = "ALA VAL PHE PRO MET ILE LEU";
	const char *charged     = "ASP GLU LYS ARG";
	const char *polar       = "SER THR TYR HIS CYS CYX ASN GLN TRP";

    if      (atom->residueNameIsIn( hydrophobic )) propertyValue[ "RESIDUE_CLASS1_IS_HYDROPHOBIC" ] = 1;
    else if (atom->residueNameIsIn( charged ))     propertyValue[ "RESIDUE_CLASS1_IS_CHARGED" ]     = 1;
    else if (atom->residueNameIsIn( polar ))       propertyValue[ "RESIDUE_CLASS1_IS_POLAR" ]       = 1;
    else                                           propertyValue[ "RESIDUE_CLASS1_IS_UNKNOWN" ]     = 1;
}

void AtomProperties::setResidueClass2() {
	const char *nonpolar = "ALA VAL LEU ILE PRO PHE TRP MET";
	const char *polar    = "GLY SER THR CYS CYX TYR ASN GLN";
	const char *acidic   = "ASP GLU";
	const char *basic    = "LYS ARG HIS";

    if      (atom->residueNameIsIn( nonpolar )) propertyValue[ "RESIDUE_CLASS2_IS_NONPOLAR" ] = 1;
    else if (atom->residueNameIsIn( polar ))    propertyValue[ "RESIDUE_CLASS2_IS_POLAR" ] = 1;
    else if (atom->residueNameIsIn( acidic ))   propertyValue[ "RESIDUE_CLASS2_IS_ACIDIC" ] = 1;
    else if (atom->residueNameIsIn( basic ))    propertyValue[ "RESIDUE_CLASS2_IS_BASIC" ] = 1;
    else                                        propertyValue[ "RESIDUE_CLASS2_IS_UNKNOWN" ] = 1;

}

void AtomProperties::setSecondaryStructure1() {
	switch (atom->getSecondaryStructure()) {
		case Atom::Helix3: propertyValue[ "SECONDARY_STRUCTURE1_IS_3HELIX" ]  = 1; break;
		case Atom::Helix4: propertyValue[ "SECONDARY_STRUCTURE1_IS_4HELIX" ]  = 1; break;
		case Atom::Helix5: propertyValue[ "SECONDARY_STRUCTURE1_IS_5HELIX" ]  = 1; break;
		case Atom::Strand: propertyValue[ "SECONDARY_STRUCTURE1_IS_STRAND" ]  = 1; break;
		case Atom::Bend:   propertyValue[ "SECONDARY_STRUCTURE1_IS_BEND" ]    = 1; break;
		case Atom::Turn:   propertyValue[ "SECONDARY_STRUCTURE1_IS_TURN" ]    = 1; break;
		case Atom::Bridge: propertyValue[ "SECONDARY_STRUCTURE1_IS_BRIDGE" ]  = 1; break;
		case Atom::Coil:   propertyValue[ "SECONDARY_STRUCTURE1_IS_COIL" ]    = 1; break;
		case Atom::Het:    propertyValue[ "SECONDARY_STRUCTURE1_IS_HET" ]     = 1; break;
		default:           propertyValue[ "SECONDARY_STRUCTURE1_IS_UNKNOWN" ] = 1; break;
	}
}

void AtomProperties::setSecondaryStructure2() {
    switch (atom->getSecondaryStructure()) {
        case Atom::Helix3:
        case Atom::Helix4:
        case Atom::Helix5: propertyValue[ "SECONDARY_STRUCTURE2_IS_HELIX" ]   = 1; break;
        case Atom::Strand:
        case Atom::Bend:
        case Atom::Turn:
        case Atom::Bridge: propertyValue[ "SECONDARY_STRUCTURE2_IS_BETA" ]    = 1; break;
        case Atom::Coil:   propertyValue[ "SECONDARY_STRUCTURE2_IS_COIL" ]    = 1; break;
        case Atom::Het:    propertyValue[ "SECONDARY_STRUCTURE2_IS_HET" ]     = 1; break;
        default:           propertyValue[ "SECONDARY_STRUCTURE2_IS_UNKNOWN" ] = 1; break;
    }

}

bool AtomProperties::LookupTable::has( string lookup ) {
	LookupTable::iterator search;
	search = find( lookup );
	return (search != end());
}

AtomProperties::LookupTables::LookupTables() {
	// =======================================================================
	// AMIDE LOOKUP TABLE
	// =======================================================================
	{
		LookupTable table;
		table[ "PRO N" ]   = table[ "ASN ND2"]  = table[ "GLN NE2" ] = 1.0;
		table[ "ASN AD1" ] = table[ "ASN AD2" ] = table[ "GLN AE1" ] = table[ "GLN AE2" ] = table[ "HIS ND1" ] = table[ "HIS NE2" ] = table[ "ARG NH1" ] = table[ "ARG NH2" ] = 0.5;
		table[ "HIS AD1" ] = table[ "HIS AD2" ] = table[ "HIS AE1" ] = table[ "HIS AE2" ] = 0.25;
		add( "AMIDE", table );
	}

	// =======================================================================
	// AMINE LOOKUP TABLE
	// =======================================================================
	{
		LookupTable table;
		table[ "ARG NE" ]  = table[ "LYS NZ" ]  = table[ "TRP NE1" ] = 1.0;
		table[ "ARG NH1" ] = table[ "ARG NH2" ] = table[ "HIS ND1" ] = table[ "HIS NE2" ] = 0.5;
		table[ "HIS AD1" ] = table[ "HIS AD2" ] = table[ "HIS AE1" ] = table[ "HIS AE2" ] = 0.25;
		add( "AMINE", table );
	}

	// =======================================================================
	// CARBONYL LOOKUP TABLE
	// =======================================================================
	{
		LookupTable table;
		table[ "ASN OD1" ] = table[ "GLN OE1" ] = 1.0;
		table[ "ASP OD1" ] = table[ "ASP OD2" ] = table[ "GLU OE1" ] = table[ "GLU OE2" ] = table[ "ASN AD1" ] = table[ "ASN AD2" ] = table[ "GLN AE1" ] = table[ "GLN AE2" ] = 0.5;
		add( "CARBONYL", table );
	}

	// =======================================================================
	// CHARGE LOOKUP TABLE
	// =======================================================================
	// Polar atoms in acidic or basic residues
	{
		LookupTable table;
		table[ "ASP CG"  ] = table[ "ASP OD1" ] = table[ "ASP OD2" ] = -1.0/3.0;
		table[ "GLU CD"  ] = table[ "GLU OE1" ] = table[ "GLU OE2" ] = -1.0/3.0;
		table[ "LYS NZ"  ] = 1.0;
		table[ "ARG CZ"  ] = table[ "ARG NH1" ] = table[ "ARG NH2" ] =  1.0/3.0;
		add( "CHARGE", table );
	}
	// Metal ions and halides
	{
		LookupTable table;
		table[ "CA" ] = table[ "CU" ] = table[ "FE" ] = 2.0;
		table[ "MG" ] = table[ "MN" ] = table[ "ZN" ] = 2.0;
		table[ "CL" ] = -1.0;
		add( "CHARGE_IONS", table );
	}

	// =======================================================================
	// HYDROPHOBICITY LOOKUP TABLE
	// =======================================================================
	{
		LookupTable table;
		table[ "PRO CD"  ] =  0.0; table[ "PRO"     ] =  1.0;
		table[ "ASP CB"  ] =  1.0; table[ "ASP CG"  ] =  0.0; table[ "ASP OD1" ] = -1.0; table[ "ASP OD2" ] = -1.0;
		table[ "GLU CB"  ] =  1.0; table[ "GLU CG"  ] =  1.0; table[ "GLU CD"  ] =  0.0; table[ "GLU OE1" ] = -1.0; table[ "GLU OE2" ] = -1.0;
		table[ "LYS CB"  ] =  1.0; table[ "LYS CG"  ] =  1.0; table[ "LYS CD"  ] =  1.0; table[ "LYS NZ"  ] =  1.0; table[ "LYS CE"  ] = -1.0;
		table[ "ARG CB"  ] =  1.0; table[ "ARG CG"  ] =  1.0; table[ "ARG CD"  ] =  0.0; table[ "ARG CZ"  ] =  0.0; table[ "ARG NE"  ] = -1.0; table[ "ARG NH1" ] = -1.0; table[ "ARG NH2" ] = -1.0;
		table[ "SER CB"  ] =  0.0; table[ "SER"     ] = -1.0;
		table[ "THR CB"  ] =  0.0; table[ "THR CG2" ] =  1.0; table[ "THR OG1" ] = -1.0;
		table[ "TYR OH"  ] = -1.0; table[ "TYR CZ"  ] =  0.0; table[ "TYR"     ] =  1.0;
		table[ "HIS CB"  ] =  1.0; table[ "HIS CG"  ] =  0.0; table[ "HIS CE1" ] =  0.0; table[ "HIS CD2" ] =  0.0; table[ "HIS ND1" ] = -1.0; table[ "HIS NE2" ] = -1.0;
		table[ "CYS CB"  ] =  0.0; table[ "CYS SG"  ] = -1.0;
		table[ "CYX CB"  ] =  0.0; table[ "CYX SG"  ] = -1.0;
		table[ "ASN CB"  ] =  1.0; table[ "ASN CG"  ] =  0.0; table[ "ASN OD1" ] = -1.0; table[ "ASN ND2" ] = -1.0; table[ "ASN AD1" ] = -1.0; table[ "ASN AD2" ] = -1.0;
		table[ "GLN CB"  ] =  1.0; table[ "GLN CG"  ] =  1.0; table[ "GLN CD"  ] =  0.0; table[ "GLN OE1" ] = -1.0; table[ "GLN NE2" ] = -1.0; table[ "GLN AE1" ] = -1.0; table[ "GLN AE2" ] = -1.0;
		table[ "TRP NE1" ] = -1.0; table[ "TRP CD1" ] =  0.0; table[ "TRP CE2" ] =  0.0; table[ "TRP"     ] =  1.0;
		add( "HYDROPHOBICITY", table );
	}

	// =======================================================================
	// HYDROXYL LOOKUP TABLE
	// =======================================================================
	{
		LookupTable table;
		table[ "SER OG"  ] = table[ "THR OG1" ] =  table[ "TYR OH"  ] = 1.0;
		table[ "CYS SG"  ] = table[ "CYX SG"  ] =  0.5;
		add( "HYDROXYL", table );
	}

	// =======================================================================
	// MOBILITY LOOKUP TABLE
	// =======================================================================
	// The mobility is defined as the number of bonds from the backbone.
	// Therefore mobility is proportional to degrees of freedom.
	// -----------------------------------------------------------------------
	{
		LookupTable table;
		table[ "ALA CB"  ] = 1;
		table[ "ARG CB"  ] = 1; table[ "ARG CG"  ] = 2; table[ "ARG CD"  ] = 3; table[ "ARG NE"  ] = 4; table[ "ARG CZ"  ] = 5; table[ "ARG NH1" ] = 6; table[ "ARG NH2" ] = 6;
		table[ "ASN CB"  ] = 1; table[ "ASN CG"  ] = 2; table[ "ASN OD1" ] = 3; table[ "ASN AD1" ] = 3; table[ "ASN ND2" ] = 3; table[ "ASN AD2" ] = 3;
		table[ "ASP CB"  ] = 1; table[ "ASP CG"  ] = 2; table[ "ASP OD1" ] = 3; table[ "ASP OD2" ] = 3;
		table[ "CYS CB"  ] = 1; table[ "CYS SG"  ] = 2;
		table[ "CYX CB"  ] = 1; table[ "CYX SG"  ] = 2; 
		table[ "GLN CB"  ] = 1; table[ "GLN CG"  ] = 2; table[ "GLN CD"  ] = 3; table[ "GLN OE1" ] = 4; table[ "GLN AE1" ] = 4; table[ "GLN NE2" ] = 4; table[ "GLN AE2" ] = 4;
		table[ "GLU CB"  ] = 1; table[ "GLU CG"  ] = 2; table[ "GLU CD"  ] = 3; table[ "GLU OE1" ] = 4; table[ "GLU OE2" ] = 4;
		table[ "HIS CB"  ] = 1; table[ "HIS CG"  ] = 2; table[ "HIS ND1" ] = 3; table[ "HIS AD1" ] = 3; table[ "HIS CD2" ] = 3; table[ "HIS AD2" ] = 3; table[ "HIS CE1" ] = 4; table[ "HIS AE1" ] = 4; table[ "HIS NE2" ] = 4; table[ "HIS AE2" ] = 4;
		table[ "HYP CD"  ] = 1; table[ "HYP CB"  ] = 1; table[ "HYP CG"  ] = 2; table[ "HYP OD"  ] = 3;
		table[ "ILE CB"  ] = 1; table[ "ILE CG1" ] = 2; table[ "ILE CG2" ] = 2; table[ "ILE CD1" ] = 3;
		table[ "LEU CB"  ] = 1; table[ "LEU CG"  ] = 2; table[ "LEU CD1" ] = 3; table[ "LEU CD2" ] = 3;
		table[ "LYS CB"  ] = 1; table[ "LYS CG"  ] = 2; table[ "LYS CD"  ] = 3; table[ "LYS CE"  ] = 4; table[ "LYS NZ"  ] = 5;
		table[ "MET CB"  ] = 1; table[ "MET CG"  ] = 2; table[ "MET SD"  ] = 3; table[ "MET CE"  ] = 4;
		table[ "PHE CB"  ] = 1; table[ "PHE CG"  ] = 2; table[ "PHE CD1" ] = 3; table[ "PHE CD2" ] = 3; table[ "PHE CE1" ] = 4; table[ "PHE CE2" ] = 4; table[ "PHE CZ"  ] = 5;
		table[ "PRO CD"  ] = 1; table[ "PRO CB"  ] = 1; table[ "PRO CG"  ] = 2;
		table[ "SER CB"  ] = 1; table[ "SER OG"  ] = 2; 
		table[ "THR CB"  ] = 1; table[ "THR OG1" ] = 2; table[ "THR CG2" ] = 2;
		table[ "TRP CB"  ] = 1; table[ "TRP CG"  ] = 2; table[ "TRP CD1" ] = 3; table[ "TRP CD2" ] = 3; table[ "TRP CE3" ] = 4; table[ "TRP NE1" ] = 4; table[ "TRP CE2" ] = 4; table[ "TRP CZ3" ] = 5; table[ "TRP CZ2" ] = 5; table[ "TRP CH2" ] = 6;
		table[ "TYR CB"  ] = 1; table[ "TYR CG"  ] = 2; table[ "TYR CD1" ] = 3; table[ "TYR CD2" ] = 3; table[ "TYR CE1" ] = 4; table[ "TYR CE2" ] = 4; table[ "TYR CZ"  ] = 5; table[ "TYR OH"  ] = 6;
		table[ "VAL CB"  ] = 1; table[ "VAL CG1" ] = 2; table[ "VAL CG2" ] = 2;
		add( "MOBILITY", table );
	}

	// =======================================================================
	// RADIUS LOOKUP TABLE
	// =======================================================================
	// Atom radius constants values
	{
		LookupTable radius;
		radius[ "C4H"    ] = 2.00; radius[ "C4HH"   ] = 2.00; radius[ "C4HHH"  ] = 2.00; radius[ "C3"     ] = 1.70;
		radius[ "C3H"    ] = 1.85; radius[ "C3HH"   ] = 1.85; radius[ "O1"     ] = 1.40; radius[ "O2H"    ] = 1.60;
		radius[ "O1O2H"  ] = 1.50; radius[ "N4H"    ] = 2.00; radius[ "N4HH"   ] = 2.00; radius[ "N4HHH"  ] = 2.00;
		radius[ "N3"     ] = 1.50; radius[ "N3H"    ] = 1.70; radius[ "N3HH"   ] = 1.80; radius[ "S2"     ] = 1.85;
		radius[ "S2H"    ] = 2.00; radius[ "O1N3HH" ] = 1.60; radius[ "Z2ION"  ] = 0.74; radius[ "FE"     ] = 1.70;
		radius[ "HOH"    ] = 1.40;
		add( "RADIUS_CONSTANTS", radius );

		LookupTable table;
		table[ "ALA CB"  ] = radius[ "C4HHH"  ];
		table[ "ARG NH2" ] = radius[ "N3HH"   ]; table[ "ARG NEH" ] = radius[ "N3HH"   ]; table[ "ARG NH1" ] = radius[ "N3HH"   ]; table[ "ARG CZ"  ] = radius[ "C3"     ]; table[ "ARG NE"  ] = radius[ "N3H"    ]; table[ "ARG CD"  ] = radius[ "C4HH"   ]; table[ "ARG CG"  ] = radius[ "C4HH"   ]; table[ "ARG CB"  ] = radius[ "C4HH"   ];
		table[ "ASN ND2" ] = radius[ "N3HH"   ]; table[ "ASN OD1" ] = radius[ "O1"     ]; table[ "ASN AD1" ] = radius[ "O1N3HH" ]; table[ "ASN AD2" ] = radius[ "O1N3HH" ]; table[ "ASN NOD" ] = radius[ "O1N3HH" ]; table[ "ASN CG"  ] = radius[ "C3"     ]; table[ "ASN CB"  ] = radius[ "C4HH"   ]; table[ "ASP OD2" ] = radius[ "O1O2H"  ]; table[ "ASP OD1" ] = radius[ "O1O2H"  ]; table[ "ASP CG"  ] = radius[ "C3"     ]; table[ "ASP CB"  ] = radius[ "C4HH"   ];
		table[ "CYS SG"  ] = radius[ "S2"     ]; table[ "CYS CB"  ] = radius[ "C4HH"   ];
		table[ "CYX SG"  ] = radius[ "S2"     ]; table[ "CYX CB"  ] = radius[ "C4HH"   ];
		table[ "GLN NE2" ] = radius[ "N3HH"   ]; table[ "GLN OE1" ] = radius[ "O1"     ]; table[ "GLN AE1" ] = radius[ "O1O2H"  ]; table[ "GLN AE2" ] = radius[ "O1O2H"  ]; table[ "GLN NOE" ] = radius[ "O1O2H"  ]; table[ "GLN CD"  ] = radius[ "C3"     ]; table[ "GLN CG"  ] = radius[ "C4HH"   ]; table[ "GLN CB"  ] = radius[ "C4HH"   ];
		table[ "GLU OE2" ] = radius[ "O1O2H"  ]; table[ "GLU OE1" ] = radius[ "O1O2H"  ]; table[ "GLU CD"  ] = radius[ "C3"     ]; table[ "GLU CG"  ] = radius[ "C4HH"   ]; table[ "GLU CB"  ] = radius[ "C4HH"   ];
		table[ "HIS CD2" ] = radius[ "C3H"    ]; table[ "HIS NE2" ] = radius[ "N3H"    ]; table[ "HIS CE1" ] = radius[ "C3H"    ]; table[ "HIS ND1" ] = radius[ "N3"     ]; table[ "HIS CG"  ] = radius[ "C3"     ]; table[ "HIS CB"  ] = radius[ "C4HH"   ];
		table[ "ILE CD1" ] = radius[ "C4HHH"  ]; table[ "ILE CG1" ] = radius[ "C4HH"   ]; table[ "ILE CB"  ] = radius[ "C4H"    ]; table[ "ILE CG2" ] = radius[ "C4HHH"  ];
		table[ "LEU CD2" ] = radius[ "C4HHH"  ]; table[ "LEU CD1" ] = radius[ "C4HHH"  ]; table[ "LEU CG"  ] = radius[ "C4H"    ]; table[ "LEU CB"  ] = radius[ "C4HH"   ];
		table[ "LYS NZ"  ] = radius[ "N4HHH"  ]; table[ "LYS CE"  ] = radius[ "C4HH"   ]; table[ "LYS CD"  ] = radius[ "C4HH"   ]; table[ "LYS CG"  ] = radius[ "C4HH"   ]; table[ "LYS CB"  ] = radius[ "C4HH"   ];
		table[ "MET CE"  ] = radius[ "C4HHH"  ]; table[ "MET SD"  ] = radius[ "S2"     ]; table[ "MET CG"  ] = radius[ "C4HH"   ]; table[ "MET CB"  ] = radius[ "C4HH"   ];
		table[ "PHE CD2" ] = radius[ "C3H"    ]; table[ "PHE CE2" ] = radius[ "C3H"    ]; table[ "PHE CZ"  ] = radius[ "C3H"    ]; table[ "PHE CE1" ] = radius[ "C3H"    ]; table[ "PHE CD1" ] = radius[ "C3H"    ]; table[ "PHE CG"  ] = radius[ "C3"     ]; table[ "PHE CB"  ] = radius[ "C4HH"   ];
		table[ "PRO CG"  ] = radius[ "C4HH"   ]; table[ "PRO CD"  ] = radius[ "C4HH"   ]; table[ "PRO CB"  ] = radius[ "C4HH"   ];
		table[ "SER OG"  ] = radius[ "O2H"    ]; table[ "SER OG1" ] = radius[ "O2H"    ]; table[ "SER CB"  ] = radius[ "C4HH"   ];
		table[ "THR CG2" ] = radius[ "C4HHH"  ]; table[ "THR OG"  ] = radius[ "O2H"    ]; table[ "THR OG1" ] = radius[ "O2H"    ]; table[ "THR CB"  ] = radius[ "C4H"    ];
		table[ "TRP CD2" ] = radius[ "C3"     ]; table[ "TRP CE3" ] = radius[ "C3H"    ]; table[ "TRP CZ3" ] = radius[ "C3H"    ]; table[ "TRP CH2" ] = radius[ "C3H"    ]; table[ "TRP CEH" ] = radius[ "C3H"    ]; table[ "TRP CZ2" ] = radius[ "C3H"    ]; table[ "TRP CE2" ] = radius[ "C3"     ]; table[ "TRP NE1" ] = radius[ "N3H"    ]; table[ "TRP CD1" ] = radius[ "C3H"    ]; table[ "TRP CG"  ] = radius[ "C3"     ]; table[ "TRP CB"  ] = radius[ "C4HH"   ];
		table[ "TYR OH"  ] = radius[ "O2H"    ]; table[ "TYR OEH" ] = radius[ "O2H"    ]; table[ "TYR CD2" ] = radius[ "C3H"    ]; table[ "TYR CE2" ] = radius[ "C3H"    ]; table[ "TYR CZ"  ] = radius[ "C3"     ]; table[ "TYR CE1" ] = radius[ "C3H"    ]; table[ "TYR CD1" ] = radius[ "C3H"    ]; table[ "TYR CG"  ] = radius[ "C3"     ]; table[ "TYR CB"  ] = radius[ "C4HH"   ];
		table[ "VAL CG2" ] = radius[ "C4HHH"  ]; table[ "VAL CG1" ] = radius[ "C4HHH"  ]; table[ "VAL CB"  ] = radius[ "C4H"    ];
		add( "RADIUS", table );

		LookupTable element;
		element[ "N"  ] =  1.50; element[ "C"  ] =  2.00; element[ "O"  ] =  1.40; element[ "H"  ] =  1.00;
		element[ "1H" ] =  1.00; element[ "2H" ] =  1.00; element[ "3H" ] =  1.00; element[ "S"  ] =  1.85;
		element[ "FE" ] =  1.70; element[ "CU" ] =  0.65; element[ "P"  ] =  1.90; element[ "CA" ] =  0.99;
		element[ "CL" ] =  1.80;
		add( "RADIUS_ELEMENT", element );
	}
	// =======================================================================
	// RING SYSTEM LOOKUP TABLE
	// =======================================================================
	{
		LookupTable table;
		table[ "PHE CG"  ] = table[ "PHE CD1" ] = table[ "PHE CD2" ] = table[ "PHE CE1" ] = table[ "PHE CE2" ] = table[ "PHE CZ"  ] = 1.0;
		table[ "TYR CG"  ] = table[ "TYR CD1" ] = table[ "TYR CD2" ] = table[ "TYR CE1" ] = table[ "TYR CE2" ] = table[ "TYR CZ"  ] = 1.0;
		table[ "TRP CG"  ] = table[ "TRP CD1" ] = table[ "TRP CD2" ] = table[ "TRP NE1" ] = table[ "TRP CE2" ] = table[ "TRP CE3"  ] = table[ "TRP CZ2" ] = table[ "TRP CZ3"  ] = table[ "TRP CH2" ] = 1.0;
		table[ "HIS CG"  ] = table[ "HIS ND1" ] = table[ "HIS CD2" ] = table[ "HIS CE1" ] = table[ "HIS NE2" ] = table[ "HIS AD1"  ] = table[ "HIS AD2" ] = table[ "HIS AE1"  ] = table[ "HIS AE2" ] = 1.0;
		add( "RING_SYSTEM", table );
	}
}

AtomProperties::LookupTables *AtomProperties::LookupTables::getInstance() {
	if( ! isInitialized ) {
		singleton = new LookupTables();
		isInitialized = true;
	}
	return singleton;
}

AtomProperties::LookupTable &AtomProperties::LookupTables::get( string name ) {
	unordered_map< string, LookupTable>::iterator search;
	search = tables.find( name );
	if( search != tables.end() ) return tables[ name ];

	error( "Table '%s' not defined.\n", name.c_str() );
	exit( -1 );
}

void AtomProperties::LookupTables::add( string name, LookupTable &table ) {
	tables[ name ] = table;
}
