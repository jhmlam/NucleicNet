// FEATURE
// See COPYING and AUTHORS for copyright and license agreement terms.

// $Id: Atom.cc 1603 2013-12-04 00:00:59Z mikewong899 $

#include "Atom.h"

Atom::Atom() { 
	serialNumber        = 0;
	altLocation         = 0;
	chainID             = 0;
	residueSequence     = 0;
	insCode             = 0;
	coord               = Point3D();
	occupancy           = 0.0;
	bFactor             = 0.0;
	secondaryStructure  = Coil;
	solventAcc          = 0;
	isHetatm            = false;
	partialCharge       = 0;
	mark                = false;

	name                = "";
	fullName            = "";
	residueName         = "";
	residueFullSequence = "";
	atomType            = "";
}

Atom::Atom( Atom *atom ) {
   	serialNumber        = atom->serialNumber;
	altLocation         = atom->altLocation;
	chainID             = atom->chainID;
	residueSequence     = atom->residueSequence;
	insCode             = atom->insCode;
	coord               = Point3D( atom->coord.x, atom->coord.y, atom->coord.z );
	occupancy           = atom->occupancy;
	bFactor             = atom->bFactor;
	secondaryStructure  = atom->secondaryStructure;
	solventAcc          = atom->solventAcc;
	isHetatm            = atom->isHetatm;
	partialCharge       = atom->partialCharge;

	name                = atom->name;
	fullName            = atom->fullName;
	residueName         = atom->residueName;
	residueFullSequence = atom->residueFullSequence;
	atomType            = atom->atomType;
	mark                = atom->mark;
	residueID           = ResidueID( atom->residueSequence, atom->chainID, atom->insCode );
}

Atom::~Atom() {
}
 
/**
 *  A "strict weak ordering" for atoms for use in set containers First looks
 *  at residueSequence, then chainID, then insCode, and finally full atom
 *  name.
 *
 *  This ordering is from an old for-loop based implementation of atom
 *  comparison, but it seems to work well enough
 *
 * @note It may make more sense to just order from residueSequence, chainID
 * and insCode to a vector containing all atoms, as the number of atoms per
 * residue is rather low and a linear search would not cost much
 **/
bool Atom::operator< (const Atom &b) const {
    if     (residueSequence < b.residueSequence) return true;
    else if(residueSequence > b.residueSequence) return false;
    else if(chainID < b.chainID)                 return true;
    else if(chainID > b.chainID)                 return false;
    else if(insCode < b.insCode)                 return true;
    else if(insCode > b.insCode)                 return false;
    else if(fullName.compare( b.fullName ) < 0)  return true;
    else                                         return false;
}

/**
 * @brief Atoms are considered equal if they have the same full name and residue id.
 **/
bool Atom::operator== (const Atom &b) const {
    return fullName == b.fullName
        && residueSequence == b.residueSequence
        && chainID         == b.chainID
        && insCode         == b.insCode;
}

Atom *Atom::parsePDBAtom( const string &line ) {
    Atom *atom = new Atom();
	Parser::FixedWidth fwp( line );

    fwp.parseLong(    atom->serialNumber,     6, 6 );
    fwp.parseString(  atom->name,            76, 2 );
    fwp.parseString(  atom->fullName,        12, 4 );
    fwp.parseChar(    atom->altLocation,     16    );
    fwp.parseString(  atom->residueName,     17, 3 );
    fwp.parseChar(    atom->chainID,         21    );
    fwp.parseLong(    atom->residueSequence, 22, 4 );
    fwp.parseChar(    atom->insCode,         26    );

    // Use element columns first; otherwise use first 2 characters of the full name 
	if( ! atom->nameIsIn( "C D H N O P S" )) fwp.parseString( atom->name, 12, 2 );
	// Throw away Hydrogen or Deuterium
	if( atom->nameIsIn( "H D" )) return (NULL);

    // Throw away alternate site (primary sites are denoted by a space, the
    // letter A, or the number 1).
	char alt = atom->altLocation;
    if ((alt != ' ') && (alt != 'A') && (alt != '1')) return (NULL);

    setResidueFullSequence( atom );

    fwp.parseDouble(  atom->coord.x,         30, 8 );
    fwp.parseDouble(  atom->coord.y,         38, 8 );
    fwp.parseDouble(  atom->coord.z,         46, 8 );
    fwp.parseDouble(  atom->occupancy,       54, 6 );
    fwp.parseDouble(  atom->bFactor,         60, 6 );

    atom->secondaryStructure = Coil;
    atom->solventAcc         = 0;
    atom->isHetatm           = false;
	atom->mark               = false;

    return atom;
}

void Atom::setResidueFullSequence(Atom *atom) {
	char buffer[80];
    sprintf(buffer, "%c%ld%c", atom->chainID, atom->residueSequence, atom->insCode);
	atom->residueID = ResidueID( atom );
	atom->residueFullSequence = buffer;
}

/**
 * @brief method to write an Atom
 **/
ostream &operator<<( ostream &stream, Atom &atom ) {
	stream 
		<< setw( 5 ) << setfill( ' ' ) << left <<                      "ATOM"
		<< setw( 7 ) << setfill( ' ' ) << left <<                      atom.serialNumber
		<< setw( 1 ) <<                                                atom.altLocation
		<< setw( 4 ) << setfill( ' ' ) << left <<                      atom.residueName
		<< setw( 1 ) <<                                                atom.chainID
		<< setw( 4 ) << setfill( ' ' ) << left <<                      atom.residueSequence
		<< setw( 1 ) <<                                                atom.insCode
		<<                                                             "   "
		<< setw( 8 ) << setprecision( 3 ) << setfill( ' ' ) << left << atom.coord.x
		<< setw( 8 ) << setprecision( 3 ) << setfill( ' ' ) << left << atom.coord.y
		<< setw( 8 ) << setprecision( 3 ) << setfill( ' ' ) << left << atom.coord.z
		<< setw( 6 ) << setprecision( 2 ) << setfill( ' ' ) << left << atom.occupancy
		<< setw( 6 ) << setprecision( 2 ) << setfill( ' ' ) << left << atom.bFactor;

	return stream;
}

/**
 * @brief accessor to return the Residue Label (e.g. CYS41:A)
 **/
string Atom::getResidueLabel() {
	stringstream residueLabel;

	residueLabel << residueName << residueSequence;
	if( insCode != ' ' ) residueLabel << insCode;
	residueLabel << ':';
    if( chainID != ' ' ) residueLabel << chainID;
    return residueLabel.str();
}

/**
 * @brief method to set secondary structure
 **/
void Atom::setSecondaryStructure(char structCode) {
    switch (tolower(structCode)) {
        case 'e': secondaryStructure = Strand; break;
        case 's': secondaryStructure = Bend;   break;
        case 't': secondaryStructure = Turn;   break;
        case 'b': secondaryStructure = Bridge; break;
        case 'g': secondaryStructure = Helix3; break;
        case 'h': secondaryStructure = Helix4; break;
        case 'i': secondaryStructure = Helix5; break;
        default:  secondaryStructure = Coil;   break;
    }
}

Atom::ResidueID::ResidueID() {
	residueSequence = 0;
	chainID         = '\0';
	insCode         = '\0';
}
Atom::ResidueID::ResidueID( long int _residueSequence, char _chainID, char _insCode ):
            residueSequence( _residueSequence ),
            chainID( _chainID ),
            insCode( _insCode ) {}

Atom::ResidueID::ResidueID( Atom *atom ) {
	residueSequence = atom->getResidueSequence();
	chainID         = atom->getChainID();
	insCode         = atom->getInsCode();
}

ostream &operator<<( ostream &stream, Atom::ResidueID &id ) {
	stream << id.residueSequence;
	if( id.insCode != ' ' ) stream << id.insCode;
	stream << ':';
    if( id.chainID != ' ' ) stream << id.chainID;

	return stream;
}


