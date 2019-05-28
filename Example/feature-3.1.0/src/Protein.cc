// FEATURE
// See COPYING and AUTHORS for copyright and license agreement terms.

// $Id: Protein.cc 1609 2013-12-10 08:03:27Z mikewong899 $

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "Protein.h"

#define DSSPHeader      "RESIDUE AA STRUCTURE"

#define MAX_NUM_MODELS  1

Protein::Protein() : modelNum(0) {
}

Protein::Protein( const string &filename ) : modelNum(0) {
	readPDBFile( filename );
}

Protein::~Protein() {
        for( Residues::iterator it = uniqueResidues.begin(); it != uniqueResidues.end(); it++ ) {
            delete it->second;
            it->second = NULL;
        }
		for( Atoms::iterator i = atoms.begin(); i != atoms.end(); i++ ) {
			delete (*i);
		}
        uniqueResidues.clear();
        hetAtoms.clear(); // @note atoms includes hetAtom instances and deletes both
}

/**
 * @brief returns the protein name as a string
 **/
const char *Protein::getName() {
	return name.c_str();
}

bool Protein::addAtomToUniqueResidues( Atom *atom ) {
	// Check for previously seen atoms
	// Is this a previously unseen, unique residue or part of a known residue?
	Atom::ResidueID    residueID            = atom->getResidueID();
	Residues::iterator residueSearchResults = uniqueResidues.find( residueID );
	// If the atom is part of a previously unseen residue, add a new unique
	// residue and add the atom to that residue
	if( residueSearchResults == uniqueResidues.end() ) {
		Atoms* residueAtoms = new Atoms();
		residueAtoms->addAtom( atom );
		uniqueResidues[ residueID ] = residueAtoms;

	// If the atom is part of a known residue, iterate through all atoms of the
	// residue to check for duplicate atoms. This is a tiny list, maybe 20
	// atoms. Duplicate atoms are deleted and not added to the list of atoms
	// belonging to the residue.
	} else {
		Atoms* residueAtoms = residueSearchResults->second;
		for( Atoms::iterator residueAtom = residueAtoms->begin(); 
                     residueAtom != residueAtoms->end(); 
                     residueAtom++ ) {
			if(*atom == **residueAtom) return false;
		}
		residueAtoms->addAtom( atom );
	}
	return true;
}

/**
 * @brief Parses an ATOM entry from a PDB and associates the atom with the
 * protein and a residue of the protein.
 * @param line A string of text to be parsed
 * @note Atoms may be non-unique and have multiple alternate locations. We only need one of the possible locations.
 **/
void Protein::parseAtom( string & line ) {
	Atom *atom = Atom::parsePDBAtom( line );
	if( ! atom )                            return; // Skip if the PDB line is not valid
	if( ! addAtomToUniqueResidues( atom ) ) return; // Skip atoms already seen

	atom->setIsHetatm( false );

	// Add the atom to the protein.
	atoms.addAtom( atom );
}

void Protein::parseHeader( string & line ) {
	Parser::FixedWidth fwp( line );
	fwp.parseString( name, 62 );
}

void Protein::parseHetatm( string & line ) {
	Atom *atom = Atom::parsePDBAtom( line );
	if ( ! atom )                            return; // Skip if the PDB line is not a valid
	if ( ! addAtomToUniqueResidues( atom ) ) { 
		delete atom;
		return;
	}

	atom->setSecondaryStructureToHetatm();
	atom->setIsHetatm( true );
	hetAtoms.addAtom( atom );
	atoms.addAtom( atom );
}

/**
 * @brief Parses a line from a PDB file
 * @param line A character string of PDB data to parse.
 **/
void Protein::parsePDBLine( string &line ) {
	string recordType;
	Parser::FixedWidth fwp( line );
	fwp.parseString(recordType, 0, 6);

	if ((modelNum < MAX_NUM_MODELS) ) {
		if(       recordType == "ATOM"   ) parseAtom( line );
		else if ( recordType == "HETATM" ) parseHetatm( line );
	}
	if (          recordType == "HEADER" ) parseHeader( line );
	else if (     recordType == "ENDMDL" ) modelNum++;
}

/**
 * @param filename The name of the PDB file to parse.
 *
 * PDB files describe primary structure (amino acid sequence) and tertiary
 * structure (3-D coordinates of atoms). Secondary structure (motifs such as
 * beta sheets and alpha helices) are described by DSSP files.
 **/
void Protein::readPDBFile( const string &filename ) {
	igzstream infile( filename.c_str() );
	string    line;

	if( ! infile ) fatal_error( "Error opening file '%s'\n", filename.c_str() );
	name = filename;

	while ( infile ) {
		getline( infile, line );
		parsePDBLine( line );
	}

	ForceFields *forceFields = ForceFields::getInstance();
	forceFields->setTypeAndCharge( this );
}

void Protein::parseDSSPLine( string & line ) {
	long int residueSequence;
	char insCode, chainID;

	// Protein residue specifiers
	if( line.size() < 5 ) return;
	Parser::FixedWidth fwp( line );
	fwp.parseLong( residueSequence, 5, 5 );
	fwp.parseChar( insCode,         10   );
	fwp.parseChar( chainID,         11   );

	long int solventAcc;
	char structureCode;

	// DSSP data
	fwp.parseChar( structureCode,   16   );
	fwp.parseLong( solventAcc,      34, 4);

	// Apply DSSP data for all atoms in the given residue
	Atom::ResidueID residueID( residueSequence, chainID, insCode );
	Residues::iterator residueSearchResults = uniqueResidues.find( residueID );

	if( residueSearchResults == uniqueResidues.end() ) return; // Can't find the residue; should this be an error?

	Atoms *residueAtoms = residueSearchResults->second;
	for( Atoms::iterator i = residueAtoms->begin(); i != residueAtoms->end(); i++ ) {
		Atom* atom = (*i);
		atom->setSolventAcc( solventAcc );
		atom->setSecondaryStructure( structureCode );
	}
}

/**
 * @param filename The name of the DSSP file to parse.
 *
 * DSSP files offer additional information, specifically secondary structure (e.g. 
 * peptide bond angles), partial charge effects, and other useful information.
 **/
void Protein::readDSSPFile( const string &filename ) {
	igzstream infile( filename.c_str() );
	string    line;

	if ( ! infile ) {
		error( "Cannot open file for reading '%s'\n", filename.c_str() );
		return;
	}

	// Eat up all the information prior to the secondary structure data
	while ( getline( infile, line ) ) 
		if( line.find( DSSPHeader ) != string::npos ) break;

	// Parse the DSSP File
	while ( getline( infile, line ) ) parseDSSPLine( line );
}

const string          Protein::ForceFields::defaultForceFieldParametersFilename  = "amberM2_params.dat";
const string          Protein::ForceFields::defaultResidueTemplateFilename       = "residue_templates.dat";
const float           Protein::ForceFields::disulfideBondLength                  = 2.2;
bool                  Protein::ForceFields::initialized                          = false;
Protein::ForceFields *Protein::ForceFields::instance                             = NULL;

Protein::ForceFields::ForceFields() {
	forceField              = NULL;
	forceFieldTerminus      = NULL;
	residueTemplate         = NULL;
	residueTemplateTerminus = NULL;
	nTerminalAtoms          = NULL;
	cTerminalAtoms          = NULL;
}

Protein::ForceFields *Protein::ForceFields::getInstance() {
	if( initialized ) return instance;

	instance = new ForceFields();
	instance->forceField      = new ForceField( defaultForceFieldParametersFilename );
	instance->residueTemplate = new ResidueTemplate( defaultResidueTemplateFilename );

	instance->nTerminalAtoms = new set<string>();
	instance->nTerminalAtoms->insert( "HT" );  instance->nTerminalAtoms->insert( "HT1" );
	instance->nTerminalAtoms->insert( "HT2" ); instance->nTerminalAtoms->insert( "HT3" );
	instance->nTerminalAtoms->insert( "H1" );  instance->nTerminalAtoms->insert( "H2" );
	instance->nTerminalAtoms->insert( "H3" );  instance->nTerminalAtoms->insert( "1H" );
	instance->nTerminalAtoms->insert( "2H" );  instance->nTerminalAtoms->insert( "3H" );
	instance->nTerminalAtoms->insert( "1HT" ); instance->nTerminalAtoms->insert( "2HT" );
	instance->nTerminalAtoms->insert( "3HT" );

	instance->cTerminalAtoms = new set<string>();
	instance->cTerminalAtoms->insert( "OXT" ); instance->cTerminalAtoms->insert( "O2" );
	instance->cTerminalAtoms->insert( "OT1" ); instance->cTerminalAtoms->insert( "OT2" );

	initialized = true;
	return instance;
}

void Protein::ForceFields::setTypeAndCharge( Protein *protein ) {
	Protein::Residues *residues = protein->getUniqueResidues();
	if (protein->size() <= 0) return;
	if (residues->empty())    return;

	Neighborhood *neighborhood = new Neighborhood( protein->getAtoms(), disulfideBondLength );
	for (Protein::Residues::iterator i = residues->begin(); i != residues->end(); i++ ) {
		Atoms  *residueAtoms = i->second;
		Atom   *atom         = NULL;
		Atom   *base         = residueAtoms->at( 0 );
		string  residueName  = base->getResidueName();

		if (base->getIsHetatm()) continue; /* ignore hetatms */

		/* Handle N-methyl from Insight II Molecular Modeler */
		if ( residueName == "N-M" ) {
			for (Atoms::iterator j = residueAtoms->begin(); j != residueAtoms->end(); j++ ) {
				atom = (*j);
				atom->setResidueName( "NME" );
			}
		}

		setTypeAndCharge( residueName, residueAtoms );

		if( residueName == "CYS" ) {
			setTypeAndChargeForDisulfideBridge( residues, residueAtoms, neighborhood );
		}
	}
	delete neighborhood;
}

/**
 * @brief Determines if the residue is at the N-terminus, C-terminus, or in the
 * middle of a polypeptide chain. Sets the state of the Protein instance to
 * perform type and charge lookups according to the residue position.
 *
 * This method should be called at least once per residue. 
 * 
 * Each position (N-terminus, C-terminus) has unique names for atoms. For each
 * atom in the residue, see if the atom's full name matches an entry on either
 * list. If the atom's full name matches an entry in the terminus atom name
 * list, then the entire residue must be at the relevant terminus.
 **/
void Protein::ForceFields::determinePosition( Atoms *atoms ) {

	/* Step 1. Get names of all the atoms in this residue */
	set<string> *names = new set<string>();
	for( Atoms::iterator i = atoms->begin(); i != atoms->end(); i++ ) {
		Atom *atom = (*i);
		names->insert( atom->getFullName() );
	}

	/* Step 2. Set template and force field lookups to Normal by default */
	residueTemplateTerminus = &residueTemplate->normal;
	forceFieldTerminus      = &forceField->normal;

	/* Step 3. Use the results from step 1 to check if this is N-terminal residue */
	if ( terminalAtomsMatchAny( nTerminalAtoms, names )) {
		residueTemplateTerminus = &residueTemplate->n_terminal;
		forceFieldTerminus      = &forceField->n_terminal;
	}

	/* Step 4. Use the results from step 1 to check if this is C-terminal residue */
	if ( terminalAtomsMatchAny( cTerminalAtoms, names )) {
		residueTemplateTerminus = &residueTemplate->c_terminal;
		forceFieldTerminus      = &forceField->c_terminal;
	}
	delete names;
}

bool Protein::ForceFields::terminalAtomsMatchAny( set<string> *terminalAtoms, set<string> *atomNames ) {
	set<string>::iterator i;
	for( i = terminalAtoms->begin(); i != terminalAtoms->end(); i++ ) {
		if (atomNames->find( *i ) != atomNames->end()) {
			return true;
		}
	}
	return false;
}

/**
 * @brief Sets the atom type and partial charge for each atom in the residue,
 * according to the residue name.
 *
 * @note The if() statement guards against a silent failure; consider making
 * this a fatal failure, since the partial charge and atom type properties will
 * be left empty (zero) if the residue name isn't found in the AMBER force
 * field parameters file.
 **/
void Protein::ForceFields::setTypeAndCharge( string residueName, Atoms *residueAtoms ) {
	determinePosition( residueAtoms );
	if (residueTemplateTerminus->find( residueName ) == residueTemplateTerminus->end()) return;

	ResidueTemplate::AtomAliases *aliases = (*residueTemplateTerminus)[ residueName ];
	for( Atoms::iterator i = residueAtoms->begin(); i != residueAtoms->end(); i++ ) {
		Atom* residueAtom = (*i);
		string fullName = residueAtom->getFullName();

		setTypeAndCharge( residueName, residueAtom );

		/* Convert alias to canonical name */
		if (aliases->find( fullName ) != aliases->end()) {
			string alias = (*aliases)[ fullName ];
			residueAtom->setFullName( alias );
		}
	}
}

/**
 * @brief Sets the atom type and partial charge for the given atom, according
 * to the given residue name. 
 *
 * @warning Only call this method after determinePosition() is called.
 *
 * @note The if() statement guards against a silent failure; consider making
 * this a fatal failure, since the partial charge and atom type properties will
 * be left empty (zero) if the atom ID isn't found in the AMBER force field
 * parameters file.
 **/
void Protein::ForceFields::setTypeAndCharge( string residueName, Atom *atom ) {
	string fullName = atom->getFullName();
	ForceField::AtomID atomID( residueName, fullName );
	if( forceFieldTerminus->find( atomID ) == forceFieldTerminus->end() ) return;

	ForceField::Parameters* parameters = (*forceFieldTerminus)[ atomID ];
	atom->setAtomType( parameters->type );
	atom->setPartialCharge( parameters->charge );
}

/**
 * @brief Searches the given Atoms for a gamma sulfur; used to find disulfide
 * bonds
 *
 * CYS residues that are part of disulfide bonds have different force field
 * partial charge characteristics than CYS residues not in disulfide bonds.
 * Therefore it's important to search for disulfide bonds and apply the
 * correct force field parameters.
 *
 * @note It is possible for a CYS residue in a PDB structure to not have a
 * gamma sulfur if the method for structure determination is of very poor
 * resolution or has errors.
 **/
Atom *Protein::ForceFields::findSulfurA( Atoms *cysA ) {
	Atom *sulfurA   = NULL;
	Atom *candidate = NULL;

	for (Atoms::iterator i = cysA->begin(); i != cysA->end(); i++) {
		candidate = (*i);
		if( candidate->getFullName() != "SG" ) continue;

		sulfurA = candidate;
		break;
	}
	return sulfurA;
}

/**
 * @brief Searches the given Atoms for a nearby gamma sulfur in a different CYS
 * than sulfur A; used to find disulfide bonds
 *
 * Used in conjunction with the findSulfurA() method. See the findSulfurA()
 * method for more details.
 **/
Atom *Protein::ForceFields::findSulfurB( Atoms *neighbors, Atom *sulfurA ) {
	Atom *sulfurB   = NULL;
	Atom *candidate = NULL;
	for (Atoms::iterator i = neighbors->begin(); i != neighbors->end(); i++) {
		candidate = (*i);
		if( candidate->getFullName() != "SG" )                        continue; // Must be a sulfur
		if( candidate->getResidueID() == sulfurA->getResidueID() )    continue; // Can't be in CYS A
		if( candidate->getDistance( sulfurA ) > disulfideBondLength ) continue; // Must be close to sulfur A

		sulfurB = candidate;
		break;
	}
	return sulfurB;
}

/**
 * @brief Determines if the CYS is part of a disulfide bridge; if so, this
 * method sets the type and charge accordingly.
 *
 * CYX is the AMBER residue name for a CYS involved in a disulfide bridge. The
 * atom types and charges are significantly different than CYS because the
 * electron distribution is shifted towards the covalent SS-bridge bond,
 * causing the partial charges for CYX atoms to tend towards positive.
 *
 * @note The two CYS residues in a disulfide bond are labeled as A and B,
 * respectively.
 *
 * @note It's possible to have a protein structure with a missing CYS SG;
 * especially for bad resolution NMR or X-ray approaches (bad PDB data).
 **/
void Protein::ForceFields::setTypeAndChargeForDisulfideBridge( Residues *residues, Atoms *cysA, Neighborhood *neighborhood ) {

	Atom *sulfurA = findSulfurA( cysA );
	if( sulfurA == NULL ) return;

	/* Find the sulfur atom for CYS B residue among the neighbors of sulfur A */
	Atoms *neighbors = neighborhood->getNeighbors( sulfurA );
	Atom  *sulfurB   = findSulfurB( neighbors, sulfurA );
	if( sulfurB == NULL ) return;

	/* Find the other CYS that this is bonded to */
	Atom::ResidueID id = sulfurB->getResidueID();
	if( residues->find( id ) == residues->end() ) {
		fatal_error( "Protein::ForceFields: Failed to find other CYS in disulfide bridge" );
	}
	Atoms *cysB = (*residues)[ id ];

	/* This is a confirmed SS Bond; set type and charge for both residues */
	setTypeAndCharge( "CYX", cysA );
	setTypeAndCharge( "CYX", cysB );
}
