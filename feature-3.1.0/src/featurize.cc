// $Id: featurize.cc 1174 2011-10-28 00:56:50Z mikewong899 $
// Copyright (c) 2004 Mike Liang. All rights reserved.

/**
 * \page featurize featurize
 *
 * Generates feature vectors at given Cartesian points in or near one or more proteins
 *
 * \htmlinclude featurize_concept.html
 *
 * \section featurize_input Input
 * 
 * - An optional \ref point_file "Point File" and
 * - One or more \ref pdb "PDB files" and their related \ref dssp "DSSP files". A PDB file contains protein structure and function data. A DSSP file contains protein secondary structure data; it is derived from a PDB file.
 * 
 * You have three options for specifying the \ref pdb "PDB" and related \ref dssp "DSSP" files:
 * 
 * -# Provide one or more PDB IDs as a command-line argument.<br><tt>$ featurize 1bqy [...]</tt><br><br>
 * -# Specify one or more PDB IDs in a text file, using the \c -f option.<br><tt>$ featurize -f pdb_files.txt</tt><br><br>
 * -# Specify a \ref point_file "Point File" wth the <tt>-P</tt> option. A point file specifies a set of points for one or more PDB IDs.<br><tt>$ featurize -P sites.ptf</tt>
 *
 * \c featurize will look for the \ref pdb "PDB" and \ref dssp "DSSP" files in three locations:
 * the current working directory, and the paths specified by the environment variables
 * <tt>PDB_DIR</tt> and <tt>DSSP_DIR</tt>. These environment variables are set when
 * FEATURE is installed; to change them, see 
 * \ref featurize_environment_variables "Environment Variables"
 *
 * \section featurize_output Output
 * \li Prints a \ref feature_vector "feature vector"
 *
 * \section featurize_usage Usage
 * \code
 * Usage: featurize [OPTIONS] [PDBID...]
 * 
 * Must exclude .ent or other filename extension in PDBID
 * 
 * Options:
 *     -v  Increase verbosity
 *     -n SHELLS [Default: 6]
 *         Set number of shells to NUMSHELLS
 *     -w SHELL_WIDTH [Default: 1.25]
 *         Set thickness of each shell to SHELL_WIDTH Angstroms
 *     -x EXCLUDED_RESIDUES [Default: HETATM]
 *         Set residues to exclude to comma separated list of
 *         EXCLUDED_RESIDUES
 *     -f PDBID_FILE
 *         Read PDBIDs from PDBID_FILE
 *     -P POINT_FILE
 *         Read point list from POINT_FILE
 *     -l SELECTABLE_PROPERTIES_FILE
 *         Read a property list from SELECTABLE_PROPERTIES_FILE
 * \endcode
 *
 * \htmlinclude feature_vector_generation.html
 *
 * \subsection featurize_options_detail Options Detail
 *
 * The verbosity option (<tt>-v</tt>) increases the number of output 
 * messages, some of which are useful for understanding how \c featurize 
 * works and for debugging.
 *
 * The \c NUMSHELLS variable controls the number of radial shells \c featurize 
 * will evaluate. Larger values will create more data and take longer
 * to compute; however the data may be noisy. The default value is strongly
 * recommended.
 *
 * The \c SHELLWIDTH variable controls the size of the shells that
 * \c featurize will evaluate. Again, larger values will create more data,
 * but may contribute noise. The default value is strongly recommended.
 *
 * The \c EXCLUDERESIDUES option excludes one type of PDB-specified
 * atoms. For a list of PDB-specified atom types, see \ref pdb "PDB" files.
 * The default value is strongly recommended.
 * 
 * \c PDBFILE is either a filename or \c - (standard input). If the 
 * user provides a filename, \c featurize will read each PDB ID listed in the 
 * file and produce a feature vector for each corresponding protein at each 
 * atom location.
 *
 * If the user specifies standard input then the user may type in PDB IDs and 
 * type <tt>Control-D</tt> to begin analyzing the specified proteins. Both 
 * methods are useful for producing feature vectors for unknown proteins to 
 * be classified.
 *
 * Optionally the user can use \c featurize with one or more PDBID parameters.
 * This causes \c featurize to produce a feature vector for each corresponding
 * protein at each atom location.
 *
 * \c POINTFILE is a filename of a \ref point_file "Point file". 
 * This option is useful for creating a feature vector for sites and non-sites.
 * The point file for similar sites can be manually selected or programatically
 * extracted from databases such as ProSite. The datamining approach is left to
 * the user; FEATURE does not provide this service.
 *
 * \c SELECTABLE_PROPERTIES_FILE is a filename of a 
 * \ref selectable_properties_file "Selectable Properties file". As of FEATURE 3.0, a user
 * can specify which properties they want to use for their analysis. Two
 * property sets are provided with FEATURE 3.0: \c proteins.properties and
 * \c metals.properties. The former file has the standard 80 properties for
 * FEATURE. The latter file has 44 properties used in another version of
 * FEATURE, called FEATURE for metal ion Ligands. Users are encouraged to copy
 * these files or even concatenate and edit them to build a customized list of
 * properties. See \ref physicochemical_properties properties for more information
 * about the properties themselves.
 *
 * \section featurize_environment_variables Environment Variables
 *
 * The \c PDB_DIR and \c DSSP_DIR environment variables control where
 * \c featurize will look for \ref pdb "pdb" and \ref dssp "DSSP" files. You can change 
 * them as follows:
 * 
 * - Mac OS X:<br><tt>$ sudo open /etc/bashrc</tt><br><tt>$ . /etc/bashrc</tt><br><br>
 * - Linux:<br><tt>$ sudo vim /etc/bashrc</tt><br><tt>$ . /etc/bashrc</tt><br><br>
 * - Windows:<br><b>Start -&gt; Settings -&gt; Control Panel -&gt; System -&gt; Advanced -&gt; %Environment Variables</b><br>Find \c PDB_DIR and \c DSSP_DIR under <tt>System variables</tt>; click on them and then click on <tt>Edit</tt>; change the path and click <tt>OK</tt>.<br>
 *
 * \subsection featurize_examples Examples
 *
 * \c featurize can be used in several ways: To characterize the microenvironments of similar (or non-similar) active sites across several proteins
 * \code
 *  featurize -P trypsin_ser_og.pos.ptf > trypsin_ser_og.pos.ff
 * \endcode
 *
 * To characterize the microenvironments at all points within a protein
 *
 * \code
 *  featurize 1bqy > 1bqy.ff
 * \endcode
 *
 * To characterize the microenvironments for selected points within a protein
 * using the atomselector.py utility. In this case, the gamma oxygens for all 
 * serine residues in the structure 1bqy.
 *
 * \code
 *  atomselector.py -r SER -a OG 1bqy > 1bqy_ser_og.ptf
 *  featurize -P 1bqy_ser_og.ptf > 1bqy_ser_og.ff
 * \endcode
 *
 * \section selectable_properties Selectable Properties
 * Users can select which physicochemical properties to use for microenvironment
 * characterization. This is done by providing \c featurize with a 
 * \ref selectable_properties_file "Selectable Properties File"
 *
 **/

#include <cstdio>
#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include <exception>
#include <map>
#include "Protein.h"
#include "Atoms.h"
#include "Neighborhood.h"
#include "PropertyList.h"
#include "Atom.h"
#include "AtomProperties.h"
#include "stl_types.h"
#include "Utilities.h"
#include "logutils.h"
#include "Points.h"
#include "Environments.h"
#include "config.h"
#include "MetaData.h"

#define COMMA_SEP ", \t"
#define BUF_SIZE 1024
#define X(a) (a->coord.x)
#define Y(a) (a->coord.y)
#define Z(a) (a->coord.z)
#define SQR(v) ((v)*(v))
#define SQRDIST(a,b) (SQR(X(a)-X(b))+SQR(Y(a)-Y(b))+SQR(Z(a)-Z(b)))

using std::cin;
using std::endl;
using std::exception;
using std::ifstream;
using std::istream;
using std::map;
using std::string;

int    DEFAULT_VERBOSITY          = 0;
int    DEFAULT_NUMSHELLS          = 6;
double DEFAULT_SHELLWIDTH         = 1.25;
char   DEFAULT_EXCLUDE_RESIDUES[] = "HETATM";
static const string progname      = "featurize";

extern char *optarg;
extern int optind;

typedef map< Atom *, AtomProperties *> PropertiesCache;

void eusage() {
    printf("\n%s (build %s) - %s\n\n", PACKAGE_STRING, SVN_REV, progname.c_str());
    printf("Usage: %s [OPTIONS] [PDBID...]\n", progname.c_str());
    printf("\n");
    printf("Must exclude .ent or other filename extension in PDBID\n");
    printf("\n");
    printf("Options:\n");
    printf("    -v  Increase verbosity\n");
    printf("    -n SHELLS [Default: %d]\n", DEFAULT_NUMSHELLS);
    printf("        Set number of shells to NUMSHELLS\n");
    printf("    -w SHELL_WIDTH [Default: %g]\n", DEFAULT_SHELLWIDTH);
    printf("        Set thickness of each shell to SHELL_WIDTH Angstroms\n");
    printf("    -x EXCLUDED_RESIDUES [Default: %s]\n", DEFAULT_EXCLUDE_RESIDUES);
    printf("        Set residues to exclude to comma separated list of\n");
    printf("        EXCLUDED_RESIDUES\n");
    printf("    -f PDBID_FILE\n");
    printf("        Read PDBIDs from PDBIDFILE\n");
    printf("    -P POINT_FILE\n");
    printf("        Read point list from POINTFILE\n");
    printf("    -l PROPERTIES_FILE\n");
    printf("        Read a property list from PROPERTIES_FILE\n");
    printf("    -s SEARCH_PATH\n");
    printf("        Look for protein files in SEARCH_PATH first before system directories\n");

    exit(-1);
}

void printAtomPropertiesHeader(FILE *outs) {
    for(
		vector<Property *>::iterator property = PropertyList::begin();
		property < PropertyList::end();
		property++
	) {
        fprintf(outs, "\t%s", (*property)->getName().c_str());
    }
}

void printAtomDescriptionHeader(FILE *outs) {
    fprintf(outs, "\tCOORD_X");
    fprintf(outs, "\tCOORD_Y");
    fprintf(outs, "\tCOORD_Z");
    fprintf(outs, "\tRESIDUE_NAME");
    fprintf(outs, "\tRESIDUE_SEQ");
    fprintf(outs, "\tINSERTION_CODE");
    fprintf(outs, "\tCHAIN_ID");
    fprintf(outs, "\tFULL_NAME");
    fprintf(outs, "\tALT_LOC");
    fprintf(outs, "\tATOM_SERIAL");
}

Atoms * getNonExcludedResidues(Protein *protein, Strings &list) {
	Atoms *atoms = protein->getAtoms();
    Atoms *nonExcludedResidues;
    string residueName;
    Strings::const_iterator iter;

	atoms->clearAtomMarks();

	// ===== MARK EXCLUDED ATOMS
    for (iter = list.begin(); iter != list.end(); iter++) {
        residueName = *iter;
        if ( residueName == "HETATM" ) {
            MarkCriteria::Hetatm thatAreHetatms;
			atoms->markAtoms( thatAreHetatms );

        } else {
			MarkCriteria::ResidueName thatHaveResidueName( residueName );
			atoms->markAtoms( thatHaveResidueName );
        }
    }

	// ===== TOGGLE MARKS TO GET NON-EXCLUDED ATOMS
	atoms->invertAtomMarks();
    nonExcludedResidues = atoms->extractMarkedAtoms();
	atoms->clearAtomMarks();
    return nonExcludedResidues;
}

void loadPdbidsFromFile(char *filename, Strings &pdbids) {
    istream *ins = NULL;
    if (strcmp(filename, "-") == 0)
        ins = &cin;
    else
        ins = new ifstream(filename);
    if (!(ins && *ins)) {
        warning("Could not open pdbidfile %s\n", filename);
        return;
    }

    string pdbid;
    while ((*ins) >> pdbid) {
        pdbids.append( pdbid );
    }
}

void loadProteinStructure( char *searchPath, const string &pdbid, Protein **protein ) {
	const string search = searchPath == NULL ? "" : searchPath;
	string dsspid       = getPdbIdFromFilename( pdbid );
	string pdbFilename  = findPdbFile( pdbid, search );
	bool   needDSSP     = false;

	debug(1, "Found pdb: %s\n", pdbFilename.c_str());
	(*protein)          = new Protein( pdbFilename );
	debug(1, "LOADED pdb: %s\n", pdbid.c_str());

	// Iterate through all properties to see if any rely on DSSP file
	for( 
		vector<Property *>::iterator property = PropertyList::begin();
		property < PropertyList::end();
		property++
	) {
		if( (*property)->getSource() == Property::DSSP ) needDSSP = true;
	}

	if( ! needDSSP ) return;

	string dsspFilename = findDsspFile( dsspid, search );
	debug(1, "Found dssp: %s\n", dsspFilename.c_str());
	(*protein)->readDSSPFile( dsspFilename );
	debug(1, "LOADED dssp: %s\n", dsspid.c_str());
}

Points *getPoints( Protein *protein ) {
    Points *points = new Points;
    for (Atoms::iterator i = protein->begin(); i != protein->end(); i++ ) {
		Atom *atom = (*i);
		Point3D *coord;
		coord = atom->getCoordinates();
    	string label = "#\t" + atom->getResidueLabel() + "@" + atom->getFullName();;
        Point *point = new Point( *coord );
        point->setDescription( label );
        points->append( point );
    }
    return points;
}

/**
 * @brief Namespace class for all FEATURE command-line programs (binary exectuables)
 **/
class Application {
	public:
/**
 * \brief The heart of the \ref featurize "featurize" application, this class generates a @ref feature_vector "Feature Vector File"
 * \ingroup featurize_application
 **/
class Featurize {
public:
    int        verbosity;
    int        numShells;
    double     shellWidth;
    Strings    excludeResidues;
    Strings    pdbids;
    PointFile *pointFile;
    char      *propertyFile;
    char      *searchPath;
	Metadata  *metadata;

    Featurize() {
        verbosity    = DEFAULT_VERBOSITY;
        numShells    = DEFAULT_NUMSHELLS;
        shellWidth   = DEFAULT_SHELLWIDTH;
        pointFile    = NULL;
        propertyFile = NULL;
        metadata     = new Metadata( "featurize" );
        searchPath   = NULL;

		Environments::enableMetadataKeywords( metadata );
    }

    ~Featurize() {
		excludeResidues.clear();
		pdbids.clear();
		if( pointFile != NULL ) delete pointFile;
    }

    void parseArguments(int argc, char *argv[]) {
        int setExcludeResiduesFlag = 0;
        int opt;
        while ((opt = getopt(argc, argv, ":vn:w:x:f:P:l:s:")) != -1) {
            switch (opt) {
                case 'v':
                    setdebuglevel(++verbosity);
					metadata->set( "VERBOSITY", verbosity );
                    break;
                case 'n':
                    numShells = atoi(optarg);
					metadata->set( "SHELLS", numShells );
                    break;
                case 'w':
                    shellWidth = atof(optarg);
					metadata->set( "SHELL_WIDTH", shellWidth );
                    break;
                case 'x':
                    excludeResidues.extend( optarg, COMMA_SEP );
                    setExcludeResiduesFlag = 1;
					metadata->set( "EXCLUDED_RESIDUES", optarg );
                    break;
                case 'f':
                    loadPdbidsFromFile(optarg, pdbids);
					metadata->set( "PDBID_FILE", optarg );
                    break;
                case 'P':
                    pointFile = new PointFile( optarg );
					metadata->set( "POINT_FILE", optarg );
                    break;
                case 'l':
                    propertyFile = optarg;
					metadata->set( "PROPERTIES_FILE", propertyFile );
                    break;
                case 's':
                    searchPath = optarg;
					metadata->set( "SEARCH_PATH", searchPath );
                    break;
                default:
                    error("ERROR: option %c not defined\n", opt);
                    eusage();
                    break;
            }
        }

        if( propertyFile ) {
			PropertyList::load( propertyFile );
			PropertyList::set( metadata );
        } else if( metadata->hasProperties() ) {
            PropertyList::load( metadata );
        }
        while (optind != argc) {
            pdbids.append( argv[optind] );
            optind++;
        }
        if (pdbids.size() == 0 && pointFile) {
            vector< string > *ids = pointFile->getPdbids();
            for( vector< string >::iterator id = ids->begin(); id != ids->end(); id++ ) {
                pdbids.append( *id );
            }
            delete ids;
        }
        if (pdbids.size() == 0) {
            error( "ERROR: No PDB IDs requested\n" );
            eusage();
        }
		string pdbidsList;
		for( Strings::iterator i = pdbids.begin(); i != pdbids.end(); i++ ) {
			if( i == pdbids.begin() ) {
				pdbidsList += (*i);
			} else {
				pdbidsList += ", " + (*i);
			}
		}
		metadata->set( "PDBID_LIST", pdbidsList );
        if (!setExcludeResiduesFlag)
            excludeResidues.extend( DEFAULT_EXCLUDE_RESIDUES, COMMA_SEP );
    }

	void calculateEnvironment( Point *point, Environment::Shell *shell, Neighborhood *neighborhood, PropertiesCache &atomPropertiesCache ) {
		// ===== GET POINT NEIGHBORS
		Atoms *neighbors = neighborhood->getNeighbors( point->getCoordinates() );

		// ===== FOR EACH NEIGHBOR
		for (Atoms::iterator idx = neighbors->begin(); idx != neighbors->end(); idx++ ) {
			Atom *neighborAtom  = (*idx);

			// ===== CALCULATE ATOM PROPERTIES
			AtomProperties *atomProps;
			if( atomPropertiesCache.find( neighborAtom ) == atomPropertiesCache.end() ) {
				atomProps = new AtomProperties( neighborAtom );
				atomPropertiesCache[ neighborAtom ] = atomProps;
			} else {
				atomProps = atomPropertiesCache[ neighborAtom ];
			}

			// ===== CALCULATE SHELL
			Point3D *coord          = neighborAtom->getCoordinates();
			double neighborDistance = point->getCoordinates().getDistance( *coord );
			int shellAffected       = (int) floor( neighborDistance / shellWidth );

			// ===== IGNORE POINTS TOO FAR AWAY
			if (shellAffected >= numShells) {
				continue;
			}

			// ===== UPDATE PROPERTY
			shell[ shellAffected ].include( atomProps );
		}

		delete neighbors;
	}

	void dumpParameters(int lvl) {
		debug( lvl, "FEATURE_DIR: %s\n", getEnvironmentVariable( FeatureDirEnvStr ).c_str());
		debug( lvl, "verbosity: %d\n", verbosity );
		debug( lvl, "numShells: %d\n", numShells );
		debug( lvl, "shellWidth: %g\n", shellWidth );
		debug( lvl, "excludeResidues: %d [ %s ]\n", excludeResidues.size(), toString(excludeResidues) );
		debug( lvl, "pdbids: %d [ %s ]\n", pdbids.size(), toString(pdbids) );
		if(searchPath) {
			debug( lvl, "search path: %s\n", searchPath); 
		}
	}

	void getPointsFromProtein( const string &pdbid, Protein **protein, Neighborhood **neighborhood, Points **points ) {
		// ===== LOAD THE STRUCTURE
		loadProteinStructure( searchPath, pdbid, protein );

		// ===== FILTER OUT USER-DEFINED EXCLUDED RESIDUES
		Atoms *atoms = getNonExcludedResidues( (*protein), excludeResidues );

		// ===== CREATE NEIGHBORHOOD
		(*neighborhood) = new Neighborhood( atoms, numShells * shellWidth );

		// ===== GET POINTS FROM PDB OR FROM POINT FILE
		if (pointFile == NULL) {
			(*points) = getPoints( (*protein) );
		} else {
			(*points) = pointFile->getPoints( pdbid );
		}

		if( (*points) == NULL ) {
			warning("No points for %s\n", pdbid.c_str() );
			delete protein;
			delete neighborhood;
		}
	}


    int runPointEnvironmentMode() {
        // ===== FOR EACH PROTEIN STRUCTURE
		metadata->write();
        for( Strings::iterator pdbid = pdbids.begin(); pdbid != pdbids.end(); pdbid++ ) {
            Protein      *protein      = NULL;
            Neighborhood *neighborhood = NULL;
            Points       *points       = NULL;
            getPointsFromProtein( *pdbid, &protein, &neighborhood, &points );
            if( points == NULL ) continue;

            // ===== FOR EACH POINT
            PropertiesCache atomPropertiesCache;
            int numPoints = points->numPoints();
            for ( int pointIndex = 0; pointIndex < numPoints; pointIndex++ ) {
                Point              *point = points->getPoint( pointIndex );
                Environment::Shell *shell = new Environment::Shell[ numShells ];

                // ===== CALCULATE THE ENVIRONMENT
                calculateEnvironment( point, shell, neighborhood, atomPropertiesCache );

                // ===== PRINT THIS ENVIRONMENT
                fprintf( stdout, "Env_%s_%d", getPdbIdFromFilename( *pdbid ).c_str(), pointIndex );
                for ( int shellIndex = 0; shellIndex < numShells; shellIndex++ ) {
                    shell[ shellIndex ].write( stdout );
                }
                fprintf(stdout, "\t#");
                fprintf(stdout, "\t");
                point->write( stdout );
                fprintf(stdout, "\n");

                delete [] shell;

            }

			for( PropertiesCache::iterator i = atomPropertiesCache.begin(); i != atomPropertiesCache.end(); i++ ) {
				delete i->second;
			}
            if (pointFile == NULL) delete points;
            delete neighborhood;
            delete protein;
        }

        return 0;
    }
    
    void main(int argc, char *argv[]) {
        parseArguments(argc, argv);
        dumpParameters( verbosity );
        runPointEnvironmentMode();
    }
};
};

int main(int argc, char *argv[]) {
    try {
        Application::Featurize featurize;
        featurize.main(argc, argv);
        return 0;
    } catch (const char *msg) {
        error("Exception message: %s\n", msg);
    } catch (const int ec) {
        error("Exception error code: %d\n", ec);
    } catch ( exception &e ) {
        error("Exception description: %s\n", e.what() );
    } catch (...) {
        error("Unknown Exception occurred\n");
    }
    return -1;
}
