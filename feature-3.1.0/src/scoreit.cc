// $Id: scoreit.cc 1285 2012-03-22 01:12:42Z mikewong899 $

/**
 * \page scoreit scoreit
 *
 * %Scores a set of given points in an unknown protein for similarity to a given active site model
 *
 * \section scoreit_input Input
 * \li A \ref model "model file" that contains a protein function model, as produced by \ref buildmodel 
 * \li A \ref feature_vector "feature vector file" that describes one or more points in or near an unknown protein, as produced by \ref featurize
 *
 * \section scoreit_output Output
 * \li A \ref score_file "score file" that contains a probability score for 
 * each point described by the \ref feature_vector "feature vector file" that
 * the given point may have the same function as described by the model
 *
 * \section scoreit_usage Usage
\verbatim
Usage: scoreit [OPTIONS] MODEL_FILE FEATURE_VECTOR_FILE

Options:
    -v  Increase verbosity
    -n NUMSHELLS [Default: 6]
        Set number of shells
    -p PLEVEL [Default: 0.01]
        Set minimum acceptable p-Value; ignore contributions from scores with lower p-Values
    -c CUTOFF [Default: 0]
        Set minimum score to print out
    -l PROPERTIESFILE
        Read a property list from PROPERTIESFILE
    -a  Shows all scores
    -e  Toggle count empty shells [Default: 1]
\endverbatim
 *
 * \subsection scoreit_options_detail Options Detail
 *
 * The verbosity option (<tt>-v</tt>) increases the number of output messages, 
 * some of which are useful for understanding how \c scoreit works and for 
 * debugging.
 *
 * The \c NUMSHELLS variable controls number of shells \c scoreit 
 * will evaluate. This variable should be consistent with the number of
 * shells used to build the model and generate the feature vectors.
 * The default value is strongly recommended.
 *
 * The \c PLEVEL variable controls the maximum value for the
 * Mann-Whitney-Wilcoxon (MWW) Rank Sum rho score that will be considered.
 * A MWW Rank Sum rho score of 0.5 indicates that there is no difference
 * between the distributions between the positive and negative training
 * data for a given property. Therefore the property is likely to be
 * unimportant. A MWW Rank Sum rho score that approaches zero indicates
 * that the property distribution has strong correlation to a classification.
 * The default value is strongly recommended.
 *
 * The \c CUTOFF variable controls the minimum score to print out;
 * because scores are relative, the \c -a option is recommended.
 *
 * The \c -e option directs <tt>scoreit</tt> to ignore empty shells by default.
 * The default value is recommended.
 * 
 * \section scoreit_examples Examples
 * The following example will show only significant hits (positive scores)
 * for the protein structure 2zpb against the KAZAL.3.CYS.SG model.
 * \code
 * featurize 2zpb > 2zpb.ff
 * scoreit KAZAL.3.CYS.SG.model 2zpb.ff > 2zpb.hits
 * \endcode
 *
 * The following example will show all significant hits (all scores) for a
 * subset of the protein structure 1bqy (namely the gamma oxygens for all
 * serine residues) against the trypsin_ser_og model.
 * \code
 * atomselector.py -r SER -a OG 1bqy > 1bqy_ser_og.ptf
 * featurize -P 1bqy_ser_og.ptf > 1bqy_ser_og.ff
 * scoreit -a trypsin_ser_og.model 1bqy_ser_og.ff > 1bqy_ser_og.hits
 * \endcode
 **/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <zlib.h>
#include "gzstream.h"
#include "Utilities.h"
#include "Environments.h"
#include "Model.h"
#include "config.h"

#define DEFAULT_VERBOSITY       0
#define DEFAULT_NUM_SHELLS      6
#define DEFAULT_P_LEVEL         0.01
#define DEFAULT_SCORE_CUTOFF    0.0
#define DEFAULT_NO_CUTOFF       0
#define DEFAULT_COUNT_EMPTY     1

#define BUF_SIZE 8192

using std::string;

static const string progname = "scoreit";

extern char *optarg;
extern int optind;

class Application {
public:
/**
 * @brief The heart of the @ref scoreit "scoreit" application, this class generates a FEATURE @ref score_file "Score File"
 * @ingroup scoreit_application
 **/
class Scoreit {
	public:
		int               verbosity;
		int               numShells;
		double            pLevel;
		double            scoreCutoff;
		int               noCutoff;
		int               countEmpty;
		char             *propertyFile;
		char             *modelFilename;
		char             *environmentFilename;
		Metadata         *metadata;
		Metadata         *environmentMetadata;

	Scoreit() {
		verbosity    = DEFAULT_VERBOSITY;
		numShells    = DEFAULT_NUM_SHELLS;
		pLevel       = DEFAULT_P_LEVEL;
		scoreCutoff  = DEFAULT_SCORE_CUTOFF;
		noCutoff     = DEFAULT_NO_CUTOFF;
		countEmpty   = DEFAULT_COUNT_EMPTY;
		propertyFile = NULL;

		metadata = new Metadata( "scoreit" );

		metadata->enable( "ALL_SCORES" );
		metadata->enable( "CUTOFF" );
		metadata->enable( "EXCLUDED_RESIDUES" );
		metadata->enable( "FEATURE_VECTOR_FILE" );
		metadata->enable( "IGNORE_EMPTY_SHELLS" );
		metadata->enable( "MODEL_FILE" );
		metadata->enable( "P_LEVEL" );
		metadata->enable( "PROPERTIES" );
		metadata->enable( "PROPERTIES_FILE" );
		metadata->enable( "SHELLS" );
		metadata->enable( "SHELL_WIDTH" );
		metadata->enable( "VERBOSITY" );

		metadata->set( "ALL_SCORES",           noCutoff );
		metadata->set( "CUTOFF",               scoreCutoff );
		metadata->set( "IGNORE_EMPTY_SHELLS", !countEmpty );
		metadata->set( "P_LEVEL",              pLevel );
		metadata->set( "SHELLS",               numShells );
		metadata->set( "VERBOSITY",            verbosity );
	};

	void eusage() {
		printf( "\n%s (build %s) - %s\n\n", PACKAGE_STRING, SVN_REV, progname.c_str() );
		printf("Usage: %s [OPTIONS] MODELFILE FEATURES\n", progname.c_str());
		printf("\n");
		printf("Options:\n");
		printf("    -v  Increase verbosity\n");
		printf("    -n NUMSHELLS [Default: %d]\n", DEFAULT_NUM_SHELLS);
		printf("        Set number of shells\n");
		printf("    -p PLEVEL [Default: %g]\n", DEFAULT_P_LEVEL);
		printf("        Set p-level of scorefile\n");
		printf("    -c CUTOFF [Default: %g]\n", DEFAULT_SCORE_CUTOFF);
		printf("        Set minimum score to print out\n");
		printf("    -l PROPERTIESFILE\n");
		printf("        Read a property list from PROPERTIESFILE\n");
		printf("    -a  Shows all scores\n");
		printf("    -e  Toggle count empty shells [Default: %d]\n", DEFAULT_COUNT_EMPTY);
		exit(-1);
	}

	/**
	 * @brief Calculates the score for an environment based on a given model.
	 * Higher scores mean higher likelihood that the environment may have the
	 * functionality described by the model.
	 *
	 * @param environment An Environment (aka feature vector) for a given point in a Protein structure
	 * @param model A protein function Model
	 * @param countEmpty If false then ignore shells with no atoms (empty shells); otherwise tally empty shells.
	 *
	 * The NaiveBayesClassifier clusters every value for each property into bins.
	 * The bin scores are the proportional log probabilities for each of these
	 * bins. The bin scores may be summed over each Property/Environment::Shell
	 * combination to get a total score for the given Environment.
	 * 
	 **/
	double calculateScore( Environment *environment, Model *model, bool countEmpty ) {
		int    numShells     = environment->getNumShells();
		int    numProperties = PropertyList::size();
		double score         = 0.0;

		for( int shellIndex = 0; shellIndex < numShells; shellIndex++ ) {
			Environment::Shell *shell = environment->getShell( shellIndex );
			if( ! shell->hasAtoms() && ! countEmpty ) continue;

			for( int propertyIndex = 0; propertyIndex < numProperties; propertyIndex++ ) {
				Property *property     = PropertyList::get( propertyIndex );
				string    propertyName = property->getName();
				double    value        = shell->getValue( propertyIndex );

				score += model->getBinScore( propertyName, shellIndex, value );
			}
		}
		return score;
	}

	void parseArguments( int argc, char *argv[] ) {
		// Command line arguments
		// ----------------------
		int opt;
		while ((opt = getopt(argc, argv, ":vn:p:c:al:e")) != -1) {
			switch (opt) {
				case 'v':
					setdebuglevel(++verbosity);
					break;
				case 'n':
					numShells = atoi(optarg);
					metadata->set( "SHELLS", numShells );
					break;
				case 'p':
					pLevel = atof(optarg);
					metadata->set( "P_LEVEL", pLevel );
					break;
				case 'c':
					scoreCutoff = atof(optarg);
					metadata->set( "CUTOFF", scoreCutoff );
					break;
				case 'a':
					noCutoff = 1;
					metadata->set( "ALL_SCORES", 1 );
					break;
				case 'l':
					propertyFile = optarg;
					metadata->set( "PROPERTIES_FILE", propertyFile );
					break;
				case 'e':
					countEmpty = ! countEmpty;
					metadata->set( "IGNORE_EMPTY_SHELLS", 1 );
					break;
				default:
					eusage();
					break;
			}
		}
		char **args = argv + optind;
		int numargs = argc - optind;
		if (numargs != 2) {
			eusage();
		}
		modelFilename       = args[0];
		environmentFilename = args[1];
		metadata->set( "MODEL_FILE",          modelFilename );
		metadata->set( "FEATURE_VECTOR_FILE", environmentFilename );
	}

	int main( int argc, char *argv[] ) {
		setdebuglevel(verbosity);
		parseArguments( argc, argv );

		if( propertyFile ) { 
			PropertyList::load( propertyFile );
			PropertyList::set( metadata );
		}

		if (! fileExists( environmentFilename )) fatal_error( "Can't open Feature Vector file '%s' for reading.\n", environmentFilename );
		if (! fileExists( modelFilename ))       fatal_error( "Can't open Model file '%s' for reading.\n", modelFilename );

		// Load Model File
		Model *model = new Model( pLevel );
		model->read( modelFilename );
		Metadata *modelMetadata = model->getMetadata();
		if( ! propertyFile )
			metadata->copy( "PROPERTIES", modelMetadata );

		// ===== INHERIT SOME METADATA VALUES
		metadata->copy( "EXCLUDED_RESIDUES", modelMetadata );
		metadata->copy( "PROPERTIES",        modelMetadata );
		metadata->copy( "SHELLS",            modelMetadata );
		metadata->copy( "SHELL_WIDTH",       modelMetadata );
		numShells = metadata->asInt( "SHELLS" );

		// Load Environment File (aka Feature vector)
		igzstream environmentFile( environmentFilename );
		string buffer;

		environmentMetadata = new Metadata( string( environmentFilename ));
		Environments::enableMetadataKeywords( environmentMetadata );
		do {
			getline( environmentFile, buffer );
		} while( environmentMetadata->parse( buffer ) );

		metadata->checkSupport( "EXCLUDED_RESIDUES", environmentMetadata );
		metadata->checkSupport( "PROPERTIES",        environmentMetadata );
		metadata->checkSupport( "SHELLS",            environmentMetadata );
		metadata->checkSupport( "SHELL_WIDTH",       environmentMetadata );

		metadata->write();

		// Load the rest of the Environment File and calculate the scores for each line
		do {
			Environment *environment = new Environment( numShells );
			environment->parse( buffer );
			double score = calculateScore( environment, model, countEmpty );
			if ( noCutoff || score >= scoreCutoff ) {
				printf( "%s\t%3f", environment->getLabel().c_str(), score);
				if (! environment->getDescription().empty())
					printf( "\t%s", environment->getDescription().c_str());
				printf( "\n");
			}
			delete environment;
		} while( getline( environmentFile, buffer ));

		delete model;
		delete environmentMetadata;
		delete metadata;
		return 0;
	}
};
};

int main( int argc, char *argv[] ) {
	Application::Scoreit scoreit;
	return scoreit.main( argc, argv );
}
