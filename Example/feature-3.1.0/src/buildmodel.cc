// $Id: buildmodel.cc 1285 2012-03-22 01:12:42Z mikewong899 $

/**
 * \page buildmodel buildmodel
 *
 * Creates a Naive Bayesian classifier \ref model "model" from a set of
 * positive and negative training data.
 *
 * \section buildmodel_input Input
 * \li A \ref feature_vector "feature vector file" of known similar sites
 * \li A \ref feature_vector "feature vector file" of known non-similar sites
 *
 * A feature vector file is produced by \ref featurize and summarizes the 
 * physicochemical properties around one or more points in or near a 
 * biomolecule. 
 *
 * \section buildmodel_output Output
 * \li A \ref model "model file" that contains information that can be used for protein function classification.
 *
 * \section buildmodel_usage Usage
\verbatim
Usage: buildmodel [OPTIONS] SITE_FEATURE_VECTOR_FILE NONSITE_FEATURE_VECTOR_FILE

Options:
    -v  Increase verbosity
    -n NUMSHELLS [Default: 6]
        Set number of shells to NUMSHELLS for environment mode
    -b NUMBINS [Default: 5]
        Set number of bins
    -s SITEPRIOR [Default: 0.01]
        Set the site prior
    -B BOUNDSFILE
        Specify the min/max in shell major order
    -x
        Increase extended output
    -l PROPERTIESFILE
        Read a property list from PROPERTIESFILE

\endverbatim
 * \subsection buildmodel_options_detail Options Detail
 *
 * The \c SITEFEATURES variable is the filename for a \ref feature_vector "Feature Vector File"
 *
 * The \c NONSITEFEATURES variable is the filename for a \ref feature_vector "Feature Vector File"
 *
 * The verbosity option (<tt>-v</tt>) increases the number of output messages, 
 * some of which are useful for understanding how \c buildmodel works and for 
 * debugging.
 *
 * The \c NUMSHELLS variable controls number of shells \c featurize 
 * will evaluate. Larger values will create more data and take longer
 * to compute; however the data may be noisy. The default value is strongly
 * recommended.
 * 
 * The \c NUMBINS variable controls the number of discrete probability states
 * to use for the purposes of calculating the Bayesian probability.
 *
 * The \c SITEPRIOR variable controls the prior probability for the Bayesian
 * classification model for positive site classification.
 *
 * The \c BOUNDSFILE is a text file that specifies the min/max for each bin
 * for each shell to calculate the Bayesian probability. The default behavior
 * is to linearly discretize the property values into \c NUMBINS bins.
 *
 * For more details about the theory behind the Bayesian classification model
 * We recommend <em>Artificial Intelligence: A Modern Approach</em>, by Russell
 * and Norvig, Prentice Hall Press &copy;2003. Chapters 13 and 14 as a suitable
 * introductory text for advanced undergraduates and graduate students.
 *
 * \section buildmodel_examples Examples
 * \code
 * buildmodel model.pos.ff model.neg.ff > SNAKE_TOXIN.2.CYS.SG.model
 *
 * buildmodel trypsin_ser_og.pos.ff trypsin_ser_og.neg.ff > trypsin_ser_og.model
 * \endcode
 **/

#include <cstdio>
#include <unistd.h>
#include <cmath>
#include <vector>
#include <string>
#include <zlib.h>
#include "gzstream.h"
#include "Environments.h"
#include "Model.h"
#include "PropertyList.h"
#include "Utilities.h"
#include "config.h"

using std::ifstream;
using std::string;

const int    DEFAULT_VERBOSITY = 0;
const double DEFAULT_SITEPRIOR = 0.01;

const static string progname   = "buildmodel";

extern char *optarg;
extern int optind;

void eusage() {
    printf("\n%s (build %s) - %s\n\n", PACKAGE_STRING, SVN_REV, progname.c_str());
    printf("Usage: %s [OPTIONS] SITEFEATURES NONSITEFEATURES\n", progname.c_str());
    printf("\n");
    printf("Options:\n");
    printf("    -v  Increase verbosity\n");
    printf("    -n NUMSHELLS [Default: %d]\n", DEFAULT_NUMSHELLS);
    printf("        Set number of shells to NUMSHELLS for environment mode\n");
    printf("    -b NUMBINS [Default: %d]\n", DEFAULT_NUMBINS);
    printf("        Set number of bins\n");
    printf("    -s SITEPRIOR [Default: %g]\n", DEFAULT_SITEPRIOR);
    printf("        Set the site prior\n");
    printf("    -B BOUNDSFILE\n");
    printf("        Specify the min/max in shell major order\n");
    printf("    -l PROPERTIESFILE\n");
    printf("        Read a property list from PROPERTIESFILE\n");
    printf("    -x\n");
    printf("        Increase extended output\n");
    exit(-1);
}

class Application {
	public:
/**
 * \brief The heart of the \ref buildmodel "buildmodel" application, this class generates a FEATURE \ref model
 * \ingroup buildmodel_application
 **/
class BuildModel {
private:
    int                verbosity;
    int                numShells;
    int                numBins;
    double             sitePrior;
    Environments      *siteEnvironments;
    Environments      *nonsiteEnvironments;
    Model::BoundsFile *boundsFile;
    int                extendedOutput;
    char              *propertyFile;
	Metadata          *metadata;
	Metadata          *siteMetadata;    // Site environments metadata
	Metadata          *nonsiteMetadata; // Nonsite environments metadata

public:
    BuildModel() {
        verbosity           = DEFAULT_VERBOSITY;
        numShells           = DEFAULT_NUMSHELLS;
        numBins             = DEFAULT_NUMBINS;
        sitePrior           = DEFAULT_SITEPRIOR;
        siteEnvironments    = NULL;
        nonsiteEnvironments = NULL;
        boundsFile          = NULL;
        propertyFile        = NULL;
        extendedOutput      = 0;
		metadata            = new Metadata( "buildmodel" );

		Model::enableMetadataKeywords( metadata );

		metadata->set( "BINS",      numBins );
		metadata->set( "P_LEVEL",   DEFAULT_PLEVEL );
		metadata->set( "P_SITE",    sitePrior );
		metadata->set( "SHELLS",    numShells );
		metadata->set( "VERBOSITY", verbosity );
	}

    ~BuildModel() {
        delete siteEnvironments;
        delete nonsiteEnvironments;
        delete boundsFile;
    }

	void parseArguments(int argc, char **argv) {
		char *boundsFilename = NULL;
		int opt;

		while ((opt = getopt(argc, argv, ":vn:b:s:B:l:x")) != -1) {
			switch (opt) {
				case 'v':
					setdebuglevel(++verbosity);
					metadata->set( "VERBOSITY", verbosity );
					break;
				case 'n':
					numShells = atoi(optarg);
					metadata->set( "SHELLS", numShells );
					break;
				case 'b':
					numBins = atoi(optarg);
					metadata->set( "BINS", numBins );
					break;
				case 's':
					sitePrior = atof(optarg);
					metadata->set( "P_SITE", sitePrior );
					break;
				case 'B':
					boundsFilename = optarg;
					metadata->set( "BOUNDS_FILE", boundsFilename );
					break;
				case 'l':
					propertyFile = optarg;
					metadata->set( "PROPERTIES_FILE", propertyFile );
					break;
				case 'x':
					extendedOutput++;
					metadata->set( "EXTENDED_OUTPUT", 1 );
					break;
				default:
					eusage();
					break;
			}
		}


        if( propertyFile ) {
            PropertyList::load( propertyFile );
            PropertyList::set( metadata );
        }

        int numargs = argc - optind;
        char **args = argv + optind;
        if (numargs != 2)
            eusage();

        char *siteEnvironmentFilename    = args[0];
        char *nonsiteEnvironmentFilename = args[1];
		metadata->set( "TRAIN_POS_FILE", siteEnvironmentFilename );
		metadata->set( "TRAIN_NEG_FILE", nonsiteEnvironmentFilename );

        siteEnvironments    = new Environments( siteEnvironmentFilename,    numShells );
        nonsiteEnvironments = new Environments( nonsiteEnvironmentFilename, numShells );

		siteMetadata    = siteEnvironments->getMetadata();
		nonsiteMetadata = nonsiteEnvironments->getMetadata();

		if( ! propertyFile && siteMetadata->hasProperties() ) {
			PropertyList::set( siteMetadata );
			metadata->copy( "PROPERTIES", siteMetadata );
		}

		// ===== INHERIT METADATA VALUES
		metadata->copy( "SHELLS",            siteMetadata );
		metadata->copy( "EXCLUDED_RESIDUES", siteMetadata );
		metadata->copy( "SHELL_WIDTH",       siteMetadata );
		numShells = metadata->asInt( "SHELLS" );

		// ===== CHECK SITE METADATA VS NONSITE METADATA
		siteMetadata->checkSupport( "EXCLUDED_RESIDUES", nonsiteMetadata );
		siteMetadata->checkSupport( "PROPERTIES",        nonsiteMetadata );
		siteMetadata->checkSupport( "SHELLS",            nonsiteMetadata );
		siteMetadata->checkSupport( "SHELL_WIDTH",       nonsiteMetadata );

        if (boundsFilename != NULL) {
            int numProperties = siteEnvironments->getNumProperties();
            boundsFile        = new Model::BoundsFile( boundsFilename, numProperties );
        }
    }

    void buildModel() {
		Model model( DEFAULT_PLEVEL, numBins, numShells );
		model.setMetadata( metadata );
		model.train( siteEnvironments, nonsiteEnvironments, sitePrior, boundsFile );
		model.write( extendedOutput );
    }

    int main(int argc, char **argv) {
        parseArguments(argc, argv);
        buildModel();
        return 0;
    }
};
};

int main(int argc, char **argv) {
    Application::BuildModel bm;
    return bm.main(argc, argv);
}
