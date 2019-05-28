// FEATURE;
// See COPYING and AUTHORS for copyright and license agreement terms.

// $Id: Model.cc 1285 2012-03-22 01:12:42Z mikewong899 $

#include "Model.h"

Model::Model( double pLevel, int numBins, int numShells ) {
	metadata = new Metadata( "buildmodel" );
	enableMetadataKeywords( metadata );

	metadata->set( "P_LEVEL", pLevel );
	metadata->set( "BINS",    numBins );
	metadata->set( "SHELLS",  numShells );

	this->numBins       = numBins;
	this->numShells     = numShells;
    this->pLevel        = pLevel;
	this->numProperties = PropertyList::size();
}

Model::~Model() {
    for(Scores::iterator it = scores.begin(); it != scores.end(); it++) {
        delete it->second; // delete Score objects
        it->second = NULL;
    }
    scores.clear();
	delete metadata;
}

/**
 * @brief Enables metadata keywords relevant to models.
 **/
void Model::enableMetadataKeywords( Metadata *metadata ) {
	metadata->enable( "BINS" );
	metadata->enable( "BOUNDS_FILE" );
	metadata->enable( "EXCLUDED_RESIDUES" );
	metadata->enable( "EXTENDED_OUTPUT" );
	metadata->enable( "INSTITUTION" );
	metadata->enable( "PROPERTIES" );
	metadata->enable( "PROPERTIES_FILE" );
	metadata->enable( "P_LEVEL" );
	metadata->enable( "P_SITE" );
	metadata->enable( "SHELLS" );
	metadata->enable( "SHELL_WIDTH" );
	metadata->enable( "TRAIN_NEG_FILE" );
	metadata->enable( "TRAIN_POS_FILE" );
	metadata->enable( "VERBOSITY" );
}

/**
 * @brief Concatenates the property name and shell index
 * @param property Name of the property
 * @param shellIndex Shell index
 **/
string Model::combine( const string& property, int shellIndex ) {
	stringstream propertyShell;
	propertyShell << property << "-" << shellIndex;
	return propertyShell.str();
}

/**
 * @brief Returns the property score data for the given property and shell index
 **/
Model::Score *Model::getScore( const string& property, int shellIndex ) {
    string propertyShell = combine( property, shellIndex );
   	if( scores.find( propertyShell ) == scores.end() ) {
		fatal_error( "Model: Undefined property scores for '%s'.\n", propertyShell.c_str() );
	}
	return scores[ propertyShell ];
}

/**
 * @brief Sets the property score data for the given property and shell index
 **/
void Model::include( Model::Score *score ) {
    string propertyShell = combine( score->property, score->shellIndex );
	if( scores.find( propertyShell ) == scores.end() ) {
		scores[ propertyShell ] = score;
	} else {
		fatal_error( "Model: Duplicate definition for property scores for '%s'.\n", propertyShell.c_str() );
	}
}

/**
 * @brief Retrieves a score for a given property at a given shell index with a given property value
 **/
double Model::getBinScore( const string& property, int shellIndex, double value ) {
    Score *score = getScore( property, shellIndex );
    if (score == NULL)
        return (0.0);
    if (score->pValue > this->pLevel)
        return (0.0);
    int bin = (int) ((value - score->lowValue) / score->binSize);
	bin = bin < 0 ? 0 : bin;
	bin = bin > (numBins - 1) ? (numBins - 1) : bin;
    return ( score->binScores[ bin ] );
}

/**
 * @brief Reads tab-delimited format
 *
 * The model file has two sections: the header and the body. The header
 * contains model file Metadata information, such as properties used to build
 * the model, the filename for the positive and negative training data, and
 * the date the model was made. The body of the model file has the property 
 * name/shell index, the Mann-Whitney-Wilcoxon Rank Sum p-value, and
 * scoring statistics for each property name/shell index.
 *
 * If the PropertyList hasn't already been defined (usually via a Selectable
 * Properties file), then the Metadata is used to define the PropertyList.
 **/
void Model::read( string filename ) {
    string       line;
	igzstream    modelFile( filename.c_str() );

	if( ! modelFile )
		fatal_error( "Model: Can't open Model File '%s' for reading.\n", filename.c_str() );

	// ===== PARSE METADATA
	metadata->setName( filename );
    string token;
    while ( getline( modelFile, line )) {
		if( ! metadata->parse( line ) ) break;
    };
	if( ! PropertyList::loaded() ) PropertyList::load( metadata );
	numProperties = PropertyList::size();

	// ===== PARSE BIN SCORES
    for (int i = 0; i < numShells; i++) {
        for (int j = 0; j < numProperties; j++) {
			if( ! modelFile ) continue;
			stringstream bufferStream( line );
            Score *score = new Score( numBins );
			score->parse( bufferStream );
            include( score );
            getline( modelFile, line );
        }
    }
}

void Model::train( Environments *sites, Environments *nonsites, double pSite, Model::BoundsFile *boundsFile ) {
	metadata->set( "P_SITE",  pLevel );
	this->pSite = pSite;

	for( int shellIndex = 0; shellIndex < numShells; shellIndex++ ) {
		for( int propertyIndex = 0; propertyIndex < numProperties; propertyIndex++ ) {
			string propertyName = PropertyList::get( propertyIndex )->getName();

			Doubles *siteValues    = sites->getValues(    propertyIndex, shellIndex );
			Doubles *nonsiteValues = nonsites->getValues( propertyIndex, shellIndex );

			// ===== CALCULATE MANN-WHITNEY-WILCOXON RANK SUM P-VALUES
			RankSum rs( (*siteValues), (*nonsiteValues) );

			// ===== USE NAIVE BAYES TO CALCULATE SCORES
			NaiveBayesClassifier nbc( numBins, pSite );

			if( boundsFile ) {
				double min, max;
				boundsFile->getValueRange( propertyIndex, shellIndex, &min, &max );
				nbc.setRange( min, max );
			}

			nbc.train( siteValues, nonsiteValues );

			Score *score      = new Score( numBins );
			score->property   = propertyName;
			score->shellIndex = shellIndex;
			score->pValue     = rs.pValue;
			score->tValue     = rs.tValue;
			score->binSize    = nbc.getBinSize();
			score->lowValue   = nbc.getMin();
			score->binScores  = Doubles( numBins );
			for( int bin = 0; bin < numBins; bin++ ) {
				score->binScores[ bin ] = nbc.getBinScore( bin );
			}

			include( score );
		}
	}
}

void Model::write( bool extendedOutput ) {
	metadata->write();
	for( int shellIndex = 0; shellIndex < numShells; shellIndex++ ) {
		for( int propertyIndex = 0; propertyIndex < numProperties; propertyIndex++ ) {
			string propertyName = PropertyList::get( propertyIndex )->getName();
			Score *score = getScore( propertyName, shellIndex );

			printf( "%s-%d", propertyName.c_str(), shellIndex);
			printf( "\t%g", score->pValue);
			printf( "\t%g", score->lowValue );
			printf( "\t%g", score->binSize );
			for (int bin = 0; bin < numBins; bin++) {
				printf( "\t%g", score->binScores[ bin ]);
			}
			if (extendedOutput) {
				printf( "\t#\t%g", score->tValue);
			}
			printf( "\n" );
		}
	}
}

Model::Score::Score( int numBins ) {
	this->numBins = numBins;
}

void Model::Score::parse( stringstream &bufferStream ) {
	string token;

	// ===== PARSE PROPERTY/VOLUME DESCRIPTION
	// e.g. HYDROXYL-2 or RESIDUE_CLASS2_IS_BASIC-4
	bufferStream >> token;
	size_t i = token.rfind( '-' );
	if( i == string::npos )
		fatal_error( "Model: Problem parsing model file line: property/shell index description not found in '%s'\n", token.c_str() );

	property = token.substr( 0, i );
	stringstream( token.substr( (i + 1 ))) >> shellIndex;

	bufferStream >> pValue;
	bufferStream >> lowValue;
	bufferStream >> binSize;
	
	binScores = Doubles( numBins );
	for( int bin = 0; bin < numBins; bin++ ) {
		bufferStream >> binScores[ bin ];
	}
}
