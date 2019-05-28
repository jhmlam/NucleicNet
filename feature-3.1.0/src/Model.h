// FEATURE
// See COPYING and AUTHORS for copyright and license agreement terms.

// $Id: Model.h 1615 2013-12-16 23:05:43Z mikewong899 $

// Header file for Model.cpp

#ifndef SCOREFILE_H
#define SCOREFILE_H

#include <sstream>
#include <iostream>
#include <string>
#include <zlib.h>
#include "gzstream.h"
#include "Environments.h"
#include "MetaData.h"
#include "NaiveBayesClassifier.h"
#include "Parser.h"
#include "PropertyList.h"
#include "RankSum.h"
#include "Utilities.h"
#include "stl_types.h"
#include "unordered_map.h"

#define DEFAULT_NUMBINS 5
#define DEFAULT_NUMSHELLS 6
#define DEFAULT_PLEVEL 1.0

using std::pair;
using std::string;
using std::stringstream;

/**
 * @brief Reads a complete score table e.g. a @ref model "model file"
 * @ingroup classification_module
 *
 * The purposes of the Model class are: (1) to use the NaiveBayesClassifier
 * class to train a model of protein functionality and write the results to
 * a file; and (2) to parse the model files for subsequent use towards
 * classifying unknown proteins.
 *
 * The Model class must be responsible for managing the memory allocation
 * and deallocation of its component object instances (e.g. Score and
 * Metadata).
 *
 * @note Formerly named ScoreFile, which is inconsistent with current
 * terminology used in the Helix Group and consequently at CCLS
 * @note LISP format deprecated with FEATURE 3.0 release
 * @todo We need a tool (preferably in python) to convert model files in LISP format to tabbed format
 **/
class Model {
	public:
		/**
		 * @brief Parses a file which lists max and min values for each Property
		 * per Shell. These ranges are used to collect continuous
		 * property values into discrete bins (via linear interpolation).
		 * @ingroup classification_module
		 **/
		class BoundsFile {
			int numProperties;
			Doubles minValues;
			Doubles maxValues;

			public:
				BoundsFile(char *filename, int numProperties) {
					this->numProperties = numProperties;

					ifstream ins(filename);
					double minVal, maxVal;
					while (ins) {
						ins >> minVal >> maxVal;
						if (!ins)
							break;
						minValues.push_back(minVal);
						maxValues.push_back(maxVal);
					}
				}

				void getValueRange(int propIdx, int shellIdx, double *minValue, double *maxValue) {
					int idx = shellIdx * numProperties + propIdx;
					*minValue = minValues[idx];
					*maxValue = maxValues[idx];
				}
		};

		/**
		 * @brief Holds the property score information: Rank Sum pValue, low
		 * score, bin size, and NB scores for each bin
		 * @ingroup classification_module
		 **/
		class Score {
		public:
			string  property;
			int     shellIndex;
			double  pValue;
			double  tValue;
			double  binSize;
			double  lowValue;
			int     numBins;
			Doubles binScores;

			Score( int );
			void          parse( stringstream & );
		};

		/**
		 * @brief A group of Score objects, indexed by a string composed of a
		 * property name and shell index (e.g. ATOM_TYPE_IS_C-0)
		 * @ingroup classification_module
		 **/
		class Scores : public unordered_map< string, Score * > {};

	protected:
		double    pLevel;
		double    pSite;
		int       numBins;
		int       numShells;
		int       numProperties;
		Scores    scores;
		Metadata *metadata;

		inline static string  combine(const string& Property, int shellIndex );
		Score                *getScore( const string& Property, int shellIndex );
		void                  include( Score *pScore );

	public:
		Model( double pLevel = DEFAULT_PLEVEL, int numBins = DEFAULT_NUMBINS, int numShells = DEFAULT_NUMSHELLS );
		~Model();

		Metadata   *getMetadata() { return metadata; };
		double      getBinScore( const string& Property, int shellIndex, double value);
		void        read( string filename );
		void        setMetadata( Metadata * _metadata ) { delete metadata; metadata = _metadata; };
		void        train( Environments *, Environments *, double, BoundsFile * = NULL );
		void        write( bool = false );
		static void enableMetadataKeywords( Metadata * );

};

#endif
