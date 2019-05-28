// FEATURE
// See COPYING and AUTHORS for copyright and license agreement terms.

// $Id: NaiveBayesClassifier.cc 1618 2014-01-15 06:01:03Z mikewong899 $

#include "NaiveBayesClassifier.h"

NaiveBayesClassifier::NaiveBayesClassifier( int _numBins, double _pPos ) {
	numBins    = _numBins;
	pPos       = _pPos;
	pNeg       = 1.0 - pPos;
	isRangeSet = false;
}

NaiveBayesClassifier::~NaiveBayesClassifier() { 
}

void NaiveBayesClassifier::setRange( double min, double max ) {
	this->min = min + FUDGE_FACTOR;
	this->max = max;
	isRangeSet = true;
}

void NaiveBayesClassifier::train( Doubles *pos, Doubles *neg ) {
	posBinCount = vector<int>( numBins );
	negBinCount = vector<int>( numBins );

	if( ! isRangeSet ) calculateRange( pos, neg );
	calculateBinSize();

	numPos = count( pos, posBinCount );
	numNeg = count( neg, negBinCount );

	calculateBinScores();
}

/**
 * @brief Calculate the score for each bin
 *
 * Let pBin be the probability that the property data has a value that falls within a given range.
 * Let pPos be the probability of positive protein function classification.
 *
 * Property values are clustered into discrete bins. The bins have a uniform range
 * of values. Each bin has a probability associated with it (pBin). 
 * 
 * Read variables as "Probability x given y", e.g. pBinPos is the probability
 * of bin given site. An more complete description would be: p( bin | site ) is
 * the probability of a microenvironment having a physicochemical property
 * value that falls within a given continuous range of a discrete bin, given
 * that the microenvironment characterizes a protein function to be modeled.
 *
 * pBinPos := p( bin | site )
 * pBinNeg := p( bin | ~site )
 * pBin    := p( bin )
 * pPosBin := p( pos | bin )
 **/
void NaiveBayesClassifier::calculateBinScores() {
	binScore = vector<double>( numBins );
	for( int bin = 0; bin < numBins; bin++ ) {
		double pBinPos  = ((double) posBinCount[ bin ]) / ((double) numPos);
		double pBinNeg  = ((double) negBinCount[ bin ]) / ((double) numNeg);
		double pBin     = (pBinPos * pPos) + (pBinNeg * pNeg);
		double pPosBin  = (pBinPos * pPos) / pBin;
		double minScore = log( pPos );

		if( pBin == 0.0 ) { 
			binScore[ bin ] = 0.0;
		} else { 
			double score = log( pPosBin / pPos );
			binScore[ bin ] = score < minScore ? minScore : score;
		}
	}
}

void NaiveBayesClassifier::calculateRange( Doubles *pos, Doubles *neg ) {
    min = (*pos)[0];
    max = (*pos)[0];

    Doubles::iterator iter;
    for (iter = pos->begin(); iter != pos->end(); iter++) {
        if ((*iter) > max) { max = (*iter); }
        if ((*iter) < min) { min = (*iter); }
    }
    for (iter = neg->begin(); iter != neg->end(); iter++) {
        if ((*iter) > max) { max = (*iter); }
        if ((*iter) < min) { min = (*iter); }
    }
	min += FUDGE_FACTOR;
}

int NaiveBayesClassifier::count( Doubles *values, vector<int> &countBin ) {
	Doubles::iterator i;
	int bin;
	int total = 0;
	for( i = values->begin(); i != values->end(); i++ ) {
		bin = (int) (((*i) - min) / binSize);
		if( bin < 0 )            bin = 0;
		if( bin > (numBins - 1)) bin = (numBins - 1);
		countBin[ bin ]++;
		total++;
	}
	return total;
}
