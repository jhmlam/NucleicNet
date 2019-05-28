// FEATURE
// See COPYING and AUTHORS for copyright and license agreement terms.

// $Id: RankSum.h 1174 2011-10-28 00:56:50Z mikewong899 $

#ifndef RANKSUM_H
#define RANKSUM_H

#include <cstdlib>
#include <cmath>   // sqrt
#include <iostream>
#include "logutils.h"
#include "stl_types.h"

/**
 * @brief Compute Mann-Whitney-Wilcoxon (MWW) Rank-Sum Test and associated descriptive statistics
 * @ingroup classification_module
 *
 * The goal of the MWW Rank-Sum Test is to determine if two samples have the 
 * same distribution shape; the null-hypothesis is that the two samples
 * (site and non-site) are drawn from a single population.
 * 
 * The p statistic measures the overlap of the two samples; a p value
 * of 0.5 indicates that the two samples are drawn from a single population.
 * p values at either extreme from 0 to 1 indicate that the two samples
 * are from different populations.
 *
 * If site and non-site samples for a given property are from a single population
 * then it is unlikely that the property will give any meaningful information
 * for classification. Otherwise, if the p values are from different
 * populations, then they may indicate that the property is a significant
 * contributor for classification.
 * 
 * @note In terms of mathematical organization, Rank and Ranks should be independent classs and RankSum should be an operation of Ranks
 **/
class RankSum
{
	public:
		/** 
		 * @brief Computes the rank of a given data value
		 **/
		class Rank {
			public:
				double value;
				bool   smallList;
				double rank;

				/**
				 * @brief Provides an ordering for Ranks
				 **/
				struct sort {
					const bool operator() ( Rank *a, Rank *b ) {
						return (a->value < b->value);
					}
				};

				Rank( double = 0.0, bool = false );
		};
		/**
		 * @brief An ordered vector of Rank objects
		 **/
		class Ranks : public vector<Rank *> {
			public:
				void sort() {
					std::sort( begin(), end(), Rank::sort() );
				}
		};
		RankSum();
		RankSum( Doubles &, Doubles & ); 
		RankSum( double [], int, double [], int );
	   ~RankSum();

		double sumOfRanks;
		double mean;
		double standardDeviation;
		double degreesOfFreedom;
		double tValue;
		double pValue;
		int    nBig;
		int    nSmall;
		Ranks  ranks;

		void   calculate( Doubles &, Doubles & );
		void   computeRanks();
		void   createRanks( Doubles &, Doubles & );
		double getPValue( double, double );
		int    numSameValues( int );
		double sumRanks();
		double tauAdjustment();
};

#endif
