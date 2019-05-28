// FEATURE
// See COPYING and AUTHORS for copyright and license agreement terms.

// $Id: RankSum.cc 1174 2011-10-28 00:56:50Z mikewong899 $

#include "RankSum.h"

RankSum::Rank::Rank( double value, bool smallList ) : 
	value( value ), 
	smallList( smallList ) 
{
	rank = 0.0;
}

void
RankSum::createRanks( Doubles &bigList, Doubles &smallList ) {
    for (unsigned int i = 0; i < bigList.size(); i++) {
		ranks.push_back( new Rank( bigList[ i ], false ));
    }
    for (unsigned int i = 0; i < smallList.size(); i++) {
		ranks.push_back( new Rank( smallList[ i ], true ));
    }
}

RankSum::~RankSum() {
    for(Ranks::iterator it = ranks.begin(); it != ranks.end(); it++) {
        delete *it;
        *it = NULL;
    }
    ranks.clear();
}

int 
RankSum::numSameValues( int i ) {
    double value = ranks[i]->value;

    unsigned int num = 1;
    while ((num + i < ranks.size()) && (ranks[num + i]->value == value))
        num++;
    return (num);
}

void 
RankSum::computeRanks() {
    for (unsigned int i = 0; i < ranks.size(); i++)
        ranks[i]->rank = i + 1;

    unsigned int i = 0;
    while (i < ranks.size()) {
        int num = numSameValues( i );
        if (num > 1) {
            int low        = (int) ranks[i]->rank;
            int high       = (int) ranks[i]->rank + num - 1;
            double sumRank = (double) (low + high) * (high - low + 1) / 2.0;
            double newRank = sumRank / num;

            for (int j = 0; j < num; j++)
                ranks[i + j]->rank = newRank;
        }
        i += num;
    }
}

double 
RankSum::sumRanks() {
    double sum = 0;
    for (unsigned int i = 0; i < ranks.size(); i++) {
        if (ranks[i]->smallList)
            sum += ranks[i]->rank;
    }

    return (sum);
}

double 
RankSum::tauAdjustment() {
	vector<int> tauVector;

	// ===== COMPUTE TAU
	unsigned int i = 0;
	while (i < ranks.size()) {
        int num = numSameValues( i );
        if (num > 1) {
			tauVector.push_back( num );
        }
        i += num;
    }

	// ===== CALCULATE ADJUSTMENT
    double adjustment = 0;
    for (unsigned int i = 0; i < tauVector.size(); i++) {
        double tau = tauVector[i];
        adjustment += (tau - 1.0) * tau * (tau + 1.0);
    }
    return (adjustment);
}

double 
RankSum::getPValue(double tValue, double degreesOfFreedom) {
    tValue = fabs(tValue);
    if (degreesOfFreedom >= 1000) {
        if (tValue >= 3.2905)
            return (0.001);
        if (tValue >= 3.0902)
            return (0.002);
        if (tValue >= 2.8070)
            return (0.005);
        if (tValue >= 2.5758)
            return (0.01);
        if (tValue >= 2.3263)
            return (0.02);
        if (tValue >= 1.9600)
            return (0.05);
        if (tValue >= 1.6449)
            return (0.10);
        if (tValue >= 1.2816)
            return (0.20);
        return (0.50);
    } else if (degreesOfFreedom >= 100) {
        if (tValue >= 3.390)
            return (0.001);
        if (tValue >= 3.174)
            return (0.002);
        if (tValue >= 2.871)
            return (0.005);
        if (tValue >= 2.626)
            return (0.01);
        if (tValue >= 2.364)
            return (0.02);
        if (tValue >= 1.984)
            return (0.05);
        if (tValue >= 1.660)
            return (0.10);
        if (tValue >= 1.290)
            return (0.20);
        return (0.50);
    } else if (degreesOfFreedom >= 10) {
        if (tValue >= 3.6)
            return (0.005);
        return (1.0);
    }
    return (1.0);
}

RankSum::RankSum() {
}

RankSum::RankSum( Doubles &sites, Doubles &nonsites ) {
	calculate( sites, nonsites );
}

RankSum::RankSum( double siteValues[], int nSites, double nonsiteValues[], int nNonsites ) {
	Doubles sites;
	Doubles nonsites;
	for( int i = 0; i < nSites; i++ ) {
		sites.push_back( siteValues[ i ] );
	}
	for( int i = 0; i < nNonsites; i++ ) {
		nonsites.push_back( nonsiteValues[ i ] );
	}
	calculate( sites, nonsites );
}

/**
 * @brief This method performs the actual MWW and statistics calculation
 **/
void
RankSum::calculate( Doubles &sites, Doubles &nonsites ) {
	Doubles bigList;
	Doubles smallList;
    bool isSwapped;

    if (nonsites.size() >= sites.size()) {
		smallList  = sites;
		bigList    = nonsites;
        isSwapped  = false;
    } else {
		smallList  = nonsites;
		bigList    = sites;
        isSwapped  = true;
    }

    nBig   = bigList.size();
    nSmall = smallList.size();

    if (bigList.size() <= 8)
		warning( "Rank Sum called with list (n <= 8).\n" );

    createRanks( bigList, smallList );
    ranks.sort();
    computeRanks();

    sumOfRanks        = sumRanks();
    mean              = (double) nSmall * (ranks.size() + 1.0) / 2.0;
    double sd         = (double) nBig * nSmall * (ranks.size() + 1.0) / 12.0;
    double adjustment = (double) nBig * nSmall * tauAdjustment() / (12.0 * ranks.size() * (ranks.size() - 1.0));
    standardDeviation = sqrt(sd - adjustment);

    if (standardDeviation == 0.0)
        tValue = 0;
    else {
        double diff = (sumOfRanks - mean);
        double correction = (diff == 0) ? 0 : (diff > 0) ? -0.5 : 0.5;
        tValue = (diff + correction) / standardDeviation;
        if (isSwapped)
            tValue = - tValue;
    }

    degreesOfFreedom = 1E6;
    pValue = getPValue(tValue, degreesOfFreedom);

}

