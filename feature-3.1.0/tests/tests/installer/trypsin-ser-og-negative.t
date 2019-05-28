use lib qw( ./lib ../lib );
use Feature::InstallTest;
use Feature::File::FeatureVector;
use Test::Simple tests => 2;

my $featurize = "../src/featurize";
my $vector    = new Feature::File::FeatureVector( "./hits/trypsin_ser_og.neg.ff" );

ok( Feature::InstallTest::run( "$featurize -P ./data/serprot/trypsin_ser_og.neg.ptf > ./testout/trypsin_ser_og.neg.ff 2>/dev/null" ));
ok( ! $vector->diff( "./testout/trypsin_ser_og.neg.ff" ));
