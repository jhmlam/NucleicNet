use lib qw( ./lib ../lib );
use Feature::InstallTest;
use Feature::File::Model;
use Test::Simple tests => 2;

my $buildmodel = "../src/buildmodel";
my $model      = new Feature::File::Model( "./hits/trypsin_ser_og.model" );

ok( Feature::InstallTest::run( "$buildmodel ./testout/trypsin_ser_og.pos.ff ./testout/trypsin_ser_og.neg.ff > ./testout/trypsin_ser_og.model" ));
ok( ! $model->diff( "./testout/trypsin_ser_og.model" ));
