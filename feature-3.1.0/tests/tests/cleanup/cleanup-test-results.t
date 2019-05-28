use lib qw( ./lib ../lib );
use Feature::Test;
use Test::Simple tests => 1;

my $test = new Feature::Test();
ok( $test->run( "rm -rf testout/*" ));
