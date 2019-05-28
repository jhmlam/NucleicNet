use lib qw( ./lib ../lib );
use Feature::Test;
use Test::Simple tests => 1;

my $test = new Feature::Test();
my $tools = $test->tools();
my $command = "rm -rf $tools 2>/dev/null";

ok( $test->run( $command ));
