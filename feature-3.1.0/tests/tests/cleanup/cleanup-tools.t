use lib qw( ./lib ../lib );
use Feature::Test;
use Test::Simple tests => 1;

my $test = new Feature::Test();
my $tools = $test->tools_package();
my $command = "rm -f $tools 2>/dev/null";

ok( $test->run( $command ));
