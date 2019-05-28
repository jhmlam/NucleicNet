#! /usr/bin/perl

my $file = shift;
open FILE, $file or die "Can't open '$file' for reading $!";
while( <FILE> ) {
	chomp;
	s/\(//g;
	s/\)//g;
	s/^\s+//;
	s/\s+$//;
	my @tokens = split /\s+/;
	my $line = {};

	while( @tokens ) {

		my $keyword = shift @tokens;
		# ===== IGNORE METADATA FOR NOW
		if       ( $keyword =~ /:NUM-BINS/ ) {
			my $value = shift @tokens;
		} elsif  ( $keyword =~ /:PROB-SITE/ ) {
			my $value = shift @tokens;
		} elsif  ( $keyword =~ /:PROB-NONSITE/ ) {
			my $value = shift @tokens;

		# ===== HANDLE MODEL DATA
		} elsif  ( $keyword =~ /:PROPERTY/ ) {
			$line->{ property }  = shift @tokens;

		} elsif  ( $keyword =~ /:VOLUME/ ) {
			$line->{ shell }     = shift @tokens;

		} elsif  ( $keyword =~ /:P-VALUE/ ) {
			$line->{ p_value }   = shift @tokens;

		} elsif  ( $keyword =~ /:BIN-SIZE/ ) {
			$line->{ bin_size }  = shift @tokens;

		} elsif  ( $keyword =~ /:LOW-VALUE/ ) {
			$line->{ low_value } = shift @tokens;

		} elsif  ( $keyword =~ /:SCORES/ ) {
			@{$line->{ scores }} = splice @tokens, 0, 5;
			print join( "\t", 
				"$line->{ property }-$line->{ shell }", 
				$line->{ p_value }, 
				$line->{ low_value },
				$line->{ bin_size }, 
				@{ $line->{ scores }}
			) . "\n";
			$line = {};

		} else { 
			die "Illegal keyword '$keyword' $!";
		}
	}
}
close FILE;
