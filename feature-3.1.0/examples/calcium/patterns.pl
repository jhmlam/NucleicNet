#! /usr/bin/perl

my $last_pdbid = undef;
my $pdb        = undef;

open FILE, shift;
while( <FILE> ) {
	chomp;
	s/[\(\)]//g;
	my ($pdbid, $xlabel, $x, $ylabel, $y, $zlabel, $z, $site) = split /\s+/;
	$pdbid =~ s/"//g;
	next unless $site eq 'T';
	my $point = { pdbid => $pdbid, x => $x, y => $y, z => $z };

	my $change = $pdbid ne $last_pdbid;
	if( $change || not defined $pdb ) {
		$pdb = read_pdb( $pdbid );
		print "------------------------------------------------------------\n"
	}
	$last_pdbid = $pdbid;
	next unless defined $pdb;

	my $min = { distance => undef, from => undef };
	foreach my $atom (@$pdb) {
		my $distance = get_distance( $point, $atom );
		if( $min->{ distance } > $distance || not defined $min->{ distance } ) {
			$min->{ distance } = $distance;
			$min->{ from }     = $atom;
		}
	}

	my $atom = $min->{ from };
	printf "%-4s %-3s %-4s %-5d (%6.2f, %6.2f, %6.2f) %6.2f Angstroms\n", $pdbid, $atom->{ residue }, $atom->{ name }, $atom->{ sequence }, $atom->{ x }, $atom->{ y }, $atom->{ z }, $min->{ distance };
}
close FILE;	

# ============================================================
sub get_distance {
# ============================================================
	my $a = shift;
	my $b = shift;

	my $dx = ($a->{ x } - $b->{ x }) ** 2;
	my $dy = ($a->{ y } - $b->{ y }) ** 2;
	my $dz = ($a->{ z } - $b->{ z }) ** 2;

	my $distance = sqrt( $dx + $dy + $dz );
	return $distance;
}

# ============================================================
sub read_pdb {
# ============================================================
	my $pdbid = shift;
	my $file1 = "/usr/local/feature/data/pdb/$pdbid.pdb.gz";
	my $file2 = "/Users/mikewong899/Desktop/pdb/" . uc( $pdbid ) . ".pdb.gz";
	unless( -e $file1 || -e $file2 ) {
		warn "$pdbid\n";
		return;
	}
	my $file = -e $file1 ? $file1 : $file2;
	my $contents = `gzcat $file`;
	my @contents = split /\n/, $contents;
	my @contents = grep { /^ATOM/ || /^HETATM/ } @contents;

	my @atoms    = ();
	foreach my $atom (@contents) {
		my ($record, $sequence, $fullname, $residue, $chain, $res_sequence, $x, $y, $z, $occupancy, $bfactor, $name) = split /\s+/, $atom;
		push @atoms, { name => $fullname, residue => $residue, sequence => $sequence, x => $x, y => $y, z => $z };
	}
	return [ @atoms ];
}
