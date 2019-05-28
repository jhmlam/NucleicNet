#! /usr/bin/env perl

# Concatenates arff files

$#ARGV > -1 || die "Usage: $0 FILES...\n";

$filename = shift @ARGV;
open(INFH,"< $filename") || die "Could not open $filename\n";
while (<INFH>) {
    print;
}
close(INFH);

for $filename (@ARGV) {
    open(INFH,"< $filename") || die "Could not open $filename\n";
    $count = 0;
    while (<INFH>) {
        $count++;
        last if /\@DATA/;
    }
    while (<INFH>) {
        print;
    }
    close(INFH);
}
