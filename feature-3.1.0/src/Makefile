CC=g++
FLAGS=-O3 -D 'SVN_REV="1639M"'


# =============================================================================
# TARGET-SPECIFIC VARIABLES
# =============================================================================
# FEATURE heavily relies on maps. In 2003, C++ Standard Library implemented
# faster unordered_maps as a beta feature. In 2011, C++ Standard Library
# included unordered_maps as an official component.
# -----------------------------------------------------------------------------
fast:     CFLAGS=$(FLAGS) -D CPP11_SUPPORT
fast03:   CFLAGS=$(FLAGS) -D CPP03_SUPPORT
standard: CFLAGS=$(FLAGS)

.PHONY: clean

# =============================================================================
# FEATURE EXECUTABLE BINARY BUILD TARGETS
# =============================================================================
all: standard

standard: executables
fast:     executables
fast03:   executables

executables: featurize buildmodel scoreit

clean:
	rm -f featurize buildmodel scoreit
	rm -f *.o

Atom.o: Atom.cc Atom.h
	$(CC) -c Atom.cc $(CFLAGS)

AtomProperties.o: AtomProperties.cc AtomProperties.h
	$(CC) -c AtomProperties.cc $(CFLAGS)

Atoms.o: Atoms.cc Atoms.h
	$(CC) -c Atoms.cc $(CFLAGS)

Environments.o: Environments.cc Environments.h
	$(CC) -c Environments.cc $(CFLAGS)

ForceField.o: ForceField.cc ForceField.h
	$(CC) -c ForceField.cc $(CFLAGS)

MetaData.o: MetaData.cc MetaData.h
	$(CC) -c MetaData.cc $(CFLAGS)

Model.o: Model.cc Model.h
	$(CC) -c Model.cc $(CFLAGS)

NaiveBayesClassifier.o: NaiveBayesClassifier.cc NaiveBayesClassifier.h
	$(CC) -c NaiveBayesClassifier.cc $(CFLAGS)

Neighborhood.o: Neighborhood.cc Neighborhood.h
	$(CC) -c Neighborhood.cc $(CFLAGS)

Parser.o: Parser.cc Parser.h
	$(CC) -c Parser.cc $(CFLAGS)

Point3D.o: Point3D.cc Point3D.h
	$(CC) -c Point3D.cc $(CFLAGS)

Points.o: Points.cc Points.h
	$(CC) -c Points.cc $(CFLAGS)

PropertyList.o: PropertyList.cc PropertyList.h
	$(CC) -c PropertyList.cc $(CFLAGS)

Protein.o: Protein.cc Protein.h
	$(CC) -c Protein.cc $(CFLAGS)

RankSum.o: RankSum.cc RankSum.h
	$(CC) -c RankSum.cc $(CFLAGS)

ResidueTemplate.o: ResidueTemplate.cc ResidueTemplate.h
	$(CC) -c ResidueTemplate.cc $(CFLAGS)

Utilities.o: Utilities.cc Utilities.h
	$(CC) -c Utilities.cc $(CFLAGS)

gzstream.o: gzstream.cc gzstream.h
	$(CC) -c gzstream.cc $(CFLAGS)

logutils.o: logutils.cc logutils.h
	$(CC) -c logutils.cc $(CFLAGS)

stl_types.o: stl_types.cc stl_types.h
	$(CC) -c stl_types.cc $(CFLAGS)

featurize: Point3D.o gzstream.o logutils.o stl_types.o Utilities.o ForceField.o MetaData.o Parser.o Points.o ResidueTemplate.o Atom.o PropertyList.o AtomProperties.o Atoms.o Environments.o Neighborhood.o Protein.o
	$(CC) -o featurize -O3 -D 'SVN_REV="1639M"' featurize.cc Point3D.o gzstream.o logutils.o stl_types.o Utilities.o ForceField.o MetaData.o Parser.o Points.o ResidueTemplate.o Atom.o PropertyList.o AtomProperties.o Atoms.o Environments.o Neighborhood.o Protein.o -lz

buildmodel: Point3D.o gzstream.o logutils.o stl_types.o NaiveBayesClassifier.o RankSum.o Utilities.o MetaData.o Parser.o Atom.o PropertyList.o AtomProperties.o Environments.o Model.o
	$(CC) -o buildmodel -O3 -D 'SVN_REV="1639M"' buildmodel.cc Point3D.o gzstream.o logutils.o stl_types.o NaiveBayesClassifier.o RankSum.o Utilities.o MetaData.o Parser.o Atom.o PropertyList.o AtomProperties.o Environments.o Model.o -lz

scoreit: Point3D.o gzstream.o logutils.o stl_types.o NaiveBayesClassifier.o RankSum.o Utilities.o MetaData.o Parser.o Atom.o PropertyList.o AtomProperties.o Environments.o Model.o
	$(CC) -o scoreit -O3 -D 'SVN_REV="1639M"' scoreit.cc Point3D.o gzstream.o logutils.o stl_types.o NaiveBayesClassifier.o RankSum.o Utilities.o MetaData.o Parser.o Atom.o PropertyList.o AtomProperties.o Environments.o Model.o -lz

