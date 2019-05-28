.PHONY: build build-fast build-fast03 build-standard clean install

MAKE=make

all: build
	mkdir -p bin
	cp src/featurize src/buildmodel src/scoreit bin
	@echo ------------------------------------------------------------------------------
	@echo FEATURE has successfully been built and tested.
	@echo
	@echo To install FEATURE to /usr/local/feature, type:
	@echo '  sudo make install'
	@echo ------------------------------------------------------------------------------

clean:
	rm -rf bin
	cd src && $(MAKE) clean
	cd tests && $(MAKE) clean

build:
	- \
	$(MAKE) build-fast     || \
	$(MAKE) build-fast03   || \
	$(MAKE) build-standard

build-fast:
	@echo ==============================================================================
	@echo Attempting to build the fast version of FEATURE \(requires C++11\)
	@echo ==============================================================================
	$(MAKE) clean
	cd src && $(MAKE) fast
	@echo Testing Fast C++11 FEATURE
	cd tests && $(MAKE)
	@echo ==============================================================================
	@echo SUCCESS! Fast FEATURE \(C++11\) has been compiled and tested.
	@echo ==============================================================================

build-fast03:
	@echo Fast C++11 FEATURE didn\'t work. 
	@echo ==============================================================================
	@echo Attempting to build the fast version of FEATURE \(requires C++03\)
	@echo ==============================================================================
	$(MAKE) clean
	cd src && $(MAKE) fast03
	@echo Testing Fast C++03 FEATURE
	cd tests && $(MAKE)
	@echo ==============================================================================
	@echo SUCCESS! Fast FEATURE \(C++03\) has been compiled and tested.
	@echo ==============================================================================

build-standard:
	@echo Fast C++03 FEATURE didn\'t work. 
	@echo ==============================================================================
	@echo Building the standard version of FEATURE
	@echo ==============================================================================
	$(MAKE) clean
	cd src && $(MAKE) standard 
	@echo Testing standard FEATURE
	cd tests && $(MAKE)
	@echo ==============================================================================
	@echo SUCCESS! Standard FEATURE has been compiled and tested.
	@echo ==============================================================================

install:
	@echo ==============================================================================
	@echo Installing FEATURE to /usr/local/feature
	@echo ==============================================================================
	mkdir -p /usr/local/feature
	cp -Rf bin data tools README /usr/local/feature
	ln -sf /usr/local/feature/bin/featurize /usr/local/bin/featurize
	ln -sf /usr/local/feature/bin/buildmodel /usr/local/bin/buildmodel
	ln -sf /usr/local/feature/bin/scoreit /usr/local/bin/scoreit
	ln -sf /usr/local/feature/tools/bin/atomselector.py /usr/local/bin/atomselector.py
	@echo ==============================================================================
	@echo FEATURE installed at /usr/local/feature
	@echo ==============================================================================
	@echo To use FEATURE, don\'t forget to edit your environment variables:
	@echo PATH, PDB_DIR, DSSP_DIR, FEATURE_DIR, and PYTHONPATH
	@echo Please see Chapter 2 of the FEATURE User\'s Manual for more information
	@echo ------------------------------------------------------------------------------

