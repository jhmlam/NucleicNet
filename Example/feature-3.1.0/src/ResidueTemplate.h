/* $Id: ResidueTemplate.h 1100 2011-09-12 05:23:32Z mikewong899 $ */
/* Copyright (c) 2002 Mike Liang.  All rights reserved. */

#ifndef RESIDUE_TEMPLATE_H
#define RESIDUE_TEMPLATE_H

#include "const.h"
#include "stl_types.h"
#include "Utilities.h"
#include <fstream>
#include <string>

using std::ostream;
using std::ifstream;
using std::endl;
using std::string;

// ========================================================================
/**
 * @brief Reads the data from the AMBER residue template file which correlates
 * PDB atom names to AMBER atom names.
 * @ingroup protein_module
 *
 * Atoms, especially hydrogen atoms, tend to have different naming conventions
 * between PDB and AMBER. This makes it difficult to incorporate the AMBER atom
 * type and partial charge information to PDB structures. A ResidueTemplate
 * enables AMBER data integration with PDB structures by correlating PDB atom
 * names to AMBER atom names; these correlations are called @em aliases.  The atom
 * names are originally read from PDBs (i.e. columns 13-16; e.g.  "CD" for
 * delta carbon) and converted to AMBER names (e.g. "CD1"). 
 *
 * The default residue template file is "residue_template.dat". 
 **/
class ResidueTemplate {
	public:
		typedef map<string, string> AtomAliases;

		/**
		 * @brief Correlates atom aliases
		 **/
		class Terminus : public map<string, AtomAliases *> {
			public:
				~Terminus();
		};

		ResidueTemplate( string filename );
		Terminus normal;
		Terminus n_terminal;
		Terminus c_terminal;
		void load( string filename );
};

ostream& operator<< (ostream& outs, const ResidueTemplate::AtomAliases& aam);
ostream& operator<< (ostream& outs, const ResidueTemplate::Terminus& tmpl);
ostream& operator<< (ostream& outs, const ResidueTemplate& rtm);

#endif
