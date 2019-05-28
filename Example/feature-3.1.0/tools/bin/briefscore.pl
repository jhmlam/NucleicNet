#! /bin/bash

exec perl -ane 'print join("\t",@F[0..6]),"\n"' "$@"
