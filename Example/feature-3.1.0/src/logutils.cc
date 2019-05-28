// $Id: logutils.cc 575 2009-11-06 08:40:32Z tblackstone $
// Copyright (c) 2004 Mike Liang. All rights reserved.

// Logging utility functions

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include "logutils.h"

static int debuglevel = 0;
static FILE *debugstream = stderr;
static FILE *errorstream = stderr;
static FILE *warnstream = stderr;

// ========================================================================
void setdebuglevel(int level) {
    debuglevel = level;
}

// ========================================================================
void debug(int level, const char *fmt, ...) {
    va_list argp;

    if (debuglevel < level)
        return;

    va_start(argp, fmt);
    vfprintf(debugstream, fmt, argp);
    va_end(argp);
    fflush(debugstream);
}

// ========================================================================
void warning(const char *fmt, ...) {
    va_list argp;
    fprintf(warnstream, "Warning: ");
    va_start(argp, fmt);
    vfprintf(warnstream, fmt, argp);
    va_end(argp);
}

// ========================================================================
void error(const char *fmt, ...) {
    va_list argp;
    fprintf(errorstream, "Error: ");
    va_start(argp, fmt);
    vfprintf(errorstream, fmt, argp);
    va_end(argp);
}

// ========================================================================
void fatal_error(const char *fmt, ...) {
    va_list argp;
    fprintf(errorstream, "Fatal Error: ");
    va_start(argp, fmt);
    vfprintf(errorstream, fmt, argp);
    va_end(argp);
    exit(-1);
}
