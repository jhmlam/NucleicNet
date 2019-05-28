// $Id: logutils.h 575 2009-11-06 08:40:32Z tblackstone $
// Copyright (c) 2004 Mike Liang. All rights reserved.

// Logging utility functions

#ifndef LOGUTILS_H
#define LOGUTILS_H

// ========================================================================
void setdebuglevel(int level);
void debug(int level,const char *fmt, ...);
void warning(const char *fmt, ...);
void error(const char *fmt, ...);
void fatal_error(const char *fmt, ...);

#endif
