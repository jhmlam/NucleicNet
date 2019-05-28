#ifdef CPP03_SUPPORT
#include <tr1/unordered_map>
using std::tr1::unordered_map;
#else
#ifdef CPP11_SUPPORT
#include <unordered_map>
using std::unordered_map;
#else
#include <map>
using std::map;
#define unordered_map map
#endif
#endif
