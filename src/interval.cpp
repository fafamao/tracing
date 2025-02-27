#include "interval.h"

const Interval Interval::empty    = Interval(+RAY_INFINITY, -RAY_INFINITY);
const Interval Interval::universe = Interval(-RAY_INFINITY, +RAY_INFINITY);