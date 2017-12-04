#include "ee193_utils.hxx"

std::chrono::time_point<std::chrono::high_resolution_clock> start_time () {
    return (std::chrono::high_resolution_clock::now());
}

long int delta_usec
	(std::chrono::time_point<std::chrono::high_resolution_clock> start) {
    std::chrono::time_point<std::chrono::high_resolution_clock>
		end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<1, 1000000> > elapsed_us=end-start;
    int ticks_us = elapsed_us.count();
    return (ticks_us);
}

/* Here's an overview of how the C++ timing facility works.
 * A *duration* is a span of time. It is a number of ticks, where a tick can
 *	be, e.g., ms or us. The number of ticks is usually int or float; the
 *	tick length gets remembered as a std::ratio object that is implicit in
 *	the type name (but takes no space).
 *	* There are predefined duration types such as crono::milliseconds. They
 *	  store the # of ticks as an integer.
 *	* You can convert duration types to each other;
 *		chrono::seconds sec(2); chrono::milliseconds millie(sec);
 *		Then millie.count() -> 2000.
 *	* You cannot convert from ms to seconds, since integer division loses
 *	  precision. To convert from ms to seconds, you would to convert to
 *	  duration <double, ratio<1,1> > instead.
 *	* We declare our own microsecond duration type above:
 *		std::chrono::duration<double, std::ratio<1, 1000000> >
 *	  If we had simply used chrono::microseconds, it would be based on int
 *	  rather than double, and we would not be able to measure anything less
 *	  than one microsecond. Thus, we rolled our own.
 * A *clock* has a starting point and a tick rate (i.e., number of ticks/second)
 * A *time_point* is essentially a duration that is relative to a particular
 *	clock's beginning of time. However, it is its own type; the particular
 *	clock it uses is encoded in the type.
 *
 */
