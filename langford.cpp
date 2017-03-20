// Copyright 2017 Boris Dimitrov, Portola Valley, CA 94028.
// Questions? Contact http://www.facebook.com/boris
//
//
// 1. LANGFORD SEQUENCES AND PLANAR LANGFORD SEQUENCES
//
// This program counts all permutations of the sequence 1, 1, 2, 2, 3, 3, ..., n, n
// in which the two occurrences of each m are separated by precisely m other numbers.
// Those are called Langford sequences.
//
// This program also counts the Langford sequences for which lines connecting all
// pairs can be drawn on the page without crossing.  Those are called Planar
// Langford sequences.
//
// See http://www.dialectrix.com/langford.html ("Planar Solutions") or Knuth volume 4a
// page 3.  Todo: Provide better Knuth reference.
//
//
// 2. Compiling on Mac or Linux:
//
//     g++ -O3 -DNDEBUG -std=c++11 -o langford langford.cpp -lpthread
//
// The resulting executable will be named "langford".
//
//
// 3. HIGHER DIMENSIONS
//
// Consider a more general problem, where the numbers are placed on the real axis
// as before, but the surface on which non-crossing connections are drawn is the
// union of D half-planes that meet on the real axis.  D can be any positive integer.
//
// For D=2, the two half planes form a single plane, yielding solutions to the
// original Planar Langford problem.
//
// For D=1, removing the planarity constraint, we obtain a solution to the
// ordinary Langford problem.
//
//
// 4.  CRUX OF THE PLANAR LANGFORD ALGORITHM
//
// We place m in increasing order, taking advantage of the property that all numbers
// bracketed by the pair (m, m) in the dimension of that pair must be less than m,
// or else a disallowed crossing would occur.  After pairs (1, 1), (2, 2), ..., (m, m)
// have been placed, if any of the unfilled positions bracketed by those pairs happen
// to be bracketed by some pair in every dimension, then the partial sequence cannot
// be completed to a planar solution, and the algorithm can backtrack.
//
//
// 5. DEDUPING SYMMETRY TWINS IN 2D
//
// Any planar Lanford sequence can be reversed right-to-left and top-to-bottom
// to produce another;  we dedup these by ensuring the pair (1, 1) is placed
// in the left half (position <= n-2) and connected from above.
//
//                                  symmetry axis
//                                        |
//         0  1  2  3  ...  n-3  n-2  n-1 |  n   n+1  n+2  ...       2*n-1
//         _______________________________________________________________
//                                        |
//         x  x  x  x  ...   1    x    1  |  x    x    x  ...       case 1
//                                        |
//         x  x  x  x  ...   x    1    x  |  1    x    x  ...       case 2
//                                        |
//         x  x  x  x  ...   x    x    1  |  x    1    x  ...       case 3
//                                        |
//         x  x  x  x  ...   x    x    x  |  1    x    1  ...       case 4
//
// We omit case 3 because it is the twin of case 2 under left-to-right reversal.
// Case 4 is similarly the twin of case 1; and so on.
//
// Arrangements with the same number sequence but different above/below choices
// are also considered duplicates of one another.  For the Planar Langford problem,
// the result set is small enough that we can generate all arrangements, then sort
// and filter out dimensional duplicates.
//
// We could also modify the generation algorithm so that dimensional twins are
// generated consecutively, requiring no storage and no sorting to filter out;
// but that is incompatible with some other optimizations.
//

#include <assert.h>
#include <array>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <mutex>
using namespace std;

// Setting kD >= 2 will solve the Planar Langford problem in 2 or more dimensions.
// Setting kD == 1 will solve the ordinary Langford problem.
constexpr int kD = 2;

// Higher dimensions make sense only with planarity constraints, and vice versa.
constexpr bool kPlanar = (kD > 1);

static_assert(kD >= 1, "let's be reasonable");

// When true, each unique sequence will be printed.
constexpr bool kPrintSolutions = false;

// To avoid integer overflow, n should not exceed this constant.
constexpr int kN = 30;

// 2^n-1 works best for the silly modulus hash thingy
constexpr int kMaxThreads = 1023;

// Placing "m" at positions "k1" and "k1 + m + 1" in dimension "d" is represented
// by letting "pos[m-1] = 1<<k1" and "dim[m-1] = d".  Our algorithm constructs
// a solution in order of increasing m, so this works a bit like a stack.
template <int n>
using Positions = array<int64_t, n>;

template <int n>
using Dimensions = array<int8_t, n>;

// here the positions are ordinary k1, not 1<<k1
template <int n>
using CompressedPositions = array<int8_t, n>;

// For d<kD, bit k of avail[d] is set to 1 when position k in
// the partially constructed sequence is unobstructed in dimension d
// (this means -- no m placed in pos k, and no pair of dim d obstructs k)
// For d=kD, avail[kD] has 0 in the positions already filled.
using Availability = array<int64_t, kD+1>;

template <int n>
void append_sequence(vector<CompressedPositions<n>>& results, const Positions<n> &pos, mutex& mtx) {
    CompressedPositions<n> cp;
    for (int i=0; i<n; ++i) {
        cp[i] = __builtin_ffsll(pos[i]) - 1;
    }
    mtx.lock();
    results.push_back(cp);
    mtx.unlock();
}

// Generate all planar sequences of size 2*n in which the pair (1, 1) resides at positions
// p1 and p1 + 2 in dimension 0.  Typically called with 0 <= p1 <= n-2.
template <int n>
int64_t dfs(vector<CompressedPositions<n>>& results, const int thread_id, mutex& mtx) {
    int64_t cnt = 0;
    // Stacks representing the partial solution.
    // For d<kD, avail[m][d] represents the remaining open positions in
    // dimension d after pairs (1, 1)  (2, 2)  ...  (m, m) have been placed;
    // For d=kD, avail[m][kD] has 0 in the positions filled by 1, 2, ..., m.
    // Note pos[m-1] and avail[m] correspond to the same m.
    Availability avail[n + 1];
    Positions<n> pos;
    Dimensions<n> dim;
    // Stacks for depth-first traversal of the search tree
    constexpr int kMaxDepth = n * kD;
    int64_t stack_possible_places[kMaxDepth];
    int16_t stack_m[kMaxDepth];
    int16_t stack_d[kMaxDepth];
    int top = 0;
    // some useful bit vector constants
    const int64_t lsb = 1;
    const int64_t msb = lsb << (2 * n - 1);
    const int64_t full = msb | (msb - 1);
    // Initially all positions are available.
    for (int d=0; d<=kD; ++d) {
        avail[0][d] = full;
    }
    // a helper function to place m in positions k1 and k1 + m + 1 of dimension d;
    // it returns false when the placement is unsuccessful, i.e., when it obstructs
    // some not-yet-filled position, and threfore cannot be completed to a solution
    auto place = [&avail, &pos, &dim](const int m, const int64_t pos_k1, const int d) -> bool {
        assert(m >= 1);
        assert(m <= n);
        pos[m - 1] = pos_k1;
        dim[m - 1] = d;
        // Here is the crux of the difference between Planar and ordinary Langford.
        // In ordinary Langford, only k1 and k2 are blocked (intermediate positions remain open).
        const int64_t block_k1_k2 = ~(pos_k1 | (pos_k1 << (m + 1)));
        #pragma unroll
        for (int ddd=0;  ddd<=kD;  ++ddd) {
            avail[m][ddd] = avail[m-1][ddd] & block_k1_k2;
        }
        // In Planar Langford, we block all intermediate positions k1 .. k2 in dimension d,
        // in addition to blocking k1 and k2 in every other dimension.  This can cause
        // irreparable obstructions and thus an opportunity to backtrack much sooner.
        if (kPlanar) {
            // The constant block_k1_thru_k2 has bits k1 ... k1 + m + 1 set to 0, all others to 1.
            // TODO:  THIS WILL OVERFLOW FOR n=31, FIX (perhaps using uint64_t will suffice)
            const int64_t block_k1_thru_k2 = ~((pos_k1 << (m + 2)) - pos_k1);
            avail[m][d] = avail[m-1][d] & block_k1_thru_k2;
            // This is a crucial optimization.  If placing (m, m) makes it completely impossible
            // to fill some still-vacant position between k1 and k2 then we must backtrack.
            auto still_unobstructed = avail[m][0];
            #pragma unroll
            for (int ddd=1; ddd<kD; ++ddd) {
                still_unobstructed |= avail[m][ddd];
            }
            const auto still_unfilled = avail[m][kD];
            if ((still_unfilled & still_unobstructed) != still_unfilled) {
                // this placement obstructs some not-yet-filled position;
                // therefore we must bakctrack
                return false;
            }
        }
        // placement is valid and can possibly be extended to a complete solution
        return true;
    };
    // a helper function to compute all positions where m can be placed, for each dimension,
    // and push those on the stack
    auto multipush = [&avail, &stack_possible_places, &stack_m, &stack_d, &top](const int m) {
        assert(m >= 2);
        #pragma unroll
        for (int ddd=0; ddd<kD; ++ddd) {
            const int64_t possible_places_for_m = avail[m-1][ddd] & (avail[m-1][ddd] >> (m + 1));
            if (possible_places_for_m) {
                assert(top < kMaxDepth);
                stack_possible_places[top] = possible_places_for_m;
                stack_m[top] = m;
                stack_d[top] = ddd;
                ++top;
            }
        }
    };
    // the bounds of this for-loop are designed to dedup 2D twins
    for (int p1=0;  p1 <= n-2;  ++p1) {
        // Place 1 in positions p1 and p1 + 2 of dimension 0
        place(1, lsb << p1, 0);
        // Push all possible places for m=2
        multipush(2);
        // Depth first search
        while (top) {
            const int peek = top - 1;
            const int d = stack_d[peek];
            const int m = stack_m[peek];
            int64_t possible_places = stack_possible_places[peek];
            // extract the lowest bit that is set to 1 in possible_places
            const int64_t pos_k = possible_places & ~(possible_places - lsb);
            // pop that bit from the stack
            possible_places ^= pos_k;
            if (possible_places) {
                stack_possible_places[peek] = possible_places;
            } else {
                top = peek;
            }
            // place m in that position
            if (place(m, pos_k, d)) {
                if (m == n) {
                    // Found a solution.  Count it.
                    ++cnt;
                    if (kPlanar || kPrintSolutions) {
                        // In the Planar case, "cnt" may overcount dimensional twins.
                        // We store all generated sequences, then sort and count unique at the end.
                        append_sequence<n>(results, pos, mtx);
                    }
                } else {
                    // A simple way to divide the work equally across kMaxThreads.
                    // The Mersenne primes help even the load.
                    // Carelessness here can lead to overflow issues.
                    if (kMaxThreads == 1 ||
                        m != 3 ||
                        thread_id == ((131071 * possible_places + 255 * p1 + 511 * __builtin_ffsll(pos[m-1]) + 8191 * __builtin_ffsll(pos[m-2])) % kMaxThreads))
                    {
                        // Push all possible places for m + 1
                        multipush(m + 1);
                    }
                }
            }
        }
    }
    return cnt;
}

template <int n>
void print(const CompressedPositions<n>& pos);

template <int n>
int64_t unique_count(vector<CompressedPositions<n>> &results) {
    int64_t total = results.size();
    int64_t unique = total;
    sort(results.begin(), results.end());
    if (kPrintSolutions && total) {
        print<n>(results[0]);
    }
    for (int i=1; i<total; ++i) {
        if (results[i] == results[i-1]) {
            --unique;
        } else if (kPrintSolutions) {
            print<n>(results[i]);
        }
    }
    return unique;
}

// This is the true main function.  It counts all solution sequences in kD dimensions
// for the given n.
template <int n>
int multi_dfs() {
    if (n % 4 == 1 || n % 4 == 2) {
        return 0;
    }
    if (n <= 0 || n > kN) {
        return -1;
    }
    vector<CompressedPositions<n>> results;
    int num_running = kMaxThreads;
    mutex mtx;
    int64_t cnt = 0;
    for (int thread_id=0;  thread_id < kMaxThreads;  ++thread_id) {
        auto thread_func = [&num_running, &mtx, &cnt, &results](int thread_id) {
            int64_t c = dfs<n>(results, thread_id, mtx);
            mtx.lock();
            --num_running;
            cnt += c;
            mtx.unlock();
        };
        thread(thread_func, thread_id).detach();
    }
    bool done = false;
    while (!done) {
        this_thread::sleep_for(chrono::milliseconds(50));
        mtx.lock();
        done = (num_running == 0);
        mtx.unlock();
    }
    if (kPlanar || kPrintSolutions) {
        cnt = unique_count<n>(results);
    }
    return cnt;
}

// ----------------------------- crux of solution ends here -------------------------------
// The rest is boring utilities for pretty printing, argument parsing, validation, etc.
// ----------------------------------------------------------------------------------------

// Return number of milliseconds elapsed since Jan 1, 1970 00:00 GMT.
long unixtime() {
    // There is the unix way, the navy way, and the C++11 way... apparently.
    using namespace chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    // Using steady_clock instead of system_clock above produces comically incorrect results.
    // Probably steady_clock has the wrong epoch start.
}

void init_known_results(int64_t (&known_results)[64]) {
    for (int i=0;  i<64; ++i) {
        if (i % 4 != 3 && i % 4 != 0) {
            known_results[i] = 0;
        } else {
            // unknown
            known_results[i] = -1;
        }
    }
    if (kD == 1) {
        // Ordinary Langford.
        assert(!(kPlanar));
        known_results[3]  = 1;
        known_results[4]  = 1;
        known_results[7]  = 26;
        known_results[8]  = 150;
        known_results[11] = 17792;
        known_results[12] = 108144;
        known_results[15] = 39809640ll;
        known_results[16] = 326721800ll;
        known_results[19] = 256814891280ll;
        known_results[20] = 2636337861200ll;
        known_results[23] = 3799455942515488ll;
        known_results[24] = 46845158056515936ll;
        // beyond 24, count does not fit in 64 bits...
    }
    if (kD == 2) {
        // Published by Donald E Knuth in 2007.
        assert(kPlanar);
        known_results[3]  = 1;
        known_results[4]  = 0;
        known_results[7]  = 0;
        known_results[8]  = 4;
        known_results[11] = 16;
        known_results[12] = 40;
        known_results[15] = 194;
        known_results[16] = 274;
        known_results[19] = 2384;
        known_results[20] = 4719;
        known_results[23] = 31856;
        known_results[24] = 62124;
        known_results[27] = 426502;
        known_results[28] = 817717;
    }
}

template <int n>
void print(const CompressedPositions<n>& pos) {
    cout << unixtime() << " Sequence ";
    int s[2 * n];
    static_assert(0 < n, "hmm");
    for (int i=0; i<2*n; ++i) {
        s[i] = -1;
    }
    for (int m=1;  m<=n;  ++m) {
        int k1 = pos[m-1];
        int k2 = k1 + m + 1;
        assert(0 <= k1);
        assert(k2 < 2*n);
        assert(s[k1] == -1);
        assert(s[k2] == -1);
        s[k1] = s[k2] = m;
    }
    for (int i=0;  i<2*n;  ++i) {
        int m = s[i];
        assert(0 <= m);
        assert(m <= n);
        cout << setw(3) << m;
    }
    cout << "\n";
}

template <int n>
void run(const int64_t* known_results) {
    auto t_start = unixtime();
    cout << t_start << " Solving for n = " << n << ", kD = " << kD << ", kPlanar = " << kPlanar << "\n";
    cout << flush;
    int cnt = multi_dfs<n>();
    auto t_end = unixtime();
    cout << t_end << " Result " << cnt << " for n = " << n << ", kD = " << kD;
    if (cnt == -1 || n < 0) {
        cout << " is UNDEFINED";
    } else if (n >= 64 || known_results[n] == -1) {
        cout << " is NEW";
    } else if (known_results[n] == cnt) {
        cout << " MATCHES published result";
    } else {
        cout << " MISMATCHES published result " << known_results[n];
    }
    cout << " and took " << (t_end - t_start) << " milliseconds.\n";
    cout << flush;
}

int main(int argc, char **argv) {
    int64_t known_results[64];
    init_known_results(known_results);
    run<3>(known_results);
    run<4>(known_results);
    run<7>(known_results);
    run<8>(known_results);
    run<11>(known_results);
    run<12>(known_results);
    run<15>(known_results);
    run<16>(known_results);
    run<19>(known_results);
    run<20>(known_results);
    run<23>(known_results);
    run<24>(known_results);
    run<27>(known_results);
    run<28>(known_results);
    run<31>(known_results);
    return 0;
}
