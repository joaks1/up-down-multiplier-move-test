#! /usr/bin/env python

"""
A simple test of an updown scaling multivariate move.
"""
from __future__ import print_function, division

import argparse
import logging
import os
import random
import math
import sys

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
_LOG = logging.getLogger(os.path.basename(__file__))
_RNG = random.Random()

num_summary_bins = 10
bin_cutoff = 1.0 / num_summary_bins
_NUM_MULT_TRIES = 0
_NUM_MULT_SUCCESSES = 0

def get_multiplier(scale=5):
    return math.exp(scale * (_RNG.random() - .5))

def get_unif_multiplier(scale=0.12):
	return scale + (_RNG.random() * ((1.0 / scale) - scale))


def multiply_move(state,
        multiplier_function = get_multiplier,
        hr_hack = 1,
        scale = 5):
    global _NUM_MULT_SUCCESSES, _NUM_MULT_TRIES
    _NUM_MULT_TRIES += 1
    m = multiplier_function(scale)
    accept_prob = m ** (1 + hr_hack)
    if _RNG.random() > accept_prob:
        return state
    proposed_state = state
    proposed_state *= m
    if proposed_state < 0.0 or proposed_state > 1.0:
        return state
    _NUM_MULT_SUCCESSES += 1
    return proposed_state

def summarize_state(state, summary_list):
    for n, s in enumerate(state):
        bin_index = int(s / bin_cutoff)
        assert 0 <= bin_index < num_summary_bins
        summary_list[n][bin_index] += 1


def summarize_run_one_param(summary, exp):

    cum_chi_sq = 0.0
    print('   Ind    Obs    Exp  Chi-squared   Cum.ChiSq')
    for ind, obs in enumerate(summary):
        diff = obs - exp
        chi_sq = diff * diff / exp
        cum_chi_sq += chi_sq
        print('{:>6} {:>6}  {:>6}    {:>7}    {:>7}'.format(1 + ind, obs, exp,
                                                            '{:.2f}'.format(chi_sq),
                                                            '{:.2f}'.format(cum_chi_sq)))
def summarize_run(summary_lists, exp):
    for n, sl in enumerate(summary_lists):
        print('Param index {}'.format(n))
        summarize_run_one_param(sl, exp)
    print('the critical value for chi-square with df=10 and alpha = 0.05 is 18.3')
    print('Accepting {}% of multiplier moves'.format(100.0 * _NUM_MULT_SUCCESSES / _NUM_MULT_TRIES))


def arg_is_positive_int(i):
    try:
        assert int(i) > 0
    except:
        msg = '{0!r} is not a positive integer'.format(i)
        raise argparse.ArgumentTypeError(msg)
    return int(i)


def arg_is_nonnegative_int(i):
    try:
        assert int(i) >= 0
    except:
        msg = '{0!r} is not a nonnegative integer'.format(i)
        raise argparse.ArgumentTypeError(msg)
    return int(i)


def main_cli(argv=sys.argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--power-addend',
                        action='store',
                        type=int,
                        default=0,
                        help='Numer added to power term of hastings ratio.')
    parser.add_argument('-n', '--number-of-mcmc-generations',
                        action='store',
                        type=arg_is_positive_int,
                        default=1000000,
                        help='Number of MCMC iterations.')
    parser.add_argument('-u', '--uniform-multiplier-function',
                        action='store_true',
                        help='Use \"get_unif_multiplier\" function.')
    parser.add_argument('-s', '--seed',
                        metavar='SEED',
                        action='store',
                        type=arg_is_positive_int,
                        help='Seed for random number generator.')
    if argv == sys.argv:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)
    hr_hack = args.power_addend
    if not args.seed:
        args.seed = random.randint(1, 999999999)
    _LOG.debug('seed = {}'.format(args.seed))
    _RNG.seed(args.seed)
    nsteps = args.number_of_mcmc_generations
    mfunc = get_multiplier
    scale = 5
    if args.uniform_multiplier_function:
        mfunc = get_unif_multiplier
        scale = 0.3
        if hr_hack == -2:
            scale = 0.15
    run_mcmc(nsteps = nsteps,
            hr_hack = hr_hack,
            multiplier_function = mfunc,
            scale = scale)


def run_mcmc(nsteps, hr_hack, multiplier_function, scale):
    state = _RNG.random()
    sample_freq = 10
    summary_el = [0] * num_summary_bins
    summary_lists = [list(summary_el)]
    print("Multiplier function: {0}\nScale:{1}".format(
            multiplier_function, scale))

    for i in range(nsteps):
        if i > 0 and 10 * i % nsteps == 0:
            sys.stderr.write('  {}% done...\n'.format(100 * i / nsteps))
        state = multiply_move(state,
                multiplier_function = multiplier_function,
                hr_hack = hr_hack,
                scale = scale)
        if i % sample_freq == 0:
            summarize_state([state], summary_lists)
    expected_per_bin = nsteps / (sample_freq * num_summary_bins)
    summarize_run(summary_lists, expected_per_bin)


if __name__ == "__main__":
    main_cli()
