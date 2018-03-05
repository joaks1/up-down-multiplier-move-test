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
PARAM_INDICES = None
_NUM_MULT_TRIES = 0
_NUM_MULT_SUCCESSES = 0
HR_HACK = 0

def get_multiplier(scale=0.5):
    return math.exp(scale * (_RNG.random() - .5))


def multiply_move(num_up, num_down, state):
    global _NUM_MULT_SUCCESSES, _NUM_MULT_TRIES
    _NUM_MULT_TRIES += 1
    _RNG.shuffle(PARAM_INDICES)
    up_ind, down_ind = PARAM_INDICES[:num_up], PARAM_INDICES[num_up: num_up + num_down]
    m = get_multiplier()
    accept_prob = m ** (num_up - num_down + HR_HACK)
    if _RNG.random() > accept_prob:
        return
    ns = list(state)
    for i in up_ind:
        ns[i] *= m
        if ns[i] < 0.0 or ns[i] > 1.0:
            return
    for i in up_ind:
        ns[i] /= m
        if ns[i] < 0.0 or ns[i] > 1.0:
            return
    for i in range(len(state)):
        state[i] = ns[i]
    _NUM_MULT_SUCCESSES += 1


def slide_move(theta, window=0.2):
    offset = window * (_RNG.random() - .5)
    proposed = theta + offset
    while proposed < 0.0 or proposed > 1.0:
        if proposed < 0.0:
            proposed *= -1
        else:
            proposed = 2.0 - proposed
    return proposed


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
    global HR_HACK
    parser = argparse.ArgumentParser()

    parser.add_argument('-u', '--number-of-up-parameters',
                        action='store',
                        type=arg_is_nonnegative_int,
                        default=3,
                        help='Number of \"up\" parameters.')
    parser.add_argument('-d', '--number-of-down-parameters',
                        action='store',
                        type=arg_is_nonnegative_int,
                        default=1,
                        help='Number of \"down\" parameters.')
    parser.add_argument('-a', '--power-addend',
                        action='store',
                        type=float,
                        default=0.0,
                        help='Numer added to power term of hastings ratio.')
    parser.add_argument('-n', '--number-of-mcmc-generations',
                        action='store',
                        type=arg_is_positive_int,
                        default=1000000,
                        help='Number of MCMC iterations.')
    parser.add_argument('-m', '--num-multiplier-move-per-gen',
                        action='store',
                        type=arg_is_nonnegative_int,
                        default=1,
                        help='Number of multiplier updates run per generation.')
    parser.add_argument('-s', '--seed',
                        metavar='SEED',
                        action='store',
                        type=arg_is_positive_int,
                        help='Seed for random number generator.')
    if argv == sys.argv:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)
    HR_HACK = args.power_addend
    if not args.seed:
        args.seed = random.randint(1, 999999999)
    _LOG.debug('seed = {}'.format(args.seed))
    _RNG.seed(args.seed)
    nup, ndown = args.number_of_up_parameters, args.number_of_down_parameters
    nsteps = args.number_of_mcmc_generations
    run_mcmc(nup, ndown, nsteps, args.num_multiplier_move_per_gen)


def run_mcmc(nup, ndown, nsteps, num_multiplier_move_per_gen):
    global PARAM_INDICES
    m = ' nup={} ndown={} ngens={} n_mult_per_gen={} HR_HACK={}\n'
    sys.stderr.write(m.format(nup, ndown, nsteps, num_multiplier_move_per_gen, HR_HACK))
    np = nup + ndown
    assert np > 0
    state = [_RNG.random() for i in range(np)]
    sample_freq = 100
    summary_el = [0] * num_summary_bins
    summary_lists = [list(summary_el) for i in range(np)]
    PARAM_INDICES = range(np)
    for i in range(nsteps):
        if i > 0 and 10 * i % nsteps == 0:
            sys.stderr.write('  {}% done...\n'.format(100 * i / nsteps))
        for j in range(num_multiplier_move_per_gen):
            multiply_move(nup, ndown, state)
        for j in range(np):
            state[j] = slide_move(state[j])
        if i % sample_freq == 0:
            summarize_state(state, summary_lists)
    expected_per_bin = nsteps / (sample_freq * num_summary_bins)
    summarize_run(summary_lists, expected_per_bin)


if __name__ == "__main__":
    main_cli()
