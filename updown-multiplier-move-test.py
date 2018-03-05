#! /usr/bin/env python

"""
A simple test of an updown scaling multivariate move.
"""

import sys
import os
import math
import random
import argparse
import unittest
import logging
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
_LOG = logging.getLogger(os.path.basename(__file__))
_RNG = random.Random()
_INDENT = 4


class ScaleOperator(object):
    def __init__(self,
            scale = 0.1,
            auto_optimize = True,
            auto_optimize_delay = 1000):
        self.scale = scale
        self.auto_optimize_delay = auto_optimize_delay
        self.number_of_proposals = 0
        self.number_of_acceptions = 0
        self.number_of_proposals_for_tuning = 0
        self.target_acceptance_probability = 0.44
        self.auto_optimize = auto_optimize

    def get_acceptance_rate(self):
        if self.number_of_proposals < 1:
            return None
        return self.number_of_acceptions / float(self.number_of_proposals)

    def set_scale(self, scale):
        assert scale > 0.0
        self.scale = scale

    def get_multiplier(self, rng):
        return math.exp(self.scale * ((2.0 * rng.random()) - 1.0))

    def propose(self, value, rng):
        m = self.get_multiplier(rng)
        new_value = value * m
        return new_value, math.log(m)

    def reject(self):
        self.number_of_proposals += 1
        if self.number_of_proposals >= self.auto_optimize_delay:
            self.number_of_proposals_for_tuning += 1

    def accept(self):
        self.number_of_proposals += 1
        self.number_of_acceptions += 1
        if self.number_of_proposals >= self.auto_optimize_delay:
            self.number_of_proposals_for_tuning += 1

    def optimize(self, log_alpha):
        if (not self.auto_optimize) or (
                self.number_of_proposals < self.auto_optimize_delay):
            return
        delta = self.calc_delta(log_alpha)
        delta += math.log(self.scale)
        try:
            self.set_scale(math.exp(delta))
        except OverflowError as e:
            _LOG.warn(
                    "Overflow at math.exp(delta = {0}). "
                    "Skipping optimization".format(
                        delta))
            return

    def calc_delta(self, log_alpha):
        if (not self.auto_optimize) or (
                self.number_of_proposals < self.auto_optimize_delay):
            return 0.0
        target = self.target_acceptance_probability
        count = self.number_of_proposals_for_tuning + 1.0
        try:
            delta_p = ((1.0 / count) * (math.exp(min(log_alpha, 0.0)) - target))
        except OverflowError as e:
            _LOG.warn(
                    "Overflow at math.exp(log_alpha = {0}). "
                    "Skipping optimization".format(
                        log_alpha))
            return 0.0
        mx = sys.float_info.max
        if ((delta_p > -mx) and (delta_p < mx)):
            return delta_p
        return 0.0


class UpDownScaleOperator(ScaleOperator):
    def __init__(self,
            scale = 0.1,
            auto_optimize = True,
            auto_optimize_delay = 1000,
            power_addend = 0.0):
        ScaleOperator.__init__(self,
                scale = scale,
                auto_optimize = auto_optimize,
                auto_optimize_delay = auto_optimize_delay)
        self.power_addend = power_addend

    def get_multiplier(self, rng):
        return math.exp(self.scale * ((2.0 * rng.random()) - 1.0))

    def propose(self, up_values, down_values, rng):
        m = self.get_multiplier(rng)
        new_up_values = [u * m for u in up_values]
        new_down_values = [d * (1.0 / m) for d in down_values]
        power_term = len(new_up_values) - len(new_down_values)
        power_term += self.power_addend
        ln_hastings_ratio = math.log(m) * power_term 
        return new_up_values, new_down_values, ln_hastings_ratio


class Parameter(object):
    """
    A class for a simple parameter with a Uniform(0, 1) prior.

    >>> p = Parameter(seed = 111)
    >>> samples = []
    >>> for i in range(100000):
    ...     p.perform_scale_update()
    ...     if ((i + 1) % 10) == 0:
    ...         samples.append(p.value)
    >>> len(samples)
    10000
    >>> mn = sum(samples) / len(samples)
    >>> abs(mn - 0.5) < 0.01
    True
    """

    def __init__(self, seed):
        self._seed = seed
        self._rng = random.Random()
        self._rng.seed(self._seed)
        self._value = self._rng.random()
        self._stored_value = self._value
        self._scale_operator = ScaleOperator()
        self._samples = []
        self.summary = SampleSummarizer()

    def store_value(self):
        self._stored_value = self._value

    def restore_value(self):
        self._value = self._stored_value

    def set_value(self, v):
        self.store_value()
        ln_prior_ratio = 0.0
        if (v < 0.0) or (v >= 1.0):
            ln_prior_ratio = float("-inf")
        self._value = v
        return ln_prior_ratio

    def _get_value(self):
        return self._value

    value = property(_get_value)

    def _get_stored_value(self):
        return self._stored_value

    stored_value = property(_get_stored_value)

    def perform_scale_update(self):
        v = self._value
        new_v, ln_hastings_ratio = self._scale_operator.propose(v, self._rng)
        if (new_v < 0.0) or (new_v >= 1.0):
            self._scale_operator.reject()
            return
        ln_prior_ratio = self.set_value(new_v)
        acceptance_prob = ln_prior_ratio + ln_hastings_ratio
        u = self._rng.random()
        if u < math.exp(acceptance_prob):
            self._scale_operator.accept()
        else:
            self._scale_operator.reject()
            self.restore_value()
        self._scale_operator.optimize(acceptance_prob)

    def sample(self):
        self._samples.append(self._value)
        self.summary.add_sample(self._value)

    def _get_samples(self):
        return self._samples

    samples = property(_get_samples)

    def summarize(self, indent_level = 0):
        indent = " " * _INDENT
        margin = indent * indent_level
        s = StringIO()
        s.write("{0}mean = {1}\n".format(margin, self.summary.mean))
        s.write("{0}expected mean = 0.5\n".format(margin))
        s.write("{0}variance = {1}\n".format(margin, self.summary.variance))
        s.write("{0}expected variance = {1}\n".format(margin, 1.0 / 12.0))
        s.write("{0}univariate scaler summary:\n".format(margin))
        s.write("{0}{1}number of acceptions = {2}\n".format(margin, indent,
                self._scale_operator.number_of_acceptions))
        s.write("{0}{1}number of proposals = {2}\n".format(margin, indent,
                self._scale_operator.number_of_proposals))
        s.write("{0}{1}acceptance rate = {2}\n".format(margin, indent,
                self._scale_operator.get_acceptance_rate()))
        return s.getvalue()


class Model(object):
    def __init__(self, seed,
            number_of_up_parameters = 5,
            number_of_down_parameters = 2,
            power_addend = 0.0,
            up_down_move_on = True):
        self._seed = seed
        self._rng = random.Random()
        self._rng.seed(self._seed)
        self.up_parameters = []
        for i in range(number_of_up_parameters):
            self.up_parameters.append(Parameter(self.get_random_int()))
        self.down_parameters = []
        for i in range(number_of_down_parameters):
            self.down_parameters.append(Parameter(self.get_random_int()))
        self._up_down_operator = UpDownScaleOperator(
                power_addend = power_addend)
        self.up_down_move_on = up_down_move_on

    def get_random_int(self):
        return self._rng.randint(1, 9999999999)

    def perform_univariate_scale_updates(self):
        for p in self.up_parameters:
            p.perform_scale_update()
        for p in self.down_parameters:
            p.perform_scale_update()

    def perform_up_down_update(self):
        if self.up_down_move_on:
            ups = [p.value for p in self.up_parameters]
            downs = [p.value for p in self.down_parameters]
            new_ups, new_downs, ln_hastings_ratio = self._up_down_operator.propose(
                    ups,
                    downs,
                    self._rng)
            for new_v in new_ups:
                if (new_v < 0.0) or (new_v >= 1.0):
                    self._up_down_operator.reject()
                    return
            for new_v in new_downs:
                if (new_v < 0.0) or (new_v >= 1.0):
                    self._up_down_operator.reject()
                    return
            ln_prior_ratio = 0.0
            for i, new_v in enumerate(new_ups):
                ln_prior_ratio += self.up_parameters[i].set_value(new_v)
            for i, new_v in enumerate(new_downs):
                ln_prior_ratio += self.down_parameters[i].set_value(new_v)
            acceptance_prob = ln_prior_ratio + ln_hastings_ratio
            u = self._rng.random()
            if u < math.exp(acceptance_prob):
                self._up_down_operator.accept()
            else:
                self._up_down_operator.reject()
                for p in self.up_parameters:
                    p.restore_value()
                for p in self.down_parameters:
                    p.restore_value()
            self._up_down_operator.optimize(acceptance_prob)
        self.perform_univariate_scale_updates()

    def _get_number_of_parameters(self):
        return len(self.up_parameters) + len(self.down_parameters)

    number_of_parameters = property(_get_number_of_parameters)

    def sample(self):
        for p in self.up_parameters:
            p.sample()
        for p in self.down_parameters:
            p.sample()

    def mcmc(self,
            number_of_generations = 100000,
            sample_frequency = 10,
            burnin = 10):
        ignored_samples = 0
        for i in range(number_of_generations):
            self.perform_up_down_update()
            if ((i + 1) % sample_frequency) == 0:
                if ignored_samples < burnin:
                    ignored_samples += 1
                    continue
                self.sample()

    def summarize(self, indent_level = 0):
        indent = " " * _INDENT
        margin = indent * indent_level
        s = StringIO()
        s.write("{0}multivariate scaler summary:\n".format(margin))
        s.write("{0}{1}number of acceptions = {2}\n".format(margin, indent,
                self._up_down_operator.number_of_acceptions))
        s.write("{0}{1}number of proposals = {2}\n".format(margin, indent,
                self._up_down_operator.number_of_proposals))
        s.write("{0}{1}acceptance rate = {2}\n".format(margin, indent,
                self._up_down_operator.get_acceptance_rate()))
        s.write("{0}up parameter summaries:\n".format(margin))
        for p in self.up_parameters:
            s.write("{0}".format(p.summarize(indent_level = 1)))
        s.write("{0}down parameter summaries:\n".format(margin))
        for p in self.down_parameters:
            s.write("{0}".format(p.summarize(indent_level = 1)))
        return s.getvalue()


class SampleSummarizer(object):
    count = 0
    def __init__(self, samples = None, tag = ""):
        self.__class__.count += 1
        self.name = self.__class__.__name__ + '-' + str(self.count)
        self.tag = tag
        self._min = None
        self._max = None
        self._n = 0
        self._mean = 0.0
        self._sum_devs_2 = 0.0
        self._sum_devs_3 = 0.0
        self._sum_devs_4 = 0.0
        if samples:
            self.update_samples(samples)
    
    def add_sample(self, x):
        n = self._n + 1
        d = x - self._mean
        d_n = d / n
        d_n2 = d_n * d_n
        self._mean = self._mean + d_n
        first_term =  d * d_n * self._n
        self._sum_devs_4 += (first_term * d_n2 * ((n * n) - (3 * n) + 3)) + \
                (6 * d_n2 * self._sum_devs_2) - (4 * d_n * self._sum_devs_3)
        self._sum_devs_3 += (first_term * d_n * (n - 2)) - \
                (3 * d_n * self._sum_devs_2)
        self._sum_devs_2 += first_term
        self._n = n
        if not self._min:
            self._min = x
        elif x < self._min:
            self._min = x
        if not self._max:
            self._max = x
        elif x > self._max:
            self._max = x

    def update_samples(self, x_iter):
        for x in x_iter:
            self.add_sample(x)

    def _get_n(self):
        return self._n
    
    n = property(_get_n)

    def _get_min(self):
        return self._min

    def _get_max(self):
        return self._max

    minimum = property(_get_min)
    maximum = property(_get_max)
    
    def _get_mean(self):
        if self._n < 1:
            return None
        return self._mean
    
    def _get_variance(self):
        if self._n < 1:
            return None
        if self._n == 1:
            return float('inf')
        return (self._sum_devs_2 / (self._n - 1))

    def _get_std_dev(self):
        if self._n < 1:
            return None
        return math.sqrt(self._get_variance())

    def _get_pop_variance(self):
        if self._n < 1:
            return None
        return (self._sum_devs_2 / self._n)

    mean = property(_get_mean)
    variance = property(_get_variance)
    pop_variance = property(_get_pop_variance)
    std_deviation = property(_get_std_dev)

    def _get_skewness(self):
        return ((self._sum_devs_3 * math.sqrt(self._n)) / \
                (self._sum_devs_2 ** (float(3)/2)))
    def _get_kurtosis(self):
        return (((self._n * self._sum_devs_4) / (self._sum_devs_2 ** 2)) - 3)

    skewness = property(_get_skewness)
    kurtosis = property(_get_kurtosis)

    def __str__(self):
        s = StringIO()
        s.write('name = {0}\n'.format(self.name))
        s.write('tag = {0}\n'.format(self.tag))
        s.write('sample size = {0}\n'.format(self._n))
        s.write('min = {0}\nmax = {1}\n'.format(self._min, self._max))
        s.write('mean = {0}\n'.format(self.mean))
        s.write('variance = {0}\n'.format(self.variance))
        s.write('pop variance = {0}\n'.format(self.pop_variance))
        s.write('skewness = {0}\n'.format(self.skewness))
        s.write('kurtosis = {0}\n'.format(self.kurtosis))
        return s.getvalue()


def arg_is_positive_int(i):
    try:
        if int(i) < 1:
            raise
    except:
        msg = '{0!r} is not a positive integer'.format(i)
        raise argparse.ArgumentTypeError(msg)
    return int(i)

def arg_is_nonnegative_int(i):
    try:
        if int(i) < 0:
            raise
    except:
        msg = '{0!r} is not a nonnegative integer'.format(i)
        raise argparse.ArgumentTypeError(msg)
    return int(i)

def arg_is_positive_float(i):
    try:
        if float(i) <= 0.0:
            raise
    except:
        msg = '{0!r} is not a positive real number'.format(i)
        raise argparse.ArgumentTypeError(msg)
    return float(i)

def arg_is_nonnegative_float(i):
    try:
        if float(i) < 0.0:
            raise
    except:
        msg = '{0!r} is not a non-negative real number'.format(i)
        raise argparse.ArgumentTypeError(msg)
    return float(i)


def main_cli(argv = sys.argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-u', '--number-of-up-parameters',
            action = 'store',
            type = arg_is_nonnegative_int,
            default = 6,
            help = 'Number of \"up\" parameters.')
    parser.add_argument('-d', '--number-of-down-parameters',
            action = 'store',
            type = arg_is_nonnegative_int,
            default = 2,
            help = 'Number of \"down\" parameters.')
    parser.add_argument('-a', '--power-addend',
            action = 'store',
            type = float,
            default = 0.0,
            help = 'Numer added to power term of hastings ratio.')
    parser.add_argument('-n', '--number-of-mcmc-generations',
            action = 'store',
            type = arg_is_positive_int,
            default = 100000,
            help = 'Number of MCMC iterations.')
    parser.add_argument('-f', '--sample-frequency',
            action = 'store',
            type = arg_is_positive_int,
            default = 10,
            help = 'The number of MCMC generations between samples.')
    parser.add_argument('-b', '--burnin',
            action = 'store',
            type = arg_is_positive_int,
            default = 10,
            help = 'The number of MCMC samples to ignore.')
    parser.add_argument('-x', '--turn-up-down-move-off',
            action = 'store_true',
            help = 'Turn multivariate up/down move off.')
    parser.add_argument('-s', '--seed',
            metavar='SEED',
            action = 'store',
            type = arg_is_positive_int,
            help = 'Seed for random number generator.')

    if argv == sys.argv:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)
    if not args.seed:
        args.seed = random.randint(1, 999999999)
    _RNG.seed(args.seed)

    m = Model(
            seed = args.seed,
            number_of_up_parameters = args.number_of_up_parameters,
            number_of_down_parameters = args.number_of_down_parameters,
            power_addend = args.power_addend,
            up_down_move_on = (not args.turn_up_down_move_off))

    m.mcmc(number_of_generations = args.number_of_mcmc_generations,
            sample_frequency = args.sample_frequency,
            burnin = args.burnin)
    sys.stdout.write("{0}".format(m.summarize()))


class ParameterTestCase(unittest.TestCase):
    
    def test_basic(self):
        p = Parameter(123)
        samples = []
        niterations = 100000
        sample_freq = 10
        expected_nsamples = niterations / sample_freq
        for i in range(niterations):
            p.perform_scale_update()
            if ((i + 1) % sample_freq) == 0:
                samples.append(p.value)
        self.assertEqual(len(samples), expected_nsamples)
        s = SampleSummarizer(samples)
        self.assertAlmostEqual(s.mean, 0.5, places = 2)
        self.assertAlmostEqual(s.variance, (1.0/12.0), places = 2)
        print(str(s))
        print(p._scale_operator.get_acceptance_rate())


class ModelTestCase(unittest.TestCase):
    
    def test_univariate_moves_5up_2down(self):
        m = Model(111,
                number_of_up_parameters = 5,
                number_of_down_parameters = 2)
        up_samples = [[] for i in range(len(m.up_parameters))]
        down_samples = [[] for i in range(len(m.down_parameters))]
        niterations = 500000
        sample_freq = 5
        expected_nsamples = niterations / sample_freq
        for i in range(niterations):
            m.perform_univariate_scale_updates()
            if ((i + 1) % sample_freq) == 0:
                for i, p in enumerate(m.up_parameters):
                    up_samples[i].append(p.value)
                for i, p in enumerate(m.down_parameters):
                    down_samples[i].append(p.value)
        self.assertEqual(len(up_samples), len(m.up_parameters)) 
        self.assertEqual(len(down_samples), len(m.down_parameters)) 
        for i, samples in enumerate(up_samples):
            self.assertEqual(len(samples), expected_nsamples)
            s = SampleSummarizer(samples, tag = "up_parameter {0}".format(i))
            print(str(s))
            self.assertAlmostEqual(s.mean, 0.5, places = 2)
            self.assertAlmostEqual(s.variance, (1.0/12.0), places = 2)
        for i, samples in enumerate(down_samples):
            self.assertEqual(len(samples), expected_nsamples)
            s = SampleSummarizer(samples, tag = "down_parameter {0}".format(i))
            print(str(s))
            self.assertAlmostEqual(s.mean, 0.5, places = 2)
            self.assertAlmostEqual(s.variance, (1.0/12.0), places = 2)
        print(m._up_down_operator.get_acceptance_rate())

    def test_multivariate_moves_5up_2down(self):
        m = Model(123,
                number_of_up_parameters = 5,
                number_of_down_parameters = 2)
        up_samples = [[] for i in range(len(m.up_parameters))]
        down_samples = [[] for i in range(len(m.down_parameters))]
        niterations = 100000
        sample_freq = 10
        expected_nsamples = niterations / sample_freq
        for i in range(niterations):
            m.perform_up_down_update()
            if ((i + 1) % sample_freq) == 0:
                for i, p in enumerate(m.up_parameters):
                    up_samples[i].append(p.value)
                for i, p in enumerate(m.down_parameters):
                    down_samples[i].append(p.value)
        self.assertEqual(len(up_samples), len(m.up_parameters)) 
        self.assertEqual(len(down_samples), len(m.down_parameters)) 
        for i, samples in enumerate(up_samples):
            self.assertEqual(len(samples), expected_nsamples)
            s = SampleSummarizer(samples, tag = "up_parameter {0}".format(i))
            print(str(s))
            self.assertAlmostEqual(s.mean, 0.5, places = 2)
            self.assertAlmostEqual(s.variance, (1.0/12.0), places = 2)
        for i, samples in enumerate(down_samples):
            self.assertEqual(len(samples), expected_nsamples)
            s = SampleSummarizer(samples, tag = "down_parameter {0}".format(i))
            print(str(s))
            self.assertAlmostEqual(s.mean, 0.5, places = 2)
            self.assertAlmostEqual(s.variance, (1.0/12.0), places = 2)
        print(m._up_down_operator.get_acceptance_rate())

    def test_multivariate_moves_1up_1down(self):
        m = Model(123,
                number_of_up_parameters = 1,
                number_of_down_parameters = 1)
        up_samples = [[] for i in range(len(m.up_parameters))]
        down_samples = [[] for i in range(len(m.down_parameters))]
        niterations = 100000
        sample_freq = 10
        expected_nsamples = niterations / sample_freq
        for i in range(niterations):
            m.perform_up_down_update()
            if ((i + 1) % sample_freq) == 0:
                for i, p in enumerate(m.up_parameters):
                    up_samples[i].append(p.value)
                for i, p in enumerate(m.down_parameters):
                    down_samples[i].append(p.value)
        self.assertEqual(len(up_samples), len(m.up_parameters)) 
        self.assertEqual(len(down_samples), len(m.down_parameters)) 
        for i, samples in enumerate(up_samples):
            self.assertEqual(len(samples), expected_nsamples)
            s = SampleSummarizer(samples, tag = "up_parameter {0}".format(i))
            print(str(s))
            self.assertAlmostEqual(s.mean, 0.5, places = 2)
            self.assertAlmostEqual(s.variance, (1.0/12.0), places = 2)
        for i, samples in enumerate(down_samples):
            self.assertEqual(len(samples), expected_nsamples)
            s = SampleSummarizer(samples, tag = "down_parameter {0}".format(i))
            print(str(s))
            self.assertAlmostEqual(s.mean, 0.5, places = 2)
            self.assertAlmostEqual(s.variance, (1.0/12.0), places = 2)
        print(m._up_down_operator.get_acceptance_rate())


if __name__ == "__main__":
    if "--run-tests" in sys.argv:

        sys.stderr.write("""
*********************************************************************
Running test suite using the following Python executable and version:
{0}
{1}
*********************************************************************
\n""".format(sys.executable, sys.version))

        import doctest

        # doctest.testmod(verbose = True)
        suite = unittest.TestSuite()
        suite.addTest(doctest.DocTestSuite())

        tests = unittest.defaultTestLoader.loadTestsFromName(
               os.path.splitext(os.path.basename(__file__))[0])
        suite.addTests(tests)

        runner = unittest.TextTestRunner(verbosity = 2)
        runner.run(suite)

        sys.exit(0)

    main_cli()
