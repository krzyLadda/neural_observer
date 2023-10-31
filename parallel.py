"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
from multiprocessing import Pool


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None):
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a floats  (the genome's fitness and MSE's).
        """
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(num_workers)

    def __del__(self):
        self.pool.close()  # should this be terminate?
        self.pool.join()

    def evaluate(self, inds, config):
        jobs = []
        for ind in inds:
            # uncomment to test
            # ind.fitness, ind.MSE, ind.net.V, ind.net.W = self.eval_function(ind, config)
            jobs.append(self.pool.apply_async(self.eval_function, (ind, config)))

        # assign the fitness and MSE back to each genome
        for job, ind in zip(jobs, inds):
            ind.fitness, ind.MSE, V, W = job.get(timeout=self.timeout)
            if V is not None and W is not None:
                # weights must be rewritten to ind here,
                # because eval_function operates on copies
                # of arguments (parralel feature?)
                ind.net.V = V
                ind.net.W = W
