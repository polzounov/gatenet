import time

class Timer:
    """Encapsulate performance timing"""
    def __init__(self):
        self.prevTime = 0

    def reset_time(self):
        self.prevTime = time.time()

    def log_time(self, log_str):
        currTime = time.time()
        print(log_str + ' completed in: ' + str(currTime-self.prevTime))
        self.prevTime = currTime
    pass


def time_func(input_func):
    """Print out timing details of function. Use as a decorator (@time_func)"""
    def timed(*args, **kwargs):
        start_time = time.time()
        result = input_func(*args, **kwargs)
        end_time = time.time()
        print('Method Name - {0}, Args - {1}, Kwargs - {2}, Execution Time - {3}'.format(
            input_func.__name__,
            args,
            kwargs,
            end_time - start_time
        ))
        return result

    return timed
