import time


# class to encapsulate performance timing
class Timer:

    def __init__(self):
        self.prevTime = 0


    def reset_time(self):
        self.prevTime = time.time()

    def log_time(self, log_str):
        currTime = time.time()
        print(log_str + ' completed in: ' + str(currTime-self.prevTime))
        self.prevTime = currTime
    pass
