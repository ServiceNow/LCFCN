class Meter:
    def __init__(self):
        self.n_sum = 0.
        self.n_counts = 0.

    def add(self, n_sum, n_counts):
        self.n_sum += n_sum 
        self.n_counts += n_counts

    def get_avg_score(self):
        return self.n_sum / self.n_counts
