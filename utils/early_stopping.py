class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best_value = float('-inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, value):
        if value > self.best_value:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            self.should_stop = self.counter >= self.patience
        return self.should_stop
