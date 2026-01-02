class LinearRegression:
    def __init__(self, m = 0, b = 0, lr = 0.0001, epochs = 1000):
        self.m = m
        self.b = b
        self.lr = lr
        self.epochs = epochs

    def _gradient_descent(self, x:list, y:list):
        features_length = len(x)
        dm = 0
        db = 0

        for i in range(features_length):
            
            y_predicted = self.m * x[i] + self.b
            # print(f"{x[i]}  {y[i]}  {y_predicted}")
            error = y[i] - y_predicted
            dm += ((2 * x[i] * error) * -1)
            db += ((2 * error) * -1)

        self.m -= (self.lr * (dm / features_length))
        self.b -= (self.lr * (db / features_length))
        return self.m, self.b
        

    
    def fit(self, X:list, Y:list):
        for _ in range(self.epochs):
            self.m, self.b = self._gradient_descent(X, Y)

    def predict(self, X):
        test_length = len(X)
        test_prediction = []
        for i in range(test_length):
            y = self.m * X[i] + self.b
            test_prediction.append(y)
        return test_prediction
    


        