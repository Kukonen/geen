class GeenLearn:
    def __init__(self, learnX, testX, learnY, testY, step = 0.2):
        self.learnX = learnX
        self.testX = testX
        self.learnY = learnY
        self.testY = testY
        self.step = step
        
    def learn(self):
        