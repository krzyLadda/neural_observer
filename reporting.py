from copy import deepcopy


class reporter():
    def __init__(self, config, generation):
        self.config = config
        self.startGen = generation
        self.lastGenInds = None
        self.genBest = []
        self.genInds = []
        self.genMSEs = []

    def report(self, inds, genBest):
        self.genBest.append(genBest)
        # self.genInds.append(deepcopy(inds))
        self.lastGenInds = deepcopy(inds)
        self.genMSEs.append([x.MSE for x in inds.values()])
