import numpy as np
from ffNeuralNet import ffNet
import controlTheory2 as ct
import copy
# from myOde import myOde


class observer:
    def __init__(self, config):

        if config["BP_ver"] == "abdollahi":
            self.dwEquation = self.dwEquationAbdollahi
            self.dvEquation = self.dvEquationAbdollahi
        elif config["BP_ver"] == "BP":
            self.dwEquation = self.dwBP
            self.dvEquation = self.dvBP


        self.x = np.full((config['sysRank'], 1), np.nan)
        self.dt = config["dt"]
        self.hmU = config["hmU"]

        # network
        hmIn = config["sysRank"] + self.hmU
        hmOut = config["sysRank"]
        self.net = ffNet(self.hmN, hmIn, hmOut, config)

    """EQUATIONS"""

    def dxEquation(self, xRoz, yError):
        # estimation error yError = yMeasuredK - yEstK
        g = self.net.respond(xRoz)
        # equation (8)
        dx = np.matmul(self.A, self.x) + np.matmul(self.G, yError) + g
        return np.array(dx)

    def estimateNextX(self, xRoz, yError):
        # estimates X at time k+1, u and yMeaures and yEst are from time k
        dx = self.dxEquation(xRoz, yError)
        return self.x + self.dt * dx

    def estimate(self, u, measurements, measurementsC1, x0, hmSamples, learn=False):
        # estimates the waveforms together with the possibility of learning
        self.xEst = np.zeros([hmSamples, self.A.shape[0]])
        self.x = x0  # xEst0
        # write the estimate at time k=0
        self.xEst[0, :] = x0.reshape(-1)
        V = copy.deepcopy(self.net.V)
        W = copy.deepcopy(self.net.W)
        # hmSamples-1 because it estimates at moment k+1 from moment k
        for k in range(hmSamples-1):
            # xRoz(k) = [x(k); u(k)]]
            xRozK = np.concatenate((self.x, u[k, :].reshape(-1, 1)))
            # estimation error at time k
            yErrorK = measurements[k, :].reshape(-1, 1) - np.matmul(self.C, self.x)
            try:
                # estimate xEst which is at moment k+1
                xEst = self.estimateNextX(xRozK, yErrorK)
            except FloatingPointError:  # as the estimate flies into inf
                print(f"Estimates of ind {self.ID} flew to inf (returned None).")
                return None, None, None
            self.xEst[k+1, :] = xEst.reshape(-1)

            if learn:
                yErrorK_C1 = measurementsC1[k, :].reshape(-1, 1) - np.matmul(self.C1, self.x)
                try:
                    # calculate the new weights based on the data of moment k
                    Vk, Wk = self.weightsUpdate(xRozK, yErrorK_C1)
                    # self.net.V = Vk  # here is possible do online version
                    # self.net.W = Wk
                    V = V + Vk
                    W = W + Wk
                except (FloatingPointError, TypeError):
                    print(f"New weights of ind {self.ID} flew to inf (returned None).")
                    return None, None, None
                if Vk is None or Wk is None:  # TODO: this is probably not necessary
                    return None, None, None

            # x(k) = x(k+1)
            self.x = xEst
        if not learn:
            V = self.net.V
            W = self.net.W

        return self.xEst, V, W

    def calculateY(self, x):
        return np.matmul(self.C, x)

    def dwEquationAbdollahi(self, xRoz, yError):
        # output layer weights
        # all arguments of the function are of moment k
        # xRoz = [x u] at the time k
        # yError - estimation error at time k
        vRespond = self.net.vOutput  # vRespond(xRoz)
        # equation 11:
        foo = np.matmul(yError.T, self.C1)  # yError^T * C
        foo = np.matmul(foo, self.invAC)  # yError^T * C * AC^-1
        # (yError^T * C * AC^-1)^T * (vRespond)^T
        foo = np.matmul(foo.T, vRespond.T)
        part1 = -self.n1 * foo
        # in the equation below there is specifically a multiplication *
        # and not a matmul, because the norm of a vector is always a scalar
        # p1 * || yError || * W
        part2 = -self.p1 * ct.vectorNorm(yError) * self.net.W

        return np.array(part1 + part2)

    def dvEquationAbdollahi(self, xRoz, yError):
        # hidden layer weights
        # all arguments of the function are of moment k
        # xRoz = [x u] at the time k
        # yError - estimation error at time k

        # equation 12:
        foo = np.matmul(yError.T, self.C1)  # yError^T * C
        foo = np.matmul(foo, self.invAC)  # yError^T * C * AC^-1
        foo = np.matmul(foo, self.net.W)  # yError^T * C * AC^-1 * W
        bar = self.net.vOutput  # vRespond(xRoz)  # vRespond
        bar = np.diagflat(np.square(bar))  # lambda = diag(vRespond^2)
        bar = np.eye(bar.shape[0]) - bar  # (I - diag(vRespond^2))
        # - n2 ( yError^T * C * AC^-1 * W * (I - diag(vRespond^2)) )^T * sgn(xRoz)^T
        part1 = -self.n2 * np.matmul(foo, bar).T * np.sign(xRoz).T
        part2 = -self.p2 * ct.vectorNorm(yError) * self.net.V

        return np.array(part1 + part2)

    def dwBP(self, x, e):
        # output layer weights
        # calc derivative of outputs neuron
        # der = 1
        # d = e
        return self.n1*np.matmul(e, self.net.vOutput.reshape(1, -1))

    def dvBP(self, x, e):
        # hidden layer weights
        d = np.matmul(np.transpose(self.net.W), e)
        der = self.net.actFunDerivative(np.matmul(self.net.V, x))
        return self.n2* np.matmul((d*der), x.reshape(1, -1))

    def weightsUpdate(self, xRoz, yError):
        try:
            dw = self.dwEquation(xRoz, yError)
            dv = self.dvEquation(xRoz, yError)
            V = self.net.V + dv
            W = self.net.W + dw
        except FloatingPointError:
            V = None
            W = None
        return V, W
