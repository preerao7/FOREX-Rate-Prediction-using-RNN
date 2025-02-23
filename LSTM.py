class LongShortTermMemory:
    def __init__(self, LRate, maxItrs, stepTime, frame, hiddenLyrs=10, flag=True):
        self.LRate = LRate
        self.maxItrs = maxItrs
        self.stepTime = stepTime
        self.frame = frame
        self.hiddenLyrs = hiddenLyrs
        if(flag):
          self.initializeForgotGateParameters()
          self.initializeInputGateParameters()
          self.initializeOutputGateParameters()
          self.initializeFinalOutputParameters()
        
    def initializeForgotGateParameters(self):
        # Parameters for Forget gate
        self.WeightForget = SetParams('Weight_Forget', frame=(self.hiddenLyrs, self.hiddenLyrs+self.frame[0]), sdWeight=0.1, var=0.5)
        self.BiasedForget = SetParams('Biased_Forget', frame=(self.hiddenLyrs, 1), sdWeight=0.1) 
    def initializeInputGateParameters(self):
        # Parameters for Input gate
        self.WeightInput = SetParams('Weight_Input', frame=(self.hiddenLyrs, self.hiddenLyrs+self.frame[0]), sdWeight=0.1, var=0.5)
        self.BiasedInput = SetParams('Biased_Input', frame=(self.hiddenLyrs, 1), sdWeight=0.1)
    def initializeOutputGateParameters(self):    
        # Parameters for Ouput gate
        # weight central, final
        self.WeightCentral = SetParams('Weight_Central', frame=(self.hiddenLyrs, self.hiddenLyrs+self.frame[0]), sdWeight=0.1, var=0.5)
        self.BiasedCentral = SetParams('Biased_Central', frame=(self.hiddenLyrs, 1), sdWeight=0.1)
        self.WeightOutput = SetParams('Weight_Output', frame=(self.hiddenLyrs, self.hiddenLyrs+self.frame[0]), sdWeight=0.1, var=0.5)
        self.BiasedOutput = SetParams('Biased_Output', frame=(self.hiddenLyrs, 1), sdWeight=0.1)
    def initializeFinalOutputParameters(self):
        # Parameters for Final Output
        self.WeightFinal = SetParams('Weight_Final', frame=(self.frame[1], self.hiddenLyrs), sdWeight=0.1, var=0.5)
        self.BiasedFinal = SetParams('Biased_Final', frame=(self.frame[1], 1), sdWeight=0.1)
        flag = False
        
    def initiateFwd(self, vecX):
        self.vecX = vecX
        hiddenStateVector = [getNumpyZeros(self.hiddenLyrs, 1) for itr in range(self.stepTime)]
        cellStateVector = [getNumpyZeros(self.hiddenLyrs, 1) for itr in range(self.stepTime)]
        forgetGateVector = [getNumpyZeros(self.hiddenLyrs, 1) for itr in range(self.stepTime)]
        inputUpdateGateVector = [getNumpyZeros(self.hiddenLyrs, 1) for itr in range(self.stepTime)]
        cellInputVector = [getNumpyZeros(self.hiddenLyrs, 1) for itr in range(self.stepTime)]
        outputGateVector = [getNumpyZeros(self.hiddenLyrs, 1) for itr in range(self.stepTime)]
        
        it = 0
        while it < (self.stepTime):
            zVector = stackArray(hiddenStateVector[it-1], vecX[it])
            forgetGateVector[it] = squashingSigmoidFunc((self.WeightForget.value @ zVector) + self.BiasedForget.value)
            inputUpdateGateVector[it] = squashingSigmoidFunc((self.WeightInput.value @ zVector) + self.BiasedInput.value)
            cellInputVector[it] = tangentFunc((self.WeightCentral.value @ zVector) + self.BiasedCentral.value)
            cellStateVector[it] = (forgetGateVector[it] * cellStateVector[it-1]) + (inputUpdateGateVector[it] * cellInputVector[it])
            outputGateVector[it] = squashingSigmoidFunc((self.WeightOutput.value @ zVector) + self.BiasedOutput.value)
            hiddenStateVector[it] = outputGateVector[it] * tangentFunc(cellStateVector[it])
            it = it +1
        it=0
        self.forgetGateVector, self.inputUpdateGateVector, self.cellInputVector, self.cellStateVector, self.outputGateVector, self.hiddenStateVector = forgetGateVector, inputUpdateGateVector , cellInputVector, cellStateVector, outputGateVector, hiddenStateVector
        v = self.WeightFinal.value @ self.hiddenStateVector[-1] + self.BiasedFinal.value
        
        return v
        
    
    def initiateBkwd(self, vecY, predictY):
        hiddenStateVectorDiff = [getNumpyZeros(self.hiddenLyrs, 1) for itr in range(self.stepTime + 1)]
        cellStateVectorDiff = [getNumpyZeros(self.hiddenLyrs, 1) for itr in range(self.stepTime + 1)]
        
        eDelta = vecY - predictY
        
        self.WeightFinal.diff = eDelta * self.hiddenStateVector[-1].T
        self.BiasedFinal.diff = eDelta
        
        for step in reversed(range(self.stepTime)):
            hiddenStateVectorDiff[step] = self.WeightFinal.value.T @ eDelta + hiddenStateVectorDiff[step+1]
            outputGateVectorDiff = tangentFunc(self.cellStateVector[step]) * hiddenStateVectorDiff[step] * squashingSigmoidDiff(self.hiddenStateVector[step])
            cellStateVectorDiff[step] = self.outputGateVector[step] * hiddenStateVectorDiff[step] * tangentFuncDiff(self.cellStateVector[step]) + cellStateVectorDiff[step+1]
            cellInputVector = self.inputUpdateGateVector[step] * cellStateVectorDiff[step] * tangentFuncDiff(self.cellInputVector[step])
            inputUpdateGateVectorDiff = self.cellInputVector[step] * cellStateVectorDiff[step] * squashingSigmoidDiff(self.inputUpdateGateVector[step])
            forgetGateVectorDiff = self.cellStateVector[step-1] * cellStateVectorDiff[step] * squashingSigmoidDiff(self.forgetGateVector[step])
            
            zVector = stackArray(self.hiddenStateVector[step-1], self.vecX[step])
            
            self.WeightForget.diff += forgetGateVectorDiff @ zVector.T
            self.BiasedForget.diff += forgetGateVectorDiff
            
            self.WeightInput.diff += inputUpdateGateVectorDiff @ zVector.T
            self.BiasedInput.diff += inputUpdateGateVectorDiff
            self.WeightOutput.diff += outputGateVectorDiff @ zVector.T
            self.BiasedOutput.diff += outputGateVectorDiff
            
            self.WeightCentral.diff += cellStateVectorDiff[step] @ zVector.T
            self.BiasedCentral.diff += cellStateVectorDiff[step]
    
    def getParameters(self):
        val =  [self.WeightForget, self.BiasedForget, 
                self.WeightInput, self.BiasedInput,
                self.WeightCentral, self.BiasedCentral,
                self.WeightOutput, self.BiasedOutput,
                self.WeightFinal, self.BiasedFinal]
        return val
   
    def cleanGradients(self):
        lst = self.getParameters();
        for params in lst:
            params.diff = getNumpyZerosLike(params.value)
        
            
    def parameterUpdate(self):
        lst = self.getParameters();
        for params in lst:
            if params.title == 'Weight_Final':
                mul = self.LRate * params.diff
                params.value =  params.value + mul
            else:
                div = params.diff/self.stepTime
                mul = self.LRate * div
                params.value = params.value + mul
            
    def modelFitting(self, vecX, vecY):
        for ep in range(self.maxItrs):
            loss = 0
            for itr in range(len(vecX)):
                predictY = self.initiateFwd(vecX[itr]) 
                sub = (vecY[itr] - predictY)
                pow = sub**2
                loss = loss + pow
                self.initiateBkwd(vecY[itr], predictY)
                self.parameterUpdate()
                self.cleanGradients()
                 
    def Prediction(self, vecX):
        predictY = []
        l = len(vecX)
        x = 0
        while x < l:
          predictY.append(self.initiateFwd(vecX[x]))
          x+=1
        pred = npy.concatenate(predictY) 
        return pred
