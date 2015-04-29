require 'torch'
require 'unsup'
require 'nn'
require 'optim'

-- Initialize variables
subsetSize = 3200

-- Load the training data
loaded = torch.load('mnist.t7/train_32x32.t7', 'ascii')
trainData = {
    data = loaded.data:double(),
    labels = loaded.labels,
    size = function() return (#trainData.data)[1] end
}

-- Resize the data
trainData.data:resize( subsetSize, 32 * 32 )
collectgarbage()

-- Create the encoder and decoder
local inputSize = 32 * 32
encoder = nn.Sequential()
encoder:add(nn.Linear(inputSize,2))
encoder:add(nn.Tanh())

decoder = nn.Sequential()
decoder:add(nn.Linear(2,inputSize))

-- Build the auto-encoder
module = unsup.AutoEncoder(encoder, decoder, 1)
 
-- parameters
x,dl_dx = module:getParameters()
 
-- SGD config
sgdconf = {learningRate = 1e-3}
 
-- assuming a table trainData with the form:
-- trainData = {
--    [1] = sample1,
--    [2] = sample2,
--    [3] ...
-- }
local minibatchsize = 32
function doOneEpoch()
	for i = 1,subsetSize,minibatchsize do
	 
	    -- create minibatch of training samples
	    samples = torch.Tensor(minibatchsize,inputSize)
	    for i = 1,minibatchsize do
		samples[i] = trainData.data[i]
	    end
	 
	    -- define closure
	    local feval = function()
	      -- reset gradient/f
	      local f = 0
	      dl_dx:zero()
	 
	      -- estimate f and gradients, for minibatch
	      for i = 1,minibatchsize do
		 -- f
		 f = f + module:updateOutput(samples[i], samples[i])
	 
		 -- gradients
		 module:updateGradInput(samples[i], samples[i])
		 module:accGradParameters(samples[i], samples[i])
	      end
	 
	      -- normalize
	      dl_dx:div(minibatchsize)
	      f = f/minibatchsize
	 
	      -- return f and df/dx
	      return f,dl_dx
	   end
	 
	   -- do SGD step
	   optim.sgd(feval, x, sgdconf)
	 
	end
end

-- Do 200 epochs
for i = 1, 200 do
	doOneEpoch()
end

-- Auto-Encoder training complete
-- Loop through the subset and compute the 2D representation
data2D = {}
encoderWeights = module:parameters()[1]
for i = 1, subsetSize do
	local res = encoderWeights * trainData.data[i];
	table.insert(data2D, res)
	print(trainData.labels[i] .. ' ' .. res[1] .. ' ' .. res[2])
end

