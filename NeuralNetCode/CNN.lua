require 'torch'
require 'nn'
require 'image'
require 'optim'

-- Load training
loaded = torch.load('mnist.t7/train_32x32.t7', 'ascii')
trainData = {
    data = loaded.data,
    labels = loaded.labels,
    size = function() return (#trainData.data)[1]/100 end
}

-- Load testing
loaded = torch.load('mnist.t7/test_32x32.t7', 'ascii')
testData = {
    data = loaded.data,
    labels = loaded.labels,
    size = function() return (#testData.data)[1]/100 end
}

-- Convert to float
trainData.data = trainData.data:float()
testData.data = testData.data:float()

-- Initialize variables
ninput = 32 * 32
nhidden = 300
noutput = 10

-- Create the model
model = nn.Sequential()
model:add(nn.Reshape(1,32,32))
-- layer 1:
model:add(nn.SpatialConvolution(1, 16, 5, 5))
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- layer 2:
model:add(nn.SpatialConvolution(64, 128, 5, 5))
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- layer 3, a simple 2-layer neural net:
model:add(nn.Reshape(128*5*5))
model:add(nn.Linear(128*5*5, 200))
model:add(nn.Tanh())
model:add(nn.Linear(200,10))
model:add(nn.LogSoftMax())
-- Criterion
criterion = nn.ClassNLLCriterion()

-- Create a list of classes
classes = {'0','1','2','3','4','5','6','7','8','9'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters, gradParameters = model:getParameters()
end

optimState = {
    learningRate = 0.1,
    weightDecay = 0.1,
    momentum = 0.01,
    learningRateDecay = 1e-7
}
optimMethod = optim.sgd

-- Training Function
function train_model()

	-- Initialize variables
	epoch = epoch or 1
	local time = sys.clock()
	model:training()
	shuffle = torch.randperm(trainData:size())
	batchSize = 64

	-- Do one epoch
	for t = 1, trainData:size(), batchSize do

		local inputs = {}
		local targets = {}

		-- disp progress
		xlua.progress(t, trainData:size())

		-- Build the mini-batch
		for i = t,math.min(t+63,trainData:size()) do
			-- load new sample
			local input = trainData.data[shuffle[i]]
			local target = trainData.labels[shuffle[i]]
			input = input:double()
			table.insert(inputs, input)
			table.insert(targets, target)
		end

		-- Build a closure
		local feval = function(x)
			if x ~= parameters then
			parameters:copy(x)
			end

			gradParameters:zero()

			local f = 0
			for i = 1, #inputs do
				local output = model:forward(inputs[i])
				output = torch.reshape(output, 10)
				local err = criterion:forward(output, targets[i])
				f = f + err
				local df_do = criterion:backward(output:double(), targets[i])
				model:backward(inputs[i], df_do)
				confusion:add(output, targets[i])
			end

			gradParameters:div(#inputs)
			f = f / #inputs

			return f, gradParameters
		end

			config = {learningRate = 0.003, weightDecay = 0.01, momentum = 0.01, learningRateDecay = 5e-7}
			optim.sgd(feval, parameters, config)
	end

	-- print(confusion)
	print(confusion)
	confusion:zero()

	local filename = paths.concat('model.net')
	os.execute('mkdir -p ' .. sys.dirname(filename))
	torch.save(filename, model)

	epoch = epoch + 1
end

for i = 1,25 do
	train_model()
end

test_confusion = optim.ConfusionMatrix(classes)

function test_model()
    for t = 1, testData:size() do
        local input = testData.data[t]:double()
        local target = testData.labels[t]
    
        local pred = model:forward(input)
        pred = torch.reshape(pred, 10)
        test_confusion:add(pred, target)
    end
end

test_model()

print(test_confusion)
