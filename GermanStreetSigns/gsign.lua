require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'csvigo'

train_path_prefix = 'GTSRB/Final_Training/Images/'
test_path_prefix = 'GTSRB/Final_Test/Images/'

-- Read Training CSV and Images
train_csv = { Filename = {}, Width = {}, Height = {}, X1 = {}, Y1 = {}, X2 = {}, Y2 = {}, Label = {} }
trainData = { data = {}, labels = {}, size = function() return #trainData.data end }

for i = 0, 42 do
	-- Get the path to the file	
	local pathToFolder = train_path_prefix .. string.format( "%05d/", i )
	local pathToCSV = pathToFolder .. string.format( "GT-%05d.csv", i )

	-- Read the CSV
	local csvData = csvigo.load{ path = pathToCSV, separator = ';', mode = 'tidy', header = true }

	-- Loop through the data
	for i = 1, #(csvData.Filename) do
		table.insert( train_csv.Filename, pathToFolder .. csvData.Filename[i] )
		table.insert( train_csv.Width, tonumber( csvData.Width[i] ) )
		table.insert( train_csv.Height, tonumber( csvData.Height[i] ) )
		table.insert( train_csv.X1, tonumber( csvData["Roi.X1"][i] ) )
		table.insert( train_csv.Y1, tonumber( csvData["Roi.Y1"][i] ) )
		table.insert( train_csv.X2, tonumber( csvData["Roi.X2"][i] ) )
		table.insert( train_csv.Y2, tonumber( csvData["Roi.Y2"][i] ) )
		table.insert( train_csv.Label, tonumber(csvData["ClassId\r"][i]) + 1 )
	end
end

for i, v in ipairs(train_csv.Filename) do
	-- Load the image
	local img = image.load(v)

	-- Crop and scale
	local cropImg = image.crop( img, train_csv.X1[i], train_csv.Y1[i], train_csv.X2[i], train_csv.Y2[i] )
	local scaleImg = image.scale( cropImg, 32, 32 )

	-- Insert new image and label
	table.insert( trainData.data, scaleImg )
	table.insert( trainData.labels, train_csv.Label[i] )
end

-- Read Testing CSV and Images
test_csv = { Filename = {}, Width = {}, Height = {}, X1 = {}, Y1 = {}, X2 = {}, Y2 = {}, Label = {} }
testData = { data = {}, labels = {}, size = function() return #testData.data end }

do
	-- Get the path to the file	
	local pathToCSV = test_path_prefix .. "GT-final_test.csv"

	-- Read the CSV
	local csvData = csvigo.load{ path = pathToCSV, separator = ';', mode = 'tidy', header = true }

	-- Loop through the data
	for i = 1, #(csvData.Filename) do
		table.insert( test_csv.Filename, test_path_prefix .. csvData.Filename[i] )
		table.insert( test_csv.Width, csvData.Width[i] )
		table.insert( test_csv.Height, csvData.Height[i] )
		table.insert( test_csv.X1, csvData["Roi.X1"][i] )
		table.insert( test_csv.Y1, csvData["Roi.Y1"][i] )
		table.insert( test_csv.X2, csvData["Roi.X2"][i] )
		table.insert( test_csv.Y2, csvData["Roi.Y2"][i] )
		table.insert( test_csv.Label, tonumber(csvData["ClassId\r"][i]) + 1 )
	end
end

for i, v in ipairs(test_csv.Filename) do
	-- Load the image
	local img = image.load(v)

	-- Crop and scale
	local cropImg = image.crop( img, test_csv.X1[i], test_csv.Y1[i], test_csv.X2[i], test_csv.Y2[i] )
	local scaleImg = image.scale( cropImg, 32, 32 )

	-- Insert new image and label
	table.insert( testData.data, scaleImg )
	table.insert( testData.labels, test_csv.Label[i] )
end

-- Nulliy CSV data
train_csv = nil
test_csv = nil
collectgarbage()

-- THE MODEL ------------------------------------------------------------------
normKernel = image.gaussian1D(7)

model = nn.Sequential()

model:add(nn.SpatialContrastiveNormalization(3, normKernel))

model:add(nn.SpatialConvolution(3, 40, 5, 5))
model:add(nn.PReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(nn.SpatialConvolution(40, 80, 5, 5))
model:add(nn.PReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(nn.Reshape(80 * 5 * 5))

model:add(nn.Linear(80 * 5 * 5, 256))
--model:add(nn.Reshape(3 * 32 * 32))
--model:add(nn.Linear(3 * 32 * 32, 256))
model:add(nn.PReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(256, 43))
model:add(nn.PReLU())
model:add(nn.LogSoftMax()) 

--model:add(nn.SpatialConvolution(3, 6, 5, 5))
--model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
--model:add(nn.SpatialConvolution(6, 16, 5, 5))
--model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

--model:add(nn.View(16 * 5 * 5))

--model:add(nn.Linear(16 * 5 * 5, 256))
--model:add(nn.PReLU())
--model:add(nn.Dropout(0.5))
--model:add(nn.Linear(256, 43))
--model:add(nn.PReLU())
--model:add(nn.LogSoftMax()) 
-- THE MODEL ------------------------------------------------------------------

-- Create a list of classes
classes = {}
for i = 1, 43 do
	table.insert( classes, i )
end

-- Collect garbage
collectgarbage()

-- Create the confusion matrix
confusion = optim.ConfusionMatrix( classes )

-- Training Function
criterion = nn.ClassNLLCriterion()
parameters, gradParameters = model:getParameters()
function TrainModel( param_learningRate, param_weightDecay, param_momentum, param_learningRateDecay )

	-- Initialize variables
	model:training()
	local shuffle = torch.randperm( trainData:size() )

	-- Do one epoch
	for t = 1, trainData:size(), 64 do

		-- Display progress
		xlua.progress(t, trainData:size())

		-- Initialize variables
		local inputs = {}
		local targets = {}

		-- Build the mini-batch
		for i = t, math.min(t+63,trainData:size()) do

			-- load new sample
			local input = trainData.data[ shuffle[i] ]
			local target = trainData.labels[ shuffle[i] ]
			input = input:double()
			table.insert( inputs, input )
			table.insert( targets, tonumber(target))
		end

		-- Build a closure
		local EvalFunction = function(x)
			if x ~= parameters then
				parameters:copy(x)
			end

			gradParameters:zero()

			local f = 0
			for i = 1, #inputs do
				local output = model:forward(inputs[i])
				output = torch.reshape(output, 43)
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

		-- Set the configuration
		config = {
			learningRate = param_learningRate, 
			weightDecay = param_weightDecay, 
			momentum = param_momentum,
			learningRateDecay = param_learningRateDecay
		}

		-- SGD
		optim.sgd(EvalFunction, parameters, config)
	end

	-- Print the confusion matrix
	print(confusion)
	confusion:zero()
end

-- Run for 'i' epochs
for i = 1, 20 do
	print( "Starting Epoch = " .. i )
--	TrainModel( 0.1, 0.1, 0.01, 1e-7 )
	TrainModel( 0.1, 0.1, 0.9, 1e-7 )
	collectgarbage()
end

-- Define the function to test the model
test_confusion = optim.ConfusionMatrix(classes)
function TestModel()
    for t = 1, testData:size() do
        local input = testData.data[t]:double()
        local target = tonumber(testData.labels[t])

        local pred = model:forward(input)
        pred = torch.reshape(pred, 43)
        test_confusion:add(pred, target)
    end
end

-- Test and print the results
TestModel()
print(test_confusion)
