require 'torch'
require 'unsup'

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

-- Perform PCA
evals, evecs = unsup.pca(trainData.data)
collectgarbage()

-- projMat = [1024 x 2]
projMatTranspose = evecs:narrow(2, 1, 2):transpose(1,2)

-- Loop through the data and get the 2D representation for each image
data2D = {}
for i = 1, subsetSize do
	local res = projMatTranspose * trainData.data[i]
	table.insert(data2D, res)
	print(trainData.labels[i] .. ' ' .. res[1] .. ' ' .. res[2])
end
