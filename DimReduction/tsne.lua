local manifold = require 'manifold'

-- Initialize variables
subsetSize = 3200

-- function that performs demo of t-SNE code on MNIST:
local function demo_tsne()
	-- amount of data to use for test:
	local N = subsetSize

	-- load subset of MNIST test data:
	local mnist = require 'mnist'
	local trainset = mnist.traindataset()
	trainset.size = N
	trainset.data = trainset.data:narrow(1, 1, N)
	trainset.label = trainset.label:narrow(1, 1, N)
	local x = torch.Tensor(trainset.data:size())
	x:map(trainset.data, function(xx, yy) return yy end)
	x:resize(x:size(1), x:size(2) * x:size(3))
	local labels = trainset.label

	-- run t-SNE:
	local timer = torch.Timer()
	opts = {ndims = 2, perplexity = 30, pca = 50, use_bh = false}
	mapped_x1 = manifold.embedding.tsne(x, opts)
	print('Successfully performed t-SNE in ' .. timer:time().real .. ' seconds.')

	-- print result
	for i = 1,subsetSize do
		print((labels[i] + 1) .. ' ' .. mapped_x1[i][1] .. ' ' .. mapped_x1[i][2])
	end
end

-- run the demo:
demo_tsne()
