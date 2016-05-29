
require 'torch'   
require 'image'  
require 'nn'      
require 'optim'
require 'gnuplot'
require 'os'	

------------------------------------------------------------------------------
-- INITIALIZATION AND DATA
------------------------------------------------------------------------------

-- fix random seed so program runs the same every time
torch.manualSeed(1)   

logger = optim.Logger('loss_12net.log')
logger:setNames{'train error', 'test error'}


local opt = {}    
opt.optimization = 'sgd'
opt.batch_size = 128
--opt.train_size = 129
opt.train_size = (9/10)*255225
--opt.test_size = 128
opt.test_size = 255225 - opt.train_size   
opt.epochs = 1 --train for 100  7.9448e-01


optimState = {
	nesterov = true,
	learningRate = 0.0001,
	learningRateDecay = 1e-7,
	momentum = 0.9,
	dampening = 0,
	--weightDecay = 0.05,
}


--Trimming the training model to save space and enhance cpu performance
local function trimModel(model)
	for i=1,#model.modules do
		local layer = model:get(i)
		if layer.gradParameters ~= nil then
			layer.gradParameters = layer.gradParameters.new()
		end

		if layer.output ~= nil then
			layer.output = layer.output.new()
		end
		if layer.gradInput ~= nil then
			layer.gradInput = layer.gradInput.new()
		end
	end
	collectgarbage()
end


------------------------------------------------------------------------------
-- LOADING DATA
------------------------------------------------------------------------------

----------------- Generating random 20000 neg examples out of PASCAL -------------
----------------- 'non-face' 12X12 patches and seriallize at end. ----------------
local function GetNegPascalSamples()
	files = {}
	-- Go over all files in directory. We use an iterator, paths.files().
	for file in paths.files('../images') do
		table.insert(files, paths.concat('../images',file))
	end

	--creating 20000 of negative random samples
	negative_samples = {}
	for i = 1,#files do

		collectgarbage()
		
		_,_,ext = string.match(files[i], "(.-)([^\\]-([^\\%.]+))$")
		if (ext ~= 'jpg') then 
		else	
			local img = image.load(files[i])
			--assuming all images has identical size
			image_size_y = img:size(3)
			image_size_x = img:size(2)

			-- create negative examples by cropping PASCAL images at random locations 
			-- to produce 12X12 outputs.  (x1,y1) & (x2,y2) represents top left & buttom right point resp.
			for j = 1,40 do  -- Generating ~200,000 false exmaples as in the article
				local ran_x1 = torch.random(1,image_size_x-12)
				local ran_y1 = torch.random(1,image_size_y-12)
				local cropped = image.crop(img, ran_y1, ran_x1, ran_y1+12, ran_x1+12)
				table.insert(negative_samples, cropped:resize(1,3,12,12))
			end
		end
	end
	torch.save('negatives.t7',negative_samples)
end



--Loading the positive (12X12 faces) data from aflw DB
local pos_data = torch.load('aflw_12_tensor.t7'):double()
local pos_data_labels = torch.Tensor(pos_data:size(1)):fill(1)
--Loading the negative (non faces) data from Pascal DB
m = nn.JoinTable(1)
--To generate negative data on the fly uncomment the line below.
GetNegPascalSamples()
local negative_data = m:forward(torch.load('negatives.t7'))



local neg_data_labels = torch.Tensor(negative_data:size(1)):fill(2)
--Create a mixed data out of negatives and positives
local data = torch.cat(negative_data:double(), pos_data:double(),1)
local labels = torch.cat(neg_data_labels:double(), pos_data_labels:double(),1)


------------------------------------------------------------------------------
-- MODEL
------------------------------------------------------------------------------
local model = nn.Sequential();
-- input 3x12x12
model:add(nn.SpatialConvolution(3, 16, 3, 3))
-- outputs 16x10x10
model:add(nn.SpatialMaxPooling(3, 3, 2, 2))
model:add(nn.ReLU())
-- outputs 16x4x4
model:add(nn.SpatialConvolution(16, 16, 4, 4))
model:add(nn.ReLU())
-- outputs 16x1x1
model:add(nn.SpatialConvolution(16, 2, 1, 1))
-- outputs 2x1x1
model:add(nn.SpatialSoftMax())
-- handling with diminishing gradients
model:add(nn.AddConstant(0.000000001))
model:add(nn.Log())

------------------------------------------------------------------------------
-- LOSS FUNCTION
------------------------------------------------------------------------------

local criterion = nn.CrossEntropyCriterion()

------------------------------------------------------------------------------
-- TRAINING
------------------------------------------------------------------------------

local parameters, gradParameters = model:getParameters()

------------------------------------------------------------------------
-- Closure with mini-batches
------------------------------------------------------------------------

local counter = 0
local feval = function(x)
	if x ~= parameters then
		parameters:copy(x)
	end

	local start_index = counter * opt.batch_size + 1
	local end_index = math.min(opt.train_size, (counter + 1) * opt.batch_size)
	if end_index == opt.train_size then
		counter = 0
	else
		counter = counter + 1
	end
	
	local batch_inputs = data[{{start_index, end_index}, {}}]
	local batch_targets = labels[{{start_index, end_index}}]
	gradParameters:zero()

	-- 1. compute outputs (log probabilities) for each data point
	local batch_outputs = model:forward(batch_inputs)
	-- 2. compute the loss of these outputs, measured against the true labels in batch_target
	local batch_loss = criterion:forward(batch_outputs, batch_targets)
	-- 3. compute the derivative of the loss wrt the outputs of the model
	local loss_doutput = criterion:backward(batch_outputs, batch_targets)
	-- 4. use gradients to update weights
	model:backward(batch_inputs, loss_doutput)

	return batch_loss, gradParameters
end


------------------------------------------------------------------------
-- OPTIMIZE
------------------------------------------------------------------------
local train_losses = {}
local test_losses = {} 

-- # epoch tracker
epoch = epoch or 1

for i = 1,opt.epochs do

	trimModel(model)

	-- shuffle at each epoch
	local shuffled_indexes = torch.randperm(data:size(1)):long()
	data = data:index(1,shuffled_indexes)
	labels = labels:index(1,shuffled_indexes)

	local train_loss_per_epoch = 0
	-- do one epoch
	print('==> doing epoch on training data:')
	print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batch_size .. ']')

	for t = 1,opt.train_size,opt.batch_size do
		if opt.optimization == 'sgd' then
			_, minibatch_loss  = optim.sgd(feval, parameters, optimState)
			
			print('mini_loss: '..minibatch_loss[1])
			train_loss_per_epoch = train_loss_per_epoch + minibatch_loss[1]
		end
	end
	-- update train_losses average among all the mini batches
	train_losses[#train_losses + 1] = train_loss_per_epoch / (math.ceil(opt.train_size/opt.batch_size)-1)

	------------------------------------------------------------------------
	-- TEST
	------------------------------------------------------------------------

	trimModel(model)

	local test_data = data[{{opt.train_size+1, data:size(1)}, {}}]
	local test_labels = labels[{{opt.train_size+1, data:size(1)}}]

	local output_test = model:forward(test_data)
	local err = criterion:forward(output_test, test_labels)

	test_losses[#test_losses + 1] = err
	print('test error ' .. err)
	
	logger:add{train_losses[#train_losses], test_losses[#test_losses]}

end

------------------------------------------------------------------------
--  PLOTTING TESTING/TRAINING LOSS/CLASSIFICATION ERRORS
------------------------------------------------------------------------
gnuplot.pdffigure('loss_12net.pdf')
gnuplot.plot({'train loss',torch.range(1, #train_losses),torch.Tensor(train_losses)},{'test loss',torch.Tensor(test_losses)})
gnuplot.title('loss per epoch')
gnuplot.figure()


------------------------------------------------------------------------------
-- SAVING MODEL
------------------------------------------------------------------------------

local fmodel = model : clone (): float ()
for i =1 ,# fmodel.modules do
	local layer = fmodel : get ( i )
	if layer.output ~= nil then
		layer.output = layer.output.new ()
	end
	if layer.gradInput ~= nil then
		layer.gradInput = layer.gradInput.new ()
	end
end

torch.save('model_12net.net', fmodel)


