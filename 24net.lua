---------------------------------------------------------------------------------------
-- Ex3 - q3 
-- Igor Lapshun 304357262
---------------------------------------------------------------------------------------


require 'torch'   
require 'image'  
require 'nn'      
require 'optim'
require 'gnuplot'
require 'os'	
require 'PyramidPacker'
require 'PyramidUnPacker'
require 'nms'

------------------------------------------------------------------------------
-- EX3 Q3 - 24-net
------------------------------------------------------------------------------

local logger = optim.Logger('loss_24net.log')
logger:setNames{'train error', 'test error'}
torch.manualSeed(123)
torch.setdefaulttensortype('torch.DoubleTensor')


local opt = {}         -- these options are used throughout
opt.optimization = 'sgd'
opt.batch_size = 128	
--143519
opt.train_size = math.ceil((9/10)*71395)
opt.test_size = 71395 - opt.train_size   
opt.epochs = 300 


local optimMethod 
if opt.optimization == 'sgd' then
	optimState = {
		nesterov = true,
		learningRate = 0.001,
		learningRateDecay = 1e-7,
		momentum = 0.9,
		dampening = 0,
		--weightDecay = 0.05,
	}
	optimMethod = optim.sgd
elseif opt.optimization == 'adagrad' then
	optimState = {
		learningRate = 1e-1,
	}
	optimMethod = optim.adagrad
end  


function trimModel(model)
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
-- PREPROCESSING 
------------------------------------------------------------------------------

------ Negative mining step - loading PASCAL data-set and feed its pyramid to the 12-net, re-scale every 
------ detection (false positive) to 24X24 and feed them to 24-net 

local GetNegatives = function() 

	files = {}
	-- Go over all files in directory. We use an iterator, paths.files().
	for file in paths.files('../images') do
		table.insert(files, paths.concat('../images',file))
	end

	local smallestImgDim = 50
	local scales = {} -- list of scales
	for k =1 ,1 do
		local scale = 12/( smallestImgDim /1+ smallestImgDim *(k -1)/1)
		if scale * smallestImgDim < 12 then break end
		table.insert (scales , scale )
	end

	model_12net = torch.load('../q1/model_12net.net'):double()

	-- create pyramid packer and unpacker. scales is a table with all -- the scales you with to check. 
	local unpacker = nn.PyramidUnPacker(model_12net)
	local packer = nn.PyramidPacker(model_12net, scales) 

	local false_positive_24_pascal_crops = {}

	--load images from PASCAL 
	for i = 1,#files do
		
		collectgarbage()
		
		_,_,ext = string.match(files[i], "(.-)([^\\]-([^\\%.]+))$")
		if (ext ~= 'jpg') then 
		else	
			img = image.load(files[i])

			local pyramid , coordinates = packer:forward(img)	 

			if pyramid:size(1) == 1 then
				pyramid = torch.cat(pyramid, pyramid ,1):cat(pyramid ,1)
			end

			local multiscale  = model_12net:forward(pyramid)
			-- unpack pyramid , distributions will be table of tensors , oe -- for each scale of the sample image 
			local distributions = unpacker:forward(multiscale , coordinates)


			local val, ind, res = 0
			for j = 1,#distributions do 
				local boxes = {}

				distributions[j]:apply(math.exp)
				vals, ind = torch.max(distributions[j],1)
				ind_data = torch.data(ind)
				--collect pos candidates (with threshold p>0.5)
				local size = vals[1]:size(2)
				for t = 1,ind:nElement()-1 do 

					x_map = math.max(t%size,1)
					y_map = math.ceil(t/size)
					--converting to orig. sample coordinate
					x = math.max((x_map-1)*2 ,1)
					y = math.max((y_map-1)*2 ,1)

					if ind[1][y_map][x_map] == 1 then --prob. for a face 
						table.insert(boxes, {x,y,x+11,y+11,vals[1][y_map][x_map]})
					end
				end

				local pos_suspects_boxes = torch.Tensor(boxes)
				local nms_chosen_suspects = nms(pos_suspects_boxes, 0.01)

				if #nms_chosen_suspects:size() ~= 0 then
					pos_suspects_boxes = pos_suspects_boxes:index(1,nms_chosen_suspects)

					for p = 1,pos_suspects_boxes:size(1) do

						sus = torch.div(pos_suspects_boxes[p],scales[j])
						sus:apply(math.floor)
						croppedDetection = image.crop(img, sus[1], sus[2], sus[3], sus[4])
						croppedDetection = image.scale(croppedDetection, 24, 24)
						table.insert(false_positive_24_pascal_crops, croppedDetection:resize(1,3,24,24))

						--				for debugging Display image and get handle
						--				win:setcolor(1,0,0)
						--				win:fill()
						--				win:rectangle(sus[1], sus[2], sus[3]-sus[1], sus[4]-sus[2])
						--				win:stroke()
						--				win:setcolor(0,1,0)
						--				win:fill()
						--				win:rectangle(sus[1], sus[2], sus[3]-sus[1], sus[4]-sus[2])
						--				win:stroke()
						--				win:setcolor(0,0,1)
						--				win:fill()
						--				win:rectangle(sus[1], sus[2], sus[3]-sus[1], sus[4]-sus[2])
						--				win:stroke()

					end
				end
			end
		end
	end
	--This is 1.7gb file so be carefull!
	torch.save('false_positives.t7',false_positive_24_pascal_crops) 

end


------------------------------------------------------------------------------
-- LOADING DATA
------------------------------------------------------------------------------

local pos_data = torch.load('aflw_24_tensor.t7')
local pos_data_labels = torch.Tensor(pos_data:size(1)):fill(1) 

--loading neg examples from previously saved negative mining with Pascal false positive detections 24X24 patch
local m = nn.JoinTable(1)
--To generate data ans save to file - uncomment bellow line and comment above
GetNegatives()
local negative_data = m:forward(torch.load('false_positives.t7'))
local neg_data_labels = torch.Tensor(negative_data:size(1)):fill(2)
local data = torch.cat(negative_data:double(), pos_data:double(),1)
local labels = torch.cat(neg_data_labels:double(), pos_data_labels:double(),1)


------------------------------------------------------------------------------
-- MODEL
------------------------------------------------------------------------------

local model = nn.Sequential();
-- input 3x24x24
model:add(nn.SpatialConvolution(3, 64, 5, 5))
-- outputs 64x20x20
model:add(nn.SpatialMaxPooling(3, 3, 2, 2))
model:add(nn.ReLU())
-- outputs 64x8x8
model:add(nn.SpatialConvolution(64, 64, 9, 9))
model:add(nn.ReLU())
-- outputs 16x1x1
model:add(nn.SpatialConvolution(64, 2, 1, 1))
-- outputs 2x1x1
model:add(nn.SpatialSoftMax())
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
	
	collectgarbage()
	
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
	-- 4. use gradients to update weights, we'll understand this step more next week
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
	--print('prob face ' .. math.exp(output_test[1][1][1][1]))
	logger:add{train_losses[#train_losses], test_losses[#test_losses]}

end


model:double()

------------------------------------------------------------------------
--  PLOTTING TESTING/TRAINING LOSS/CLASSIFICATION ERRORS
------------------------------------------------------------------------
gnuplot.pdffigure('loss_24net.pdf')
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

torch.save ('model_24net.net', fmodel)

