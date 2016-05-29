
require 'torch'   
require 'image'  
require 'nn'      
require 'optim'
require 'gnuplot'
require 'os'
require 'io'
require 'PyramidPacker'
require 'PyramidUnPacker'
require 'nms'


--------- Face Detector --------------

-- Create empty table to store file names:
files = {}
fh,err = io.open("../fddb/FDDB-folds/FDDB-fold-01.txt")
if err then print("broken file!"); return; end
while true do
	line = fh:read()
	if line == nil then break end
	table.insert(files,line)
end

--load images from files 
fddb_images = {}
local smallestImgDim = 100000
minDim = 0 --big enough number
for _,value in pairs(files) do
	img = image.load('../fddb/images/'..value..'.jpg')
	minDim = math.min(img:size(2),img:size(3))
	if minDim <= smallestImgDim then smallestImgDim = minDim end
	table.insert(fddb_images, img) 
end


--play with it to increase recall
scales = {} -- list of scales
for k =1 ,39 do
	local scale = 12/( smallestImgDim /20+ smallestImgDim *(k -1)/20)
	if scale * smallestImgDim < 12 then break end
	table.insert (scales , scale )
end


local model_12net = torch.load('../q1/model_12net.net'):double()
local model_24net = torch.load('../q3/model_24net.net'):double()

-- create pyramid packer and unpacker. scales is a table with all -- the scales you with to check. 
local unpacker_12 = nn.PyramidUnPacker(model_12net)
local packer_12 = nn.PyramidPacker(model_12net, scales) 

local fileOut = io.open('fold-01-out.txt', 'w')
io.output(fileOut)



for i = 1,#fddb_images do 

	local detections = {}

	collectgarbage()

	io.write(files[i]) --write image relative path
	io.write("\n")

	local img = fddb_images[i]

	-- create multiscale pyramid 
	local pyramid , coordinates = packer_12:forward(img)	
	if pyramid:size(1) == 1 then
		pyramid = torch.cat(pyramid, pyramid ,1):cat(pyramid ,1)
	end

	local multiscale  = model_12net:forward(pyramid)
	-- unpack pyramid , distributions will be table of tensors , one -- for each scale of the sample image 
	local distributions = unpacker_12:forward(multiscale , coordinates)
	local val, ind, res = 0
	local detections_12net = {}

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
		local nms_chosen_suspects = nms(pos_suspects_boxes, 0.5)

		if #nms_chosen_suspects:size() ~= 0 then
			pos_suspects_boxes = pos_suspects_boxes:index(1,nms_chosen_suspects)

			for p = 1,pos_suspects_boxes:size(1) do
				--scalling up suspected box to orig size image
				sus = torch.div(pos_suspects_boxes[p],scales[j])
				sus:apply(math.floor)

				croppedDetection = image.crop(img, sus[1], sus[2], sus[3], sus[4])
				croppedDetection = image.scale(croppedDetection, 24, 24)
				table.insert(detections_12net, {croppedDetection:resize(1,3,24,24), sus[1], sus[2], sus[3], sus[4]})

			end
		end
	end


	---- Use 24 net to run on each 12 net detection ---- 
	local detections_24net = {}

	for d = 1,#detections_12net do 

		local dist = model_24net:forward(detections_12net[d][1])

		if math.exp(dist[1][1][1][1]) > math.exp(dist[1][2][1][1]) then
			--image.scale(croppedDetection, 24, 24)
			table.insert(detections_24net,{detections_12net[d][2], detections_12net[d][3],
					detections_12net[d][4], detections_12net[d][5], math.exp(dist[1][1][1][1])})
		end

	end

	if #detections_24net ~= 0 then
		local pos_suspects_boxes = torch.Tensor(detections_24net)
		local nms_chosen_suspects = nms(pos_suspects_boxes, 0.5)

		if #nms_chosen_suspects:size() ~= 0 then
			pos_suspects_boxes = pos_suspects_boxes:index(1,nms_chosen_suspects)
		end

		for n=1, pos_suspects_boxes:size(1) do
			sus = pos_suspects_boxes[n]
			table.insert(detections,sus)

		end
	end

	io.write(#detections)
	io.write("\n")-- write number of detections

--find circles (simple elipse) enclosing each bounding box and report detections
	for d = 1,#detections do
		box = detections[d]
		radius = 0.5*math.sqrt(math.pow((box[3] - box[1]),2)+math.pow((box[4] - box[2]),2))
		centerX = box[1] + math.floor((box[3] - box[1])/2) 
		centerY = box[2] + math.floor((box[4] - box[2])/2)
		-- write detectiodetections_24netn in ellipse format
		io.write(radius ..' '.. radius ..' '.. 0 ..' '.. centerX ..' '.. centerY ..' '.. 1) 
		io.write("\n")
	end

end


io.close()



