require 'torch'
require 'nn'
require 'loadcaffe'
require 'optim'

local utils = require 'utils'
local net_utils = require 'net_utils'
require 'DataLoader'

-------------------------------------------------------------------------------
-- Input arguments
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')

cmd:option('-model_id','dilated','model id')
cmd:option('-proto','DilatedNet.prototxt','path to prototxt file in Caffe.')
-- pretrained DilatedNet model at http://sceneparsing.csail.mit.edu/model/DilatedNet_iter_120000.caffemodel
cmd:option('-model','DilatedNet_iter_120000.caffemodel','path to trained Caffe models.')
cmd:option('-input_h5','data_ADE20K_384x384.h5','path to the h5 file containing the preprocessed dataset')
cmd:option('-input_json','data_ADE20K_384x384.json','path to the json file containing image name and split info')
cmd:option('-start_from', '', 'model to initialize model weights from.')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-seed', 1234, 'random number generator seed to use')
cmd:option('-gpuid', 3, 'which gpu to use. -1 = CPU')
cmd:option('-id', 1)

-- Segmentation parameters
cmd:option('-num_class',151, '1(other) + 150')
cmd:option('-max_iters', -1, 'max iterations, -1 = forever')
cmd:option('-batch_size', 5,'batch size')

-- Optimization parameters
cmd:option('-learning_rate',1e-3,'learning rate')
cmd:option('-momentum', 0.9,'momentum')
cmd:option('-weight_decay', 5e-4, 'L2 weight decay')

-- Evaluation/Checkpointing
cmd:option('-val_images_use', 50, 'number of images to evaluatate during training')
cmd:option('-save_checkpoint_every', 100, 'how often to save a model')
cmd:option('-checkpoint_path', 'snapshot', 'folder to save checkpoints')
cmd:option('-losses_log_every', 20)

cmd:text()
-------------------------------------------------------------------------------
-- Basic initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  local gpuidx = utils.getFreeGPU()
  print('use GPU IDX=' .. gpuidx)
  cutorch.setDevice(gpuidx)
end

paths.mkdir(opt.checkpoint_path)

-------------------------------------------------------------------------------
-- Data Loader and Optim configurations
-------------------------------------------------------------------------------
local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}
local optimConfig = {learningRate = opt.learning_rate, learningRateDecay = 0,  momentum = opt.momentum, dampening=0, weightDecay=opt.weight_decay}

-------------------------------------------------------------------------------
-- Load segmentation CNN
-------------------------------------------------------------------------------
local cnn_backend = opt.backend
if opt.gpuid == -1 then cnn_backend = 'nn' end -- override to nn if gpu is disabled

if string.len(opt.start_from) > 0 then
	print('initializing weights from ' .. opt.start_from)
	local loaded_checkpoint = torch.load(opt.start_from)
	cnn = loaded_checkpoint.cnn
	net_utils.unsanitize_gradients(cnn)
else

local model_caffe = loadcaffe.load(opt.proto, opt.model, cnn_backend)
cnn = net_utils.build_dilatednet(model_caffe, opt.num_class)

local label_weights = torch.ones(opt.num_class)
-- the first class is ignored
label_weights[1] = 0.
crit = cudnn.SpatialCrossEntropyCriterion(label_weights)

-- ship everything to GPU
if opt.gpuid >= 0 then
  cnn:cuda()
  crit:cuda()
end

local net_params, net_grads = cnn:getParameters()
local thin_cnn = cnn:clone('weight', 'bias')
net_utils.sanitize_gradients(thin_cnn)
collectgarbage() 

-------------------------------------------------------------------------------
-- Evaluation
-------------------------------------------------------------------------------
local function evalFunc()
  local split = 'val'
  cnn:evaluate()
  loader:resetIterator(split)
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local pixel_correct = 0
  local pixel_labeled = 0

  while true do

    -- fetch a batch of data
    local data = loader:nextBatch{batch_size = opt.batch_size, split = split}
    data.images, data.annotations = net_utils.prepro(data.images, data.annotations, opt.gpuid >= 0)
    n = n + data.images:size(1)

    -- forward the model to get loss
    local feats = cnn:forward(data.images)
    local loss = crit:forward(feats, data.annotations)

    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1

    -- turn the prob mask into semantic segmentation mask
    local _, predMasks = torch.max(feats, 2)
    predMasks = torch.squeeze(predMasks)
    predMasks = predMasks:byte()
    data.annotations = data.annotations:byte()

    -- calculate accuracy
    local _, cur_correct, cur_labeled = utils.pixelAccuracy(predMasks-1, data.annotations-1)
    pixel_correct = pixel_correct + cur_correct
    pixel_labeled = pixel_labeled + cur_labeled

    -- check if we evaluate till the end
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, opt.val_images_use)
    print(string.format('evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss))

    if loss_evals % 10 == 0 then collectgarbage() end
    if n >= opt.val_images_use then break end
  end

  collectgarbage() 

  return loss_sum/loss_evals, pixel_correct/pixel_labeled
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 0
local function lossFunc()
  cnn:training()
  net_grads:zero()

  -- get a batch of data  
  local data = loader:nextBatch{batch_size = opt.batch_size, split = 'train'}
  data.images, data.annotations = net_utils.prepro(data.images, data.annotations, opt.gpuid >= 0) -- preprocess in place, do data augmentation

  -- forward the network
  local feats = cnn:forward(data.images)
  local loss = crit:forward(feats, data.annotations)

  -- backprop
  local dfeats = crit:backward(feats, data.annotations)
  local dx = cnn:backward(data.images, dfeats)

  local losses = { total_loss = loss }
  return losses
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss_history = {}
local val_loss_history = {}
local best_loss

-- define function for optimizer
local function opfunc()
  return crit.output, net_grads
end

while true do  

  -- compute loss/gradient
  local losses = lossFunc()
  if iter % opt.losses_log_every == 0 then 
    loss_history[iter] = losses.total_loss
  	print(string.format('Iter %d: loss=%f', iter, losses.total_loss))
  end

  -- save checkpoint
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then
    -- evaluate the validation performance
    local val_loss, val_accuracy = evalFunc()
    print('validation loss: ', val_loss)
    print('validation accuracy: ', val_accuracy)
    val_loss_history[iter] = val_loss

    local checkpoint_path = path.join(opt.checkpoint_path, 'model_'.. opt.model_id ..'_' .. opt.id)

    -- write json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.val_loss_history = val_loss_history

    utils.write_json(checkpoint_path .. '.json', checkpoint)
    print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    if best_loss == nil or val_loss < best_loss then
      best_loss = val_loss
      if iter > 0 then
        checkpoint.cnn = thin_cnn
        torch.save(checkpoint_path .. '.t7', checkpoint)
        print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
      end
    end
  end

  -- update parameters
  optim.sgd(opfunc, net_params, optimConfig)

  -- stopping criteria
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end

end
end
