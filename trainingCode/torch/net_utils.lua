local utils = require 'utils'
local net_utils = {}
require 'cutorch'

function net_utils.build_dilatednet(net, num_class)
  local model = nn.Sequential()

  for i = 1,36 do
    local layer = net:get(i)
    local name = layer.name
    local layer_type = torch.type(layer)

    local is_convolution = (layer_type == 'cudnn.SpatialConvolution' or layer_type == 'nn.SpatialConvolution')

    -- convert kernels in first conv layer into RGB format instead of BGR,
    if i == 1 then
      local w = layer.weight:clone()
      -- swap weights to R and B channels
      print('converting first layer conv filters from BGR to RGB...')
      
      layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
      layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
      model:add(layer)

    elseif (i==24 or i==26 or i==28) then
      local conv_layer = nn.SpatialDilatedConvolution(512,512,3,3,1,1,2,2,2,2)
      conv_layer.weight:copy(layer.weight)
      conv_layer.bias:copy(layer.bias)
      model:add(conv_layer)

    elseif i==30 then
      local conv_layer = nn.SpatialDilatedConvolution(512,4096,7,7,1,1,12,12,4,4)
      conv_layer.weight:copy(layer.weight)
      conv_layer.bias:copy(layer.bias)
      model:add(conv_layer)
    else
      model:add(layer)
    end
  end

  model:add(nn.SpatialUpSamplingBilinear(8))
  print(model)
  return model 
end

function net_utils.prepro(imgs, annotations, on_gpu)
  assert(on_gpu ~= nil)

  if on_gpu then imgs = imgs:cuda() else imgs = imgs:float() end
  if on_gpu then annotations = annotations:cuda() else annotations = annotations:float() end

  -- subtract mean
  ade_mean = torch.FloatTensor{124.6901, 118.6897, 109.5388}:view(1,3,1,1) -- in RGB order
  ade_mean = ade_mean:typeAs(imgs)
  imgs:add(-1, ade_mean:expandAs(imgs))

  return imgs, annotations
end

function net_utils.list_nngraph_modules(g)
  local om = {}
  for i,node in ipairs(g.forwardnodes) do
      local m = node.data.module
      if m then
        table.insert(om, m)
      end
   end
   return om
end

function net_utils.listModules(net)
  local t = torch.type(net)
  local moduleList
  if t == 'nn.gModule' then
    moduleList = net_utils.list_nngraph_modules(net)
  else
    moduleList = net:listModules()
  end
  return moduleList
end

function net_utils.sanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and m.gradWeight then
      m.gradWeight = nil
    end
    if m.bias and m.gradBias then
      m.gradBias = nil
    end
  end
end

function net_utils.unsanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and (not m.gradWeight) then
      m.gradWeight = m.weight:clone():zero()
    end
    if m.bias and (not m.gradBias) then
      m.gradBias = m.bias:clone():zero()
    end
  end
end

return net_utils
