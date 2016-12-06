require 'image'

local cjson = require 'cjson'
local utils = {}

function utils.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end

function utils.write_json(path, j)
  -- API reference http://www.kyne.com.au/~mark/software/lua-cjson-manual.html#encode
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end

function utils.getFreeGPU()
    -- select the most available GPU to train
    local nDevice = cutorch.getDeviceCount()
    local memSet = torch.Tensor(nDevice)
    for i=1, nDevice do
        local tmp, _ = cutorch.getMemoryUsage(i)
        memSet[i] = tmp
    end
    local _, curDeviceID = torch.max(memSet,1)
    return curDeviceID[1]
end

-- Pixelwise accuracy, input can be a batch, assuming 0 is ignored
function utils.pixelAccuracy(pred, anno)
    local pixel_correct = torch.sum(torch.cmul(torch.eq(pred, anno), anno:gt(0)))
    local pixel_labeled = torch.sum(torch.gt(anno,0))
    return pixel_correct/pixel_labeled, pixel_correct, pixel_labeled
end

return utils
