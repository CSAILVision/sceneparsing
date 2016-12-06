require 'hdf5'
local utils = require 'utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
  -- open the hdf5 file
  print('DataLoader loading h5 file: ', opt.h5_file)
  self.h5_file = hdf5.open(opt.h5_file, 'r')
  self.info = utils.read_json(opt.json_file)

  -- extract image size from dataset
  local images_size = self.h5_file:read('/images'):dataspaceSize()
  assert(#images_size == 4, '/images should be a 4D tensor')
  assert(images_size[3] == images_size[4], 'width and height must match')
  self.num_images = images_size[1]
  self.num_channels = images_size[2]
  self.max_image_size = images_size[3]
  print(string.format('read %d images of size %dx%dx%d', self.num_images, 
            self.num_channels, self.max_image_size, self.max_image_size))

  -- extract annotation size from dataset
  local annotations_size = self.h5_file:read('/annotations'):dataspaceSize()
  assert(#annotations_size == 3, '/annotations should be a 3D tensor')
  assert(annotations_size[1] == images_size[1],'/images number should be equal to /annotations number')
  
  -- separate out indexes for each of the provided splits
  self.split_ix = {}
  self.iterators = {}
  for i,img in pairs(self.info.images) do
    local split = img.split
    if not self.split_ix[split] then
      -- initialize new split
      self.split_ix[split] = {}
      self.iterators[split] = 1
    end
    table.insert(self.split_ix[split], i)
  end
  for k,v in pairs(self.split_ix) do
    print(string.format('assigned %d images to split %s', #v, k))
  end
end

function DataLoader:resetIterator(split)
  self.iterators[split] = 1
end

function DataLoader:nextBatch(opt)
  local split = opt.split 
  local batch_size = opt.batch_size

  local split_ix = self.split_ix[split]
  assert(split_ix, 'split ' .. split .. ' not found.')

  -- pick an index of the datapoint to load next
  local img_batch_raw = torch.ByteTensor(batch_size, 3, self.max_image_size, self.max_image_size)
  local anno_batch_raw = torch.ByteTensor(batch_size, self.max_image_size, self.max_image_size)

  local max_index = #split_ix
  local wrapped = false

  for i=1,batch_size do
    local ri = self.iterators[split] -- get next index from iterator
    local ri_next = ri + 1 -- increment iterator
    if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
    self.iterators[split] = ri_next
    ix = split_ix[ri]
    assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)

    -- fetch the image from h5
    local img = self.h5_file:read('/images'):partial({ix,ix},{1,self.num_channels},
                            {1,self.max_image_size},{1,self.max_image_size})
    img = torch.squeeze(img)
    -- fetch the annotation from h5
    local anno = self.h5_file:read('/annotations'):partial({ix,ix},
                            {1,self.max_image_size},{1,self.max_image_size})
    anno = torch.squeeze(anno)

    -- data augmentation by flipping
    if split == 'train' then 
      if math.random(2) > 1 then
        image.hflip(img, img)
        image.hflip(anno, anno)
      end
    end

    img_batch_raw[i] = img
    anno_batch_raw[i] = anno

  end

  local data = {}
  data.images = img_batch_raw
  data.annotations = anno_batch_raw
  data.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
  return data
end
