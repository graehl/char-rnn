
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits

require './misc.lua'

local CharSplitLMMinibatchLoader = {}
CharSplitLMMinibatchLoader.__index = CharSplitLMMinibatchLoader

function strvocab(indices, ivocab, from, to)
    local t = {}
    for i = from, to do
        local id = indices[i]
        --print('['..i..']='..id..' => '..ivocab[id])
        table.insert(t, ivocab[id])
    end
    return table.concat(t)
end

function wholestrvocab(indices, ivocab)
    return strvocab(indices, ivocab, 1, indices:size(1))
end

function assertTorchIndices(tensor, max, chunk, name, ivocab)
    -- assert(torch.all(torch.cmin(torch.ge(s, 1), torch.le(s, max)))) - ran oom doing this
    -- instead: chunkwise
    local t = torch.view(tensor, -1)
    local sz = t:size(1)
    local from = 1
    while from <= sz do
        local to = from + chunk - 1
        if to > sz then
            to = sz
        end
        local lens = to - from + 1
        local s = t:sub(from, to)
        local check = torch.ge(s, 1)
        local le = torch.le(s, max)
        check:cmin(le)
        assert(torch.all(check))
        from = to + 1
    end
end

function CharSplitLMMinibatchLoader.create(data_dir, batch_size, seq_length, split_fractions, min_freq, over255, maxvocab, outivocab)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}
    if not over255 and maxvocab >= 256 then
        print('WARNING: -over255 0 and -maxvocab '..maxvocab..' disagree. reducing maxvocab to 255.')
        maxvocab = 255
    end
    if maxvocab <= 255 then
        over255 = false
    end
    local self = {}
    setmetatable(self, CharSplitLMMinibatchLoader)

    local input_file = path.join(data_dir, 'input.txt')
    local vocab_file = path.join(data_dir, 'vocab.t7')
    local tensor_file = path.join(data_dir, 'data.t7')

    -- rhs: validation file
    self.has_val_data = true
    val_data = true
    local val_input_file = path.join(data_dir, 'val_input.txt')
    local val_tensor_file = path.join(data_dir, 'val_data.t7')
    if not path.exists(val_input_file) then
        self.has_val_data = false
    end

    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    local vocab = nil
    local traindata = nil
    local valdata = nil
    if not (path.exists(vocab_file) and path.exists(tensor_file)) then
        -- prepro files do not exist, generate them
        print('.t7 do not exist. Running preprocessing...')
        run_prepro = true
    else
        -- check if the input file was modified since last time we
        -- ran the prepro. if so, we have to rerun the preprocessing
        local input_attr = lfs.attributes(input_file)
        local vocab_attr = lfs.attributes(vocab_file)
        local tensor_attr = lfs.attributes(tensor_file)
        if not vocab_attr or not tensor_attr or input_attr.modification > vocab_attr.modification or input_attr.modification > tensor_attr.modification then
            print('vocab.t7 or data.t7 detected as stale. Re-running preprocessing...')
            run_prepro = true
        end
    end
    run_prepro = run_prepro or self.has_val_data and not path.exists(val_tensor_file)
    if run_prepro then
        -- construct a tensor with all the data, and vocab file
        print('one-time setup: preprocessing input text file ' .. input_file .. '...')
        vocab, traindata = CharSplitLMMinibatchLoader.text_to_tensor(input_file, vocab_file, tensor_file, min_freq, vocab, over255, maxvocab)
        assert(vocab)
        assert(traindata)
        -- rhs: tensor with val data
        if self.has_val_data then
            _, valdata = CharSplitLMMinibatchLoader.text_to_tensor(val_input_file, nil, val_tensor_file, min_freq, vocab, over255, maxvocab)
        end
    end

    print('loading data files...')
    local data = traindata or torch.load(tensor_file)
    vocab = vocab or torch.load(vocab_file)
    self.vocab = vocab

    -- cut off the end so that it divides evenly
    local len = data:size(1)
    local multiples = batch_size * seq_length
    local ivocab, nv = invertTable(vocab)
    if outivocab then
        outivocab:write(table.concat(ivocab))
    end
    assert(nv >= 2)
    self.vocab_size = nv
    if len % (batch_size * seq_length) ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        data = data:sub(1, multiples * math.floor(len / multiples))
        assertTorchIndices(data, nv, 100, "training", ivocab)
    end

    local val_data
    -- load val data
    if self.has_val_data then
        val_data = valdata or torch.load(val_tensor_file)
        len = val_data:size(1)
        assert(len > multiples)
        if len % multiples ~= 0 then
            print('cutting off end of data so that the batches/sequences divide evenly')
            val_data = val_data:sub(1, batch_size * seq_length
                                        * math.floor(len / multiples))
            assertTorchIndices(val_data, nv, 100, "validation", ivocab)
        end
    end


    -- self.batches is a table of tensors
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length

    local ydata = data:clone()
    ydata:sub(1,-2):copy(data:sub(2,-1))
    ydata[-1] = data[1]
    self.x_batches = data:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    self.nbatches = #self.x_batches
    self.y_batches = ydata:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    assert(#self.x_batches == #self.y_batches)

    -- same thing for val data
    local val_ydata
    if self.has_val_data then
        val_ydata = val_data:clone()
        val_ydata:sub(1,-2):copy(val_data:sub(2,-1))
        val_ydata[-1] = val_data[1]
        self.val_x_batches = val_data:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
        self.val_nbatches = #self.val_x_batches
        self.val_y_batches = val_ydata:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
        assert(#self.val_x_batches == #self.val_y_batches)
    end

    -- lets try to be helpful here
    if self.nbatches < 50 then
        print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
    end

    -- perform safety checks on split_fractions
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
    assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')

    if self.has_val_data then
        self.ntrain = self.nbatches
        self.nval = self.val_nbatches
        self.ntest = 0
    else
        if split_fractions[3] == 0 then
            -- catch a common special case where the user might not want a test set
            self.ntrain = math.floor(self.nbatches * split_fractions[1])
            self.nval = self.nbatches - self.ntrain
            self.ntest = 0
        else
            -- divide data to train/val and allocate rest to test
            self.ntrain = math.floor(self.nbatches * split_fractions[1])
            self.nval = math.floor(self.nbatches * split_fractions[2])
            self.ntest = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
        end
    end

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}

    print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
    collectgarbage()
    return self
end

function CharSplitLMMinibatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function CharSplitLMMinibatchLoader:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if not (self.has_val_data) then
        if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
        if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
        return self.x_batches[ix], self.y_batches[ix]
    else
        if split_index == 1 then
            return self.x_batches[ix], self.y_batches[ix]
        else
            return self.val_x_batches[ix], self.val_y_batches[ix]
        end
    end
end

local bytemarkers = { {0x7FF,192}, {0xFFFF,224}, {0x1FFFFF,240} }
function utf8(decimal)
    if decimal<128 then return string.char(decimal) end
    local charbytes = {}
    for bytes,vals in ipairs(bytemarkers) do
        if decimal<=vals[1] then
            for b=bytes+1,2,-1 do
                local mod = decimal%64
                decimal = (decimal-mod)/64
                charbytes[b] = string.char(128+mod)
            end
            charbytes[1] = string.char(vals[2]+decimal)
            break
        end
    end
    return table.concat(charbytes)
end

function utf8frompoints(...)
  local chars,arg={},{...}
  for i,n in ipairs(arg) do chars[i] = utf8(arg[i]) end
  return table.concat(chars)
end

local UNK = "\255\252" -- ff fc object replacement
-- *** STATIC method ***
function VisitUtf8Chars(f, unordered)
    local len = 0
    local line
    local nlines = 0
    while true do
        line = f:read()
        if line == nil then break end -- no more lines to read
        local chars = UTF8ToCharArray(line)
        for i = 1, #chars do
            if unordered then
                local char = chars[i]
                unordered[char] = (unordered[char] or 0) + 1
            end
            len = len + 1
        end
        nlines = nlines + 1
        len = len + 1
    end
    if unordered then
        unordered[UNK] = 1e31
        unordered['\n'] = nlines
    end
    return len
end

function VisitUtf8CharsOpen(filename, unordered)
    f = assert(io.open(filename, "r"))
    local n = VisitUtf8Chars(f, unordered)
    print('text ' .. filename .. ' had ' .. n .. ' chars including [newline]')
    f:close()
    return n
end

-- *** STATIC method ***
function CharSplitLMMinibatchLoader.text_to_tensor(in_textfile, out_vocabfile, out_tensorfile, min_freq, vocabalready, over255, maxvocab)
    local tensorType = torch.ByteTensor
    if over255 then tensorType = torch.ShortTensor end
    local timer = torch.Timer()
    print('loading text file...')
    local cache_len = 10000
    local len = 0

    local vocab
    -- create vocabulary if it doesn't exist yet
    if vocabalready or path.exists(out_vocabfile) then
        if outvocabfile then print(out_vocabfile .. ' found') end
        vocab = vocabalready or torch.load(out_vocabfile)
        len = VisitUtf8CharsOpen(in_textfile, nil)
        -- code snippets taken from http://lua-users.org/wiki/LuaUnicode
    else
        print('creating vocabulary mapping...')
        -- record all characters to a set
        local count = {}
        len = VisitUtf8CharsOpen(in_textfile, count)

        local ordered = {}
        -- sort into a table (i.e. keys become 1..N)
        local allfreq = 0
        for char, c in pairs(count) do
            if c >= min_freq then
                table.insert(ordered, char)
            end
            allfreq = allfreq + 1
        end
        table.sort(ordered, function(a, b) return count[a] > count[b] end)
        vocab = {}
        -- invert `ordered` to create the char->int mapping
        for i, char in ipairs(ordered) do
            if i > maxvocab then
                break
            end
            vocab[char] = i
        end
        local nkept = #ordered
        print(allfreq .. ' char types observed; ' .. nkept .. ' with count >= ' .. min_freq)
        print((allfreq - nkept) .. ' char types with count < ' .. min_freq)
        print('saving ' .. out_vocabfile)
        assert(nkept > 2)
        torch.save(out_vocabfile, vocab)
    end
    vocab_size = 0
    for _ in pairs(vocab) do
        vocab_size = vocab_size + 1
    end
    if vocab_size < 256 then
        tensorType = torch.ByteTensor
    end
    print('vocab has ' .. vocab_size .. ' symbols including <unk> and [newline]')
    -- construct a tensor with all the data
    print('putting data into tensor...')
    local data = tensorType(len) -- store it into 1D first, then rearrange
    local currlen = 1

    local f = assert(io.open(in_textfile, "r"))
    local line
    local VUNK = assert(vocab[UNK])
    while true do
        line = f:read()
        if line == nil then break end -- no more lines to read
        local chars = UTF8ToCharArray(line)
        for i = 1, #chars do
            local char = chars[i]
            data[currlen] = vocab[char] or VUNK
            if currlen == 1 then
                print('first char='..char..' index='..data[currlen])
            end
            currlen = currlen + 1
        end
        -- don't forget end of line character it is excluded by f:read()
        data[currlen] = vocab['\n']
        currlen = currlen + 1
        if math.fmod(currlen, 10000) == 0 then
            print('read ' .. currlen .. ' / ' .. len)
        end
    end
    f:close()

    -- save output preprocessed files
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, data)
    return vocab, data
end

return CharSplitLMMinibatchLoader
