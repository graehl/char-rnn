--[[

    This file samples characters from a trained model

    Code is based on implementation in
    https://github.com/oxford-cs-ml-2015/practical6

    Changes (raymondhs):

    Beam search are parallelized on GPU
    UTF8 character handling

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'

local stringx = require('pl.stringx')
utf8 = require 'lua-utf8'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-primetext',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',3000,'max number of characters to sample (input/primetext must be shorter)')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:option('-beamsize',1,'defaults to 1')
cmd:option('-stop','\n\n\n\n\n','stop sampling when this string (5 newlines by default) is detected')
cmd:option('-stdin',1,"if true, ignore primetext and take stdin and generate (beam search) alternative capitalizations")
cmd:text()

-- parse input params
opt = cmd:parse(arg)

local UNK='<unk>'

function utf8chars(str)
    local t = {}
    for i in string.gfind(str, "([%z\1-\127\194-\244][\128-\191]*)") do
        table.insert(t, i)
    end
    return t
end

function vdot(v, n)
    if math.fmod(n, 10) == 0 and opt.verbose >= v then
        io.stderr:write('.')
        if math.fmod(n, 1000) == 0 then io.stderr:write(n .. '\n') end
        io:flush()
    end
end

-- gated print: simple utility function wrapping a print
function vprint(v, str)
    if opt.verbose >= v then
        str = str:gsub('\n', '<n>')
        io.stderr:write(str .. '\n')
        io:flush()
    end
end

function gprint(str)
    vprint(1, str)
end

function vprint2(v, ...)
    if opt.verbose >= v then
        vprint(v, table.concat({...}, ''))
    end
end

function fatal(str)
    opt.verbose = 1
    require 'pl.pretty'.dump(str)
    assert(not "FATAL ERROR: " .. str)
end

function clone_state(cc)
    new_state = {}
    if cc ~= nil then
        for L = 1,table.getn(cc) do
            -- c and h for all layers
            table.insert(new_state, cc[L]:clone())
        end
    else
        new_state = nil
    end
    return new_state
end

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then gprint('package cunn not found!') end
    if not ok2 then gprint('package cutorch not found!') end
    if ok and ok2 then
        gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
        gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- check that clnn/cltorch are installed if user wants to use OpenCL
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        gprint('using OpenCL on GPU ' .. opt.gpuid .. '...')
        gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
local nvocab = 0
for c,i in pairs(vocab) do
    ivocab[i] = c
    nvocab = nvocab + 1
end
for i = 1,#ivocab do
    vprint2(2, ivocab[i], ' ', i)
end
local VUNK = vocab[UNK]
gprint("vocab has " .. nvocab .. " items including unk/newline")
if VUNK == nil then
    fatal("ERROR: '" .. UNK .. "' not in vocab")
end

-- initialize the rnn state to all zeros
gprint('creating an ' .. checkpoint.opt.model .. '...')
local current_state
current_state = {}
for L = 1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
    if opt.gpuid >= 0 and opt.opencl == 0 then h_init = h_init:cuda() end
    if opt.gpuid >= 0 and opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(current_state, h_init:clone())
    if checkpoint.opt.model == 'lstm' then
        table.insert(current_state, h_init:clone())
    end
end

state_size = #current_state
default_state = clone_state(current_state)

local seed_text = opt.primetext
if not opt.stdin and not seed_text then
    gprint('missing seed text, using uniform probability over first character')
    gprint('--------------------------')
end

prediction = torch.Tensor(1, #ivocab):fill(1)/(#ivocab)
if opt.gpuid >= 0 and opt.opencl == 0 then prediction = prediction:cuda() end
if opt.gpuid >= 0 and opt.opencl == 1 then prediction = prediction:cl() end

if opt.stdin then
    vprint2(1, "recasing stdin (. = 10 lines, line of . = 1000 lines)")
    -- start sampling/argmaxing
    local lineno = 0
    local lineagain = nil
    while true do
        local line
        if lineagain == nil then
            line = io.read()
            if line == nil then break end
            line = line .. '\n'
            if lineno == 0 then
                lineagain = line
            end
        else
            line = lineagain
            lineagain = nil
        end

        vprint2(3, 'processing ', line)
        chars = utf8chars(line)

        beamsize = opt.beamsize
        beamState = {}
        beamScore = {}

        beamString = {} -- index to string
        beamLastChar = {}
        beamState[1] = clone_state(current_state)
        beamScore[1] = 0
        beamString[1] = ''
        vprint2(3, '#chars to process = ', #chars)
        prev_char = nil
        lineno = lineno + 1
        vdot(1, lineno)
        for ii = 1,#chars do
            newBeamState = {}
            newBeamScore = {}
            newBeamString = {}
            newBeamLastChar = {}
            scores = {}
            beam_index = 1

            beamSize = 0
            for _, _ in pairs(beamState) do beamSize = beamSize+1 end
            prev_chars = torch.CudaTensor(beamSize)
            current_states = {}
            for i = 1, state_size do
                table.insert(current_states, torch.CudaTensor(beamSize, checkpoint.opt.rnn_size))
            end

            cnt=1

            for cc, vv in pairs(beamState) do
                current_str = beamString[cc]
                current_state = vv
                current_score = beamScore[cc]

                strlen = utf8.len(current_str)
                if strlen > 0 then
                    local vbb = beamLastChar[cc]
                    prev_char = torch.Tensor{vbb}
                    prev_chars[cnt] = vbb
                else
                    prev_char = nil
                end
                for i=1,state_size do
                    current_states[i][cnt]:copy(current_state[i])
                end

                cnt = cnt+1
            end

            cnt = 1

            new_states = {}
            predictions = nil
            if prev_char ~= nil then
                local lst = protos.rnn:forward{prev_chars, unpack(current_states)}
                for i=1,state_size do table.insert(new_states, lst[i]) end
                predictions = lst[#lst] -- last element holds the log probabilities
                predictions:div(opt.temperature) -- scale by temperature
            end

            for cc, vv in pairs(beamState) do
                current_str = beamString[cc]
                current_state = vv
                current_score = beamScore[cc]
                local ci = chars[ii]
                local vci = vocab[ci] or VUNK
                candidates = {vci}
                if vci ~= VUNK then
                    local uci = utf8.upper(ci)
                    if uci ~= ci then
                        local vuci = vocab[uci]
                        vprint2(2, '#', ii, ' char(', ci, ')=>', vci, ' uchar(', uci, ')=>', vuci, ' currentsco= ', current_score, ')')
                        table.insert(candidates, vuci)
                    end
                end
                for jj = 1,#candidates do
                    vc = candidates[jj]
                    c = assert(ivocab[vc])
                    this_char = torch.Tensor{vc}
                    if prev_char ~= nil then
                        if this_char ~= nil then
                            newstr = current_str .. c
                            newsco = current_score + predictions[cnt][this_char[1]]
                        end
                        new_state = {}
                        local state = torch.zeros(1, checkpoint.opt.rnn_size)
                        for i=1,state_size do
                            table.insert(new_state, state:clone():copy(new_states[i][cnt]))
                        end
                        vprint2(2, '\ttesting \'', c, '\' score = ', predictions[cnt][this_char[1]] )
                    else
                        vprint2(2, '\ttesting \'', c, '\'')
                        new_state = default_state
                        newstr = current_str .. c
                        newsco = current_score
                    end
                    newBeamState[beam_index] = clone_state(new_state)
                    newBeamScore[beam_index] = newsco
                    newBeamString[beam_index] = newstr
                    newBeamLastChar[beam_index] = vc
                    beam_index = beam_index + 1
                    table.insert(scores, newsco)
                end
                cnt = cnt + 1
            end

            table.sort(scores)
            beamState = {}
            beamScore = {}
            beamString = {}
            beamLastChar = {}

            tid = #scores - beamsize + 1
            if tid < 1 then tid = 1 end
            threshold = scores[tid]
            for cc,vv in pairs(newBeamScore) do
                vprint2(2, 'Beam State:(', cc, ',', vv, ') threshold=', threshold, ')')
                if vv >= threshold then
                    beamState[cc] = newBeamState[cc]
                    beamScore[cc] = newBeamScore[cc]
                    beamString[cc] = newBeamString[cc]
                    beamLastChar[cc] = newBeamLastChar[cc]
                end
            end
        end
        threshold = scores[#scores]
        beststr = nil
        for cc,vv in pairs(newBeamScore) do
            if vv == threshold then
                beststr = newBeamString[cc]
                current_state = newBeamState[cc]
            end
        end
        if lineagain == nil then
            io.write(beststr)
            io.flush()
        end
    end
else
    -- start sampling/argmaxing
    result = ''
    for i=1, opt.length do

        -- log probabilities from the previous timestep
        -- make sure the output char is not UNKNOW
        if opt.sample == 0 then
            -- use argmax
            local _, prev_char_ = prediction:max(2)
            prev_char = prev_char_:resize(1)
        else
            -- use sampling
            real_char = UNK
            while(real_char == UNK) do
                prediction:div(opt.temperature) -- scale by temperature
                local probs = torch.exp(prediction):squeeze()
                probs:div(torch.sum(probs)) -- renormalize so probs sum to one
                prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
                real_char = ivocab[prev_char[1]]
            end
        end

        -- forward the rnn for next character
        local lst = protos.rnn:forward{prev_char, unpack(current_state)}
        current_state = {}
        for i=1,state_size do table.insert(current_state, lst[i]) end
        prediction = lst[#lst] -- last element holds the log probabilities

        -- io.write(ivocab[prev_char[1]])
        result = result .. ivocab[prev_char[1]]

        -- in my data, five \n represent the end of each document
        -- so count \n to stop sampling
        if string.find(result, opt.stop) then break end
    end
end
