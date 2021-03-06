--[[

This file trains a character-level multi-layer RNN on text data

Code is based on implementation in
    https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
    The practical6 code is in turn based on
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
local CharSplitLMMinibatchLoader = require 'util.CharSplitLMMinibatchLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local RNN = require 'model.RNN'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
local rho = config.rho or 0.9
local eps = config.eps or 1e-6
local wd = config.weightDecay or 0

cmd:option('-data_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')
cmd:option('-min_freq',50,'treat as unk any chars w/ count below this')
cmd:option('-over255',1,'allow for over 255 distinct characters (suggest aggressively language-filtering and unicode normalizing instead if possible - the resulting model will be better. this is for ~chinese mainly)')
cmd:option('-maxvocab',255,'limit character inventory to the top [this many] chars including unk and newline. if under 256 then the over255 option is irrelevant (i.e. only one byte is used)')
cmd:option('-vocabout','','if nonempty, file that the vocab chars are written to in order of vocab index. the unk char will be written U+FFFC ("object replacement character")')

-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm,gru or rnn (recommend lstm)')
cmd:option('-bn', 0, 'batch normalization for LSTM')
cmd:option('-nopeep', 0, '-model lstm: disable peepholes')
cmd:option('-couple_input', 0, 'TODO: -model lstm: set input gate = 1 - forget')
cmd:option('-no_input', 0, 'TODO: -model lstm: no input gate')
cmd:option('-forget_bias', 1.3, 'recommended >= 1 so you "learn to forget"')
cmd:option('-forget_bias_plusminus', 0.3, 'forget_bias plus or minus this random uniform')
cmd:option('-bn_eps', 1e-5, '-bn: epsilon')
cmd:option('-bn_momentum', 0.1, '-bn: momentum')
cmd:option('-bn_affine', 1, '-bn: affine (boolean)')
-- adadelta optimization
cmd:option('-adadelta', 0, 'use adadelta (uses more gpu memory)')
cmd:option('-rho_ada', 0.9, 'adadelta rho=0.9')
cmd:option('-eps_ada', 1e-6, 'adadelta eps=1e-6')
cmd:option('-wd_ada', 0, 'adadelta weight decay')
-- nag optimization
cmd:option('-nag', 1, 'use nesterov momentum (the default)')
cmd:option('-lr', .25, '-nag learning rate: default .25')
cmd:option('-momentum', .99, '-nag: default .99')
-- adam optimization
cmd:option('-adam', 0, 'use adam')
cmd:option('-beta1', .9, 'adam: first moment coeff')
cmd:option('-beta2', .999, 'adam: second moment coeff')
cmd:option('-epsilon', 1e-8, 'adam: epsilon')
cmd:option('-lrdecay', 0, 'adam: learningRateDecay')
cmd:option('-wdecay', 0, 'adam: weightDecay')
---
cmd:option('-learning_rate',2e-2,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',50,'number of timesteps to unroll for')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',0.1,'clip gradients at this value per seq_length')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
-- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',4000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-accurate_gpu_timing',0,'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
-- Scheduled Sampling
cmd:option('-use_ss', 1, 'whether use scheduled sampling during training')
cmd:option('-start_ss', 1, 'start amount of truth data to be given to the model when using ss')
cmd:option('-decay_ss', 0.005, 'ss amount decay rate of each epoch')
cmd:option('-min_ss', 0.9, 'minimum amount of truth data to be given to the model when using ss')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
math.randomseed(opt.seed)
-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac}

--[[ (from FAIR 'fairseq') A plain implementation of Nesterov's momentum
Implements Nesterov's momentum using the simplified
    formulation of https://arxiv.org/pdf/1212.0901.pdf
    ARGS:
    - `opfunc` : a function that takes a single input (X), the point
    of a evaluation, and returns f(X) and df/dX
    - `x`      : the initial point
    - `config` : a table with configuration parameters for the optimizer
    - `config.learningRate`      : learning rate
    - `config.momentum`          : momentum
    - `state`  : a table describing the state of the optimizer; after each
    call the state is modified
    - `state.evalCounter`        : evaluation counter (optional: 0, by default)
    RETURN:
    - `x`     : the new x vector
    - `f(x)`  : the function, evaluated before the update
    (Yann Dauphin, 2016)
]]
local function nag(opfunc, x, config, state)

    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-3
    local l2 = config.l2 or 0
    local mom = config.momentum or 0
    state.evalCounter = state.evalCounter or 0

    -- (1) evaluate f(x) and df/dx
    local fx,dfdx = opfunc(x)

    if not state.dfdx then
        state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):fill(0)
    end

    -- (2) weight decay
    if l2 ~= 0 then
        dfdx:add(l2, x)
    end

    -- (3) apply update
    x:add(mom*mom, state.dfdx):add(-(1 + mom) * lr, dfdx)

    -- (4) apply momentum
    state.dfdx:mul(mom):add(-lr, dfdx)

    -- (5) update evaluation counter
    state.evalCounter = state.evalCounter + 1

    -- return x*, f(x) before optimization
    return x,{fx}

end


-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        print('using OpenCL on GPU ' .. opt.gpuid .. '...')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
        print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- create the data loader class
local vocabout = opt.vocabout and opt.vocabout ~= '' and assert(io.open(opt.vocabout, "wb"))
local loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes, opt.min_freq, opt.over255, opt.maxvocab, vocabout)
local vocab_size = loader.vocab_size -- the number of distinct characters
local vocab = loader.vocab
print('vocab size: ' .. vocab_size)
-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
if string.len(opt.init_from) > 0 then
    print('loading a model from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    -- make sure the vocabs are the same
    local vocab_compatible = true
    local checkpoint_vocab_size = 0
    for c,i in pairs(checkpoint.vocab) do
        if not (vocab[c] == i) then
            vocab_compatible = false
        end
        checkpoint_vocab_size = checkpoint_vocab_size + 1
    end
    if not (checkpoint_vocab_size == vocab_size) then
        vocab_compatible = false
        print('checkpoint_vocab_size: ' .. checkpoint_vocab_size)
    end
    assert(vocab_compatible, 'error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.')
    -- overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ', model=' .. checkpoint.opt.model .. ' based on the checkpoint.')
    opt.rnn_size = checkpoint.opt.rnn_size
    opt.num_layers = checkpoint.opt.num_layers
    opt.model = checkpoint.opt.model
    do_random_init = false
else
    print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
    protos = {}
    if opt.model == 'lstm' then
        protos.rnn = LSTM.lstm(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout, opt)
    elseif opt.model == 'gru' then
        protos.rnn = GRU.gru(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elseif opt.model == 'rnn' then
        protos.rnn = RNN.rnn(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    end
    protos.criterion = nn.ClassNLLCriterion()
end

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
    if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(init_state, h_init:clone())
    if opt.model == 'lstm' then
        table.insert(init_state, h_init:clone())
    end
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 and opt.opencl == 0 then
    for k,v in pairs(protos) do v:cuda() end
end
if opt.gpuid >= 0 and opt.opencl == 1 then
    for k,v in pairs(protos) do v:cl() end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

-- initialization
if do_random_init then
    params:uniform(-0.08, 0.08) -- small uniform numbers
end
-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
if opt.model == 'lstm' then
    for layer_idx = 1, opt.num_layers do
        for _,node in ipairs(protos.rnn.forwardnodes) do
            if node.data.annotations.name == "i2h_" .. layer_idx then
                print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
                forget_biases = node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]
                local m = opt.forget_bias
                local d = opt.forget_bias_plusminus
                forget_biases:uniform(m - d , m + d) -- don't forget until you give remembering a chance
                -- http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf - and
                -- if you didn't already learn a bias, a constant bias of 1
                -- would be useful (for forget gates only).
            end
        end
    end
end

print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- preprocessing helper function
function prepro(x,y)
    x = x:transpose(1,2):contiguous() -- swap the axes for faster indexing
    y = y:transpose(1,2):contiguous()
    if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
        x = x:cl()
        y = y:cl()
    end
    return x,y
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state}

    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = loader:next_batch(split_index)
        x,y = prepro(x,y)
        -- forward pass
        for t=1,opt.seq_length do
            clones.rnn[t]:evaluate() -- for dropout proper functioning
            local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
            prediction = lst[#lst]
            loss = loss + clones.criterion[t]:forward(prediction, y[t])
        end
        -- carry over lstm state
        rnn_state[0] = rnn_state[#rnn_state]
        -- print(i .. '/' .. n .. '...')
    end

    loss = loss / opt.seq_length / n
    print('loss['..split_index..'] = '..loss)
    return loss
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
    x,y = prepro(x,y)
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        if opt.use_ss == 1 and t > 1 and math.random() > ss_current then
            local probs = torch.exp(predictions[t-1]):squeeze()
            _,samples = torch.max(probs,2)
            xx = samples:view(samples:nElement())
        else
            xx = x[t]
        end
        -- print(x[{{},t}])
        local lst = clones.rnn[t]:forward{xx, unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        loss = loss + clones.criterion[t]:forward(predictions[t], y[t])
    end
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
for t=opt.seq_length,1,-1 do
    -- backprop through loss, and softmax/linear
    local doutput_t = clones.criterion[t]:backward(predictions[t], y[t])
    table.insert(drnn_state[t], doutput_t)
    local dlst = clones.rnn[t]:backward({x[t], unpack(rnn_state[t-1])}, drnn_state[t])
    drnn_state[t-1] = {}
    for k,v in pairs(dlst) do
        if k > 1 then -- k == 1 is gradient on x, which we dont need
            -- note we do k-1 because first item is dembeddings, and then follow the
            -- derivatives of the state, starting at index 2. I know...
            drnn_state[t-1][k-1] = v
        end
    end
end
------------------------ misc ----------------------
-- transfer final state to initial state (BPTT)
init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
grad_params:div(opt.seq_length)
-- clip gradient element-wise
grad_params:clamp(-opt.grad_clip, opt.grad_clip)
return loss, grad_params
end

-- start optimization here
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil
local adaconfig = {rho = opt.rho_ada, eps = opt.eps_ada, wd= opt.wd_ada}
local adastate = {}
local nagparams = {momentum = opt.momentum, learningRate = opt.lr}
local nagstate = {}
local adamparams = {beta1 = opt.beta1, beta2 = opt.beta2, learningRate = opt.learning_rate, learningRateDecay = opt.lrdecay, weightDecay = opt.wdecay, epsilon = opt.epsilon}
local adamstate = {}
ss_current = opt.start_ss

local function false0(x)
    return not (not x or x == 0)
end

local adadelta = false0(opt.adadelta)
local adam = false0(opt.adam)
local usenag = false0(opt.nag)
local rmsprop = not (adadelta or adam or nag)
for i = 1, iterations do
    local epoch = i / loader.ntrain

    local timer = torch.Timer()
    local loss
    if adadelta then
        _, loss = optim.adadelta(feval, params, adaconfig, adastate)
    elseif adam then
        _, loss = optim.adam(feval, params, adamconfig, adamstate)
    elseif usenag then
        _, loss = nag(feval, params, nagconfig, nagstate)
    else
        assert(rmsprop)
        _, loss = optim.rmsprop(feval, params, optim_state)
    end
    if opt.accurate_gpu_timing == 1 and opt.gpuid >= 0 then
        --[[
            Note on timing: The reported time can be off because the GPU is invoked async. If one
            wants to have exactly accurate timings one must call cutorch.synchronize() right here.
            I will avoid doing so by default because this can incur computational overhead.
        --]]
        cutorch.synchronize()
    end
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if rmsprop and i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- decay schedule sampling amount
    if opt.use_ss == 1 and i % loader.ntrain == 0 and ss_current > opt.min_ss then
        ss_current = opt.start_ss - opt.decay_ss * epoch
        print('decay schedule sampling amount to ' .. ss_current)
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        break -- halt
    end
end
