--[[
n = # of time steps unrolled?

config: {bn=False --enable/disable batch norm
         ,bn_momentum=0.1 -- bn momentum
         ,bn_eps=1e-5 -- bn epsilon
         ,bn_affine=True -- bn learn some linear stuff
         ,nopeep=True -- remove peephole connections p* ⊙ ct: doesn't hurt perf.
         ,couple_input=False -- (forget) it = 1-ft (input) gate. supposedly doesn't hurt perf (task dep?).
         ,no_input=False -- don't use it (input gate) (bad for seq->seq, ok for LM?)
        }

' = prev timestep
zt = g(Wz xt + Rz yt' + bz) [g,h = tanh] usually
it = σ(Wi xt + Ri yt' + pi ⊙ ct' + bi)
ft = σ(Wf xt + Rf yt' + pf ⊙ ct' + bf)
ct = it⊙zt + ft⊙ct'
ot = σ(Wo xt + R yt' + p⊙ct + bo)
yt = ot⊙h(ct)
]]--
local LSTM = {}

local function false0(x)
    return not (not x or x == 0)
end

function LSTM.lstm(input_size, rnn_size, n, dropout, config)
    dropout = dropout or 0
    local noconfig = not config
    local bn = false0(noconfig and 0 or config.bn)
    local no_input = false0(noconfig and 0 or config.no_input)
    local nopeep = (config and false0(config.nopeep)) or True
    local couple_input = false0(noconfig and 0 or config.couple_input)
    local bn_affine = false0(noconfig and 0 or config.bn_affine)
    local bn_momentum = (config and config.bn_momentum) or 0.1
    local bn_eps = (config and config.bn_eps) or 1e-5
    -- there will be 2*n+1 inputs
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- x
    for L = 1,n do
        table.insert(inputs, nn.Identity()()) -- prev_c[L]
        table.insert(inputs, nn.Identity()()) -- prev_h[L]
    end

    local x, input_size_L
    local outputs = {}
    function annotate(x, prefix, bnprefix, L)
        x = x:annotate{name = (prefix .. L)}
        if bn then
            return x:annotate{name = (bnprefix .. L)}
        else
            return x
        end
    end
    for L = 1,n do
        -- c,h from previos timesteps
        local prev_h = inputs[L*2+1]
        local prev_c = inputs[L*2]
        -- the input to this layer
        if L == 1 then
            x = OneHot(input_size)(inputs[1])
            input_size_L = input_size
        else
            x = outputs[(L-1)*2]
            if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
            input_size_L = rnn_size
        end

        -- recurrent batch normalization
        -- http://arxiv.org/abs/1603.09025
        local bn_wx, bn_wh, bn_c
        if bn then
            bn_wx = nn.BatchNormalization(4 * rnn_size, bn_eps, bn_eps, bn_affine)
            bn_wh = nn.BatchNormalization(4 * rnn_size, bn_eps, bn_eps, bn_affine)
            bn_c = nn.BatchNormalization(rnn_size, bn_eps, bn_eps, bn_affine)

            -- initialise beta=0, gamma=0.1
            bn_wx.weight:fill(0.1)
            bn_wx.bias:zero()
            bn_wh.weight:fill(0.1)
            bn_wh.bias:zero()
            bn_c.weight:fill(0.1)
            bn_c.bias:zero()
        else
            bn_wx = nn.Identity()
            bn_wh = nn.Identity()
            bn_c = nn.Identity()
        end
        -- evaluate the input sums at once for efficiency
        local i2h = annotate(nn.Linear(input_size_L, 4 * rnn_size)(x), 'i2h_', 'bn_wx_', L)
        local h2h = annotate(nn.Linear(rnn_size, 4 * rnn_size)(prev_h), 'h2h_', 'bn_wh_', L)
        local all_input_sums = nn.CAddTable()({i2h, h2h})

        local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
        local in_gate, forget_gate, out_gate, in_transform
        if false then
            -- TODO: test - does this backprop? it should avoid copying. does SplitTable avoid it too?
            local first3 = 3 * rnn_size
            local sigmoid_chunk = nn.Narrow(2, 1, first3)(all_input_sums)
            sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
            in_gate = nn.Narrow(2, 1, rnn_size)(sigmoid_chunk)
            forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(sigmoid_chunk)
            out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk)
            in_transform = nn.Tanh()(nn.Narrow(2, first3 + 1, rnn_size)(all_input_sums))
        else
            -- decode the gates
            local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
            in_gate = nn.Sigmoid()(n1)
            forget_gate = nn.Sigmoid()(n2)
            out_gate = nn.Sigmoid()(n3)
            -- decode the write inputs
            in_transform = nn.Tanh()(n4)
            -- perform the LSTM update
        end
        local next_c = nn.CAddTable()({
                nn.CMulTable()({forget_gate, prev_c}),
                nn.CMulTable()({in_gate, in_transform})
                                     })
        -- gated cells form the output
        local bnc = next_c
        if bn then
            bnc = bn_c(next_c):annotate { name = 'bn_c_' .. L }
        end
        local next_h = nn.CMulTable()({out_gate, nn.Tanh()(bnc)})
        table.insert(outputs, next_c)
        table.insert(outputs, next_h)
    end

    -- needed for backward / updateGradInput:
    local top_h = outputs[#outputs]
    if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
    local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
    local logsoft = nn.LogSoftMax()(proj)
    table.insert(outputs, logsoft)

    return nn.gModule(inputs, outputs)
end

return LSTM
