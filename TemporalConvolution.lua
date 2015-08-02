local TemporalConvolution, parent =
   torch.class('cudnn.TemporalConvolution', 'cudnn.SpatialConvolution')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

function TemporalConvolution:__init(nInputPlane, nOutputPlane,
                            kH, dH, padH, groups)
   local kW = 1;
   local dW = 1;
   local padW = 0;
   parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, groups)
end

function TemporalConvolution:argcheck(input, name)
    if input:nDimension() == 2 then
        input = input:view(input:size(1), input:size(2), 1)
        input = input:transpose(1, 2):clone()
    elseif input:nDimension() == 3 then
        input = input:view(input:size(1), input:size(2), input:size(3), 1)
        input = input:transpose(2, 3):clone()
    else
        error(name .. ' of 2D or 3D (mini-batch) expected')
    end
    return input
end

function TemporalConvolution:adjustOutput(input)
    if input:nDimension() == 3 then
        input = input:view(input:size(1), input:size(2))
        input = input:transpose(1, 2)
    elseif input:nDimension() == 4 then
        input = input:view(input:size(1), input:size(2), input:size(3))
        input = input:transpose(2, 3)
    else
        print(#input)
        error('Unexpected output dimensions')
    end
    return input
end

function TemporalConvolution:updateOutput(input)
    input = self:argcheck(input, 'input')
    parent.updateOutput(self, input)
    self.output = self:adjustOutput(self.output)
    return self.output
end

function TemporalConvolution:updateGradInput(input, gradOutput)
    input = self:argcheck(input, 'input')
    gradOutput = self:argcheck(gradOutput, 'gradOutput')
    parent.updateGradInput(self, input, gradOutput)
    self.gradInput = self:adjustOutput(self.gradInput)
    return self.gradInput
end

function TemporalConvolution:accGradParameters(input, gradOutput, scale)
    input = self:argcheck(input, 'input')
    gradOutput = self:argcheck(gradOutput, 'gradOutput')
    parent.accGradParameters(self, input, gradOutput, scale)
end
