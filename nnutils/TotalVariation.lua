local TVLoss, parent = torch.class('nn.TotalVariation','nn.Module')

function TVLoss:__init(strength)
    parent.__init(self)
    self.strength=strength
    self.x_diff=torch.Tensor()
    self.y_diff=torch.Tensor()
    self.z_diff=torch.Tensor()
end

function TVLoss:updateOutput(input)
    self.output=input
    return self.output
end

-- TV loss backward pass inspired by jcjohnson/fast-neural-style

function TVLoss:updateGradInput(input,gradOutput)
    self.gradInput:resizeAs(input):zero()
    local N,C=input:size(1), input:size(2)
    local H,W,D=input:size(3), input:size(4), input:size(5)
    self.x_diff:resize(N, C, H - 1, W - 1, D-1)
    self.y_diff:resize(N, C, H - 1, W - 1, D-1)
    self.z_diff:resize(N, C, H - 1, W - 1, D-1)

    self.x_diff:copy(input[{{}, {}, {1, -2}, {1, -2} , {1 ,-2 }  }])
    self.x_diff:add(-1, input[{{}, {}, {1, -2}, {2, -1} , {1 , -2}}])
    
    self.y_diff:copy(input[{{}, {}, {1, -2}, {1, -2} , {1 ,-2 }  }])
    self.y_diff:add(-1, input[{{}, {}, {2, -1}, {1, -2} , {1 , -2}}])

    self.z_diff:copy(input[{{}, {}, {1, -2}, {1, -2} , {1 ,-2 }  }])
    self.z_diff:add(-1, input[{{}, {}, {1, -2}, {1, -2} , {2 , -1}}])

	self.gradInput[{{}, {}, {1, -2}, {1, -2} , {1,-2} }]:add(self.x_diff):add(self.y_diff):add(self.z_diff)
	self.gradInput[{{}, {}, {1, -2}, {2, -1} , {1,-2} }]:add(-1, self.x_diff)
	self.gradInput[{{}, {}, {2, -1}, {1, -2} , {1,-2} }]:add(-1, self.y_diff)
	self.gradInput[{{}, {}, {1, -2}, {1, -2} , {2,-1} }]:add(-1, self.z_diff)
	self.gradInput:mul(self.strength)
	self.gradInput:add(gradOutput)
	return self.gradInput
end

