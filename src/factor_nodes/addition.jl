export Addition

import Base: +
export +

"""
Description:

    An addition constraint factor node

    f(x,y,z) = δ(x + y - z)

Interfaces:

    1. in1
    2. in2
    3. out

Construction:

    Addition(out, in1, in2, id=:some_id)
"""
type Addition <: DeltaFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function Addition(out::Variable, in1::Variable, in2::Variable; id=generateId(Addition))
        self = new(id, Array(Interface, 3), Dict{Int,Interface}())
        addNode!(currentGraph(), self)
        self.i[:in1] = self.interfaces[1] = associate!(Interface(self), in1)
        self.i[:in2] = self.interfaces[2] = associate!(Interface(self), in2)
        self.i[:out] = self.interfaces[3] = associate!(Interface(self), out)

        return self
    end
end

slug(::Type{Addition}) = "+"

function +(in1::Variable, in2::Variable)
    out = Variable()
    Addition(out, in1, in2)
    return out
end