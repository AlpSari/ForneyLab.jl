@variationalRule(:node_type     => Gamma,
                 :outbound_type => Message{AbstractGamma},
                 :inbound_types => (Void, ProbabilityDistribution, ProbabilityDistribution),
                 :name          => VBGammaOut)
