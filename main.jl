using Pkg
Pkg.activate(".")
Pkg.add("LinearAlgebra")
using LinearAlgebra
Pkg.add("CUDA")
using CUDA

function forward(layers, weights)
  for i in 1:(length(layers)-1)
    layers[i+1] = map(x -> sigmoid(x), weights[i] * layers[i])
  end
end

function loss(layers, weights, data)
  averageLoss = 0
  for i in 1:(length(data))
    layers[1] = data[i][1]
    forward(layers, weights)
    averageLoss += (0.5 / length(data)) * LinearAlgebra.norm(data[i][2] - layers[length(layers)])^2
  end
  return averageLoss
end

sigmoid(x) = 1 / (1 + exp(-x))

function (@main)(args)
  architecture = [2, 2, 2, 1]
  # weights = Vector{CuMatrix{Float64}}(undef, length(architecture) - 1)
  # weightsPrime = Vector{CuMatrix{Float64}}(undef, length(architecture) - 1)
  weights = Vector{Matrix{Float64}}(undef, length(architecture) - 1)
  weightsPrime = Vector{Matrix{Float64}}(undef, length(architecture) - 1)
  for index in 1:(length(architecture)-1)
    # weights[index] = CUDA.rand(architecture[index+1], architecture[index])
    # weightsPrime[index] = CUDA.zeros(architecture[index+1], architecture[index])
    weights[index] = rand(architecture[index+1], architecture[index])
    weightsPrime[index] = zeros(architecture[index+1], architecture[index])
  end
  # layers = Vector{CuVector{Float64}}(undef, length(architecture))
  layers = Vector{Vector{Float64}}(undef, length(architecture))
  for (index, value) in enumerate(architecture)
    # layers[index] = CUDA.zeros(value)
    layers[index] = zeros(value)
  end

  data = [
    [[0.0, 0.0], [0.0]],
    [[1.0, 0.0], [1.0]],
    [[0.0, 1.0], [1.0]],
    [[1.0, 1.0], [1.0]],
  ]

  @assert length(data[1][1]) == architecture[1] "Number of data input(s) does not match architecture"
  @assert length(data[1][2]) == architecture[length(architecture)] "Number of data output(s) does not match architecture"

  epochs = 100000
  deltaX = 0.0001
  learningRate = 1
  @time for epoch in 1:epochs
    # calculate weights prime
    for (weightLayer, weight) in enumerate(weights)
      for row in 1:size(weight)[1]
        for column in 1:size(weight)[2]
          weightsCopy = deepcopy(weights)
          weightsCopy[weightLayer][row, column] += deltaX
          weightsPrime[weightLayer][row, column] = (loss(layers, weightsCopy, data) - loss(layers, weights, data)) / deltaX
        end
      end
    end

    # desend
    for (weightLayer, weight) in enumerate(weights)
      weights[weightLayer] -= learningRate * weightsPrime[weightLayer]
    end
  end

  println("--Test Training Results--")
  println("Final Trainging Error: ", loss(layers, weights, data))

  layers[1] = [0.0, 1.0]
  forward(layers, weights)
  println("--Test Input--")
  println("Input: ", layers[1])
  println("Ouput: ", layers[length(layers)])

end
