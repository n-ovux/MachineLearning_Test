using Pkg
Pkg.activate(".")
using LinearAlgebra

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
sigmoidPrime(x) = sigmoid(x) * (1 - sigmoid(x))
sigmoidV(v) = map(x -> sigmoid(x), v)
sigmoidPrimeV(v) = map(x -> sigmoidPrime(x), v)

function (@main)(args)
  architecture = [2, 2, 1]
  weights = Vector{Matrix{Float64}}(undef, length(architecture) - 1)
  weightsPrime = Vector{Matrix{Float64}}(undef, length(architecture) - 1)
  for index in 1:(length(architecture)-1)
    weights[index] = rand(architecture[index+1], architecture[index])
    weightsPrime[index] = zeros(architecture[index+1], architecture[index])
  end
  layers = Vector{Vector{Float64}}(undef, length(architecture))
  for (index, value) in enumerate(architecture)
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
  learningRate = 1
  @time for epoch in 1:epochs
    # calculate weights prime
    currentData = data[(epoch%length(data)+1)][2]
    layers[1] = data[(epoch%length(data)+1)][1]
    forward(layers, weights)
    weightsPrime[2] = (layers[3] - currentData) * sigmoidV(layers[2])'
    weightsPrime[1] = diagm(sigmoidPrimeV(layers[2])) * weights[2]' * (layers[3] - currentData) * sigmoidV(layers[1])'

    # this is for 4 layers
    # weightsPrime[3] = (layers[4] - currentData) * sigmoidV(layers[3])'
    # weightsPrime[2] = diagm(sigmoidPrimeV(layers[3])) * weights[3]' * (layers[4] - currentData) * sigmoidV(layers[2])'
    # weightsPrime[1] = diagm(sigmoidPrimeV(layers[2])) * weights[2]' * diagm(sigmoidPrimeV(layers[3])) * weights[3]' * (layers[4] - currentData) * sigmoidV(layers[1])'

    # desend
    for (weightLayer, weight) in enumerate(weights)
      weights[weightLayer] -= learningRate * weightsPrime[weightLayer]
    end
  end

  println("--Test Training Results--")
  println("Final Trainging Error: ", loss(layers, weights, data))

  println("--Test Input--")
  for input in data
    layers[1] = input[1]
    forward(layers, weights)
    println("Input: ", layers[1])
    println("Ouput: ", layers[length(layers)])
  end
end
