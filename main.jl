using Pkg
Pkg.activate(".")
using LinearAlgebra
using Plots

function forward(layers, weights, biases)
  for i in 1:(length(layers)-1)
    layers[i+1] = activate(weights[i] * layers[i] + biases[i])
  end
end

function loss(layers, weights, biases, data)
  averageLoss = 0
  for i in 1:(length(data))
    layers[1] = data[i][1]
    forward(layers, weights, biases)
    averageLoss += (0.5 / length(data)) * LinearAlgebra.norm(data[i][2] - layers[length(layers)])^2
  end
  return averageLoss
end

# activation functions
sigmoid(x) = 1 / (1 + exp(-x))
sigmoidPrime(x) = sigmoid(x) * (1 - sigmoid(x))
tanhPrime(x) = 1 - tanh(x)^2
relu(x) = max(x, 0)
reluPrime(x) = 0.5 * (sign(x) + 1)
gelu(x) = x * sigmoid(x)
geluPrime(x) = sigmoid(x) + x * sigmoidPrime(x)
activate(v) = map(x -> tanh(x), v)
activatePrime(v) = map(x -> tanhPrime(x), v)

function (@main)(args)
  architecture = [1, 2, 3, 1]
  weights = Vector{Matrix{Float64}}(undef, length(architecture) - 1)
  weightsPrime = Vector{Matrix{Float64}}(undef, length(architecture) - 1)
  biases = Vector{Vector{Float64}}(undef, length(architecture) - 1)
  biasesPrime = Vector{Vector{Float64}}(undef, length(architecture) - 1)
  randRange = 15
  for index in 1:(length(architecture)-1)
    weights[index] = rand(-randRange:randRange, architecture[index+1], architecture[index])
    weightsPrime[index] = zeros(architecture[index+1], architecture[index])
    biases[index] = rand(-randRange:randRange, architecture[index+1])
    biasesPrime[index] = zeros(architecture[index+1])
  end
  layers = Vector{Vector{Float64}}(undef, length(architecture))
  for (index, value) in enumerate(architecture)
    layers[index] = zeros(value)
  end

  amount = 30
  data = Vector{Vector{Vector{Float64}}}(undef, amount)
  for index in 1:length(data)
    value = (index / amount)
    data[index] = Vector{Vector{Float64}}(undef, 2)
    data[index][1] = Vector{Float64}([value])
    data[index][2] = Vector{Float64}([sin(value * 4)])
  end
  @assert length(data[1][1]) == architecture[1] "Number of data input(s) does not match architecture"
  @assert length(data[1][2]) == architecture[length(architecture)] "Number of data output(s) does not match architecture"

  learningRate = 0.01
  epochs = 100000
  x = range(1, epochs)
  y = Vector{Float64}(undef, epochs)
  @time for epoch in 1:epochs
    for sample in data
      # set up data
      layers[1] = sample[1]
      forward(layers, weights, biases)

      # calculate weights prime
      biasesPrime[length(biases)] = diagm(activatePrime(layers[length(layers)])) * (layers[length(layers)] - sample[2])
      weightsPrime[length(weights)] = biasesPrime[length(biases)] * layers[length(weights)]'
      for index in 1:(length(weights)-1)
        index = length(weights) - index
        biasesPrime[index] = diagm(activatePrime(layers[index+1])) * weights[index+1]' * biasesPrime[index+1]
        weightsPrime[index] = biasesPrime[index] * layers[index]'
      end

      # desend
      for index in 1:length(weights)
        weights[index] -= learningRate * weightsPrime[index]
        biases[index] -= learningRate * biasesPrime[index]
      end
      # learningRate = max(learningRate - (1 / epochs), 0)
    end
    y[epoch] = loss(layers, weights, biases, data)
  end
  plot(x, y, lc=:red, label="Error", title="Error", xlabel="Epoch", ylabel="Error")
  savefig("error.png")
  x = Vector{Float64}(undef, amount)
  y = Vector{Float64}(undef, amount)
  for (index, sample) in enumerate(data)
    x[index] = sample[1][1]
    y[index] = sample[2][1]
  end
  scatter(x, y, label="Expected Output", title="Output", xlabel="Input", ylabel="Output")
  x = Vector{Float64}(undef, amount)
  y = Vector{Float64}(undef, amount)
  for (index, sample) in enumerate(data)
    layers[1] = sample[1]
    forward(layers, weights, biases)
    x[index] = layers[1][1]
    y[index] = layers[length(layers)][1]
  end
  plot!(x, y, label="NN Output")
  savefig("output.png")


  println("--Test Training Results--")
  println("Final Trainging Error: ", loss(layers, weights, biases, data))
  println("Epochs: ", epochs)

  println("--Neural Network--")
  for index in 1:length(layers)-1
    display(weights[index])
    display(biases[index])
  end
  # println("--Test Input--")
  # for input in data
  #   layers[1] = input[1]
  #   forward(layers, weights, biases)
  #   println("Input: ", layers[1])
  #   println("Ouput: ", layers[length(layers)])
  # end
end
