using LinearAlgebra: norm2

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
    averageLoss += (0.5 / length(data)) * norm2(data[i][2] - layers[length(layers)])^2
  end
  return averageLoss
end

sigmoid(x) = 1 / (1 + exp(-x))

function (@main)(args)
  architecture = [1, 2, 2, 1]
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
    [[0.0], [0.0]],
    [[0.1], [0.2]],
    [[0.2], [0.4]],
    [[0.3], [0.6]],
    [[0.4], [0.8]],
    [[0.5], [1.0]],
  ]

  epochs = 100000
  deltaX = 0.0001
  learningRate = 1
  for epoch in 1:epochs
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
  println("Final Traingin Error: ", loss(layers, weights, data))

  layers[1] = [0.3]
  forward(layers, weights)
  println("--Test Input--")
  println("Input: ", layers[1])
  println("Ouput: ", layers[length(layers)])

end
