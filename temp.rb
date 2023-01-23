require 'ruby-fann'
require 'rest-client'


def rationalize_data(data)
  max_val = data.max.max
  data = data.collect { |_data| _data.collect { |value| value/max_val }}
  data
end


data = JSON.parse(RestClient.get("https://x.wazirx.com/api/v2/k?market=dogeinr&period=5&limit=1900"))

data.each { |_data| 
  _data.shift(1)
   _data.pop
}

data.reverse

train_data = data[0..(data.length * 0.8)]
test_data = data[(data.length * 0.8)..-1]

inputs = train_data.map { |_data| _data[0..2] }
outputs = train_data.map { |_data| _data[3] }

params = {objective: "reg:squarederror"}
dtrain = XGBoost::DMatrix.new(inputs, label: outputs)
booster = XGBoost.train(params, dtrain)

inputs = test_data.map { |_data| _data[0..2] }
outputs = test_data.map { |_data| _data[3] }


train = RubyFann::TrainData.new(:inputs=> inputs, :desired_outputs=> outputs)
fann = RubyFann::Standard.new(:num_inputs=>5, :hidden_neurons=>[12, 16, 20, 24, 28, 32, 36, 40], :num_outputs=>1)
fann.train_on_data(train, 10000, 20, 0.1)

# Gather historical data and prepare it for training
data = []
CSV.foreach("crypto_data.csv", headers: true) do |row|
  data << [row['price'], row['market_factor1'], row['market_factor2'], row['market_factor3'], row['market_factor4'], row['market_factor5'],row['buy_or_sell']]
end

# Shuffle the data for better training
data.shuffle!

# Split the data into training and testing sets
train_data = data[0..(data.length * 0.8)]
test_data = data[(data.length * 0.8)..-1]

# Create a new neural network with 5 inputs and 2 output
network = Fann::Standard.new(num_inputs: 5, num_outputs: 2)

# Set the training parameters
network.activation_function_hidden = :sigmoid_symmetric
network.activation_function_output = :sigmoid
network.training_algorithm = :rprop

# Create training data
training_data = Fann::TrainData.new(inputs: train_data.map { |d| d[0..4] }, desired_outputs: train_data.map { |d| d[6] == 'buy' ? [1, 0] : [0, 1] })

# Train the network on the historical data
network.train_on_data(training_data, max_epochs: 10000, epochs_between_reports: 1000)

# Test the network on new data
test_data.each do |data|
  prediction = network.run(data[0..4])
  puts "Predicted buy or sell: #{prediction.index(prediction.max) == 0 ? 'buy' : 'sell'}, actual: #{data[6]}"
end

# Save the trained model
network.save("crypto_model.net")
