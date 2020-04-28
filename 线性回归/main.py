import  torch
import  torch.nn  as  nn
import  numpy as  np
import  matplotlib.pyplot as  plt

input_size = 1
output_size = 1
num_epochs = 10000
learning_rate = 0.01

#dataset
x_train = np.array([[3.3],[4.4],[5.5],[6.1],[6.9],[4.2],
                  [5.6],[6.1],[7.9],[8.25],[2.16],[3.6],
                  [5.25],[4.38],[6.39]], dtype=np.float32)

y_train = np.array([[1.7],[2.76],[2.09],[3.19],[1.59],[1.573],
                  [3.366],[2.596],[2.53],[1.226],[2.635],[3.95],
                  [1.95],[2.90],[1.52]], dtype=np.float32)

model = nn.Linear(input_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

# train the model
for  epoch in range(num_epochs):
  inputs = torch.from_numpy(x_train)
  targets = torch.from_numpy(y_train)

  # Forward pass
  outputs = model(inputs)
  loss = criterion(outputs, inputs)

  # Backward and optimize
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if (epoch + 1) % 100 == 0:
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Plot the graph
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Oraginal data')
plt.plot(x_train, predicted, label="Fitted line")
plt.legend()
plt.show()

# Save the model checkpoint

torch.save(model.state_dict(), 'model.ckpt')