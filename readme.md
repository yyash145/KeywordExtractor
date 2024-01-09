Data Understanding - 

Structure of Dataset:
	We have Structured medical data with 4999 rows and 5 columns, which contains sample medical transcriptions for various medical specialties. One of the feature Transcription, using which we have to extract the Keywords, which will be our Label.

Data Preprocessing:
	First we have to do “feature selection” on our Dataset. This allows us to focus only on the relevant Information and necessary features. It helps us to reduce dimensionality, improve Model performance and make data more manageable.
	After if, we simply remove duplicate and null values.
	For cleaning data, we keep only alpha-numerical values, by using Regex. After it, I apply lowercasing to the entire dataset.
	For tokenization, we use pre-trained model’s tokenizer. That pre-trained model is of “Facebook/bart-base” and we took this as our model.
	Then, we split data into the training and validation / testing set.

Train / Fine-tune Dataset:
	First we train the model.
	Then, we iterate over the batches over the entire data.
	Clear the gradients of all optimized parameters before computing gradients.
	We perform, forward pass through Neural Network.
	We retrieve and accumulate the loss from the model’s output.
	We retrieve the logits (raw outputs) from the model.
	We applies softmax activation to the logits along dimension 2.
	We computes the indices of the maximum values along dimension 2, effectively obtaining predicted labels.
	After these steps, we compute and accumulates the accuracy by comparing predicted labels with true labels.
	Then we find Backward pass to compute gradients of the loss with respect to model parameters and update the model's parameters using the computed gradients.

Evaluation
	First, we use model.eval(), which sets the model in evaluation mode. This is important because some layers may behave differently during training and testing.
	Then, we iterate over the batches over the entire data.
	
	
out = model(transcript, labels=keyWord): Performs a forward pass through the neural network.
loss = out.loss: Retrieves the loss from the model's output.
model_loss += loss.item(): Accumulates the loss.
logits = out.logits: Retrieves the logits (raw outputs) from the model.
preds = torch.softmax(logits, dim=2): Applies softmax activation to the logits along dimension 2.
preds = torch.argmax(preds, dim=2): Computes the indices of the maximum values along dimension 2, effectively obtaining predicted labels.
acc = torch.sum(keyWord == preds).item() / (keyWord.shape[0] * keyWord.shape[1]): Computes accuracy by comparing predicted labels with true labels.
model_acc += acc: Accumulates the accuracy.Note: In the testing phase, there is no backpropagation (loss.backward() and optimizer.step()) because the model is not being updated. This phase is solely for evaluating the model's performance on the given test data.
i += 1: Updates the batch counter.

EDA

Working Code

Approach

Explanation

Challenges Faced

Solution

Findings

Results

Documentation

Algorithm Used

Reasoning



