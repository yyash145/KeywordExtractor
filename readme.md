## Structure of Dataset:
- We have Structured medical data with 4999 rows and 5 columns, which contains sample medical transcriptions for various medical specialties. One of the feature Transcription, using which we have to extract the Keywords, which will be our Label.

## Data Preprocessing:
-  First we have to do “feature selection” on our Dataset. This allows us to focus only on the relevant Information and necessary features. It helps us to reduce dimensionality, improve Model performance and make data more manageable.
- After if, we simply remove duplicate and null values.
- For cleaning data, we keep only alpha-numerical values, by using Regex. After it, I apply lowercasing to the entire dataset.
- For tokenization, we use pre-trained model’s tokenizer. That pre-trained model is of “Facebook/bart-base” and we took this as our model.
- Then, we split data into the training and validation / testing set.

## Train / Fine-tune Dataset:
- First we train the model.
- Then, we iterate over the batches over the entire data.
- In each iteration, we clear the gradients of all optimized parameters before computing gradients.
- And then, we perform forward pass through Neural Network.
- Then, we retrieved and accumulated the loss from the model’s output.
- After it, we retrieved the logits (raw outputs) from the model.
- Then, we applied softmax activation to the logits along dimension 2.
- Then, we computes the indices of the maximum values along dimension 2, effectively obtaining predicted labels.
- After these steps, we compute and accumulated the accuracy by comparing predicted labels with true labels.
- Then we find Backward pass to compute gradients of the loss with respect to model parameters and update the model's parameters using the computed gradients.

## Evaluation
- First, we use model.eval(), which sets the model in evaluation mode. This is important because some layers may behave differently during training and testing.
- Then, we iterate over the batches over the entire data.
- After it, we perform forward pass through the neural network.
- Then we retrieved and accumulate the loss from the model's output.
- After it, we retrieved the logits (raw outputs) from the model.
- Then, we apply SoftMax activation function to the logits along dimension 2.
- Then, we computes the indices of the maximum values along dimension 2, effectively obtaining predicted labels.
- After these steps, we compute and accumulated the accuracy by comparing predicted labels with true labels.
- Since, the model is not being updated, "optimizer.step()" is solely for evaluating the model's performance on the given test data.

## Exploratory Data Analysis
- Plotted the graph between loss and Accuracy with Epochs

## Working Code
- In KE.ipynb

## Approach and Explanation
- Explained above

## Challenges Faced
- Whether to use Tokenizer or Lemmatization.
- Which tokenizer to use.
- For tokenizer, whether to remove stopwords or not.
- Which pre-trained model to choose.
- Training Model takes time, and if we got something wrong during training, we had to re-train it again.
- Lack of enough GPU.

## Solution
- Use Tokenizer, because we don't need root words, as we can have keywords like, "allergy", "allergic", "allergies", etc
- Used Tokenizer instead of Lemmatization because it is faster and more efficient.
- Use "facebook/bart-base" pre-trained model, as it provides both padding and truncation.

## Findings
- Model is trained on the keywords.

## Results
- We got more than required result, when we insert any transactions into the model.
- Our Accuracy is around 90%.

