# Training RNN for Text Classification

You can use **ComfyUI-Pt-Wrapper** to train an RNN for text classification, such as sentiment analysis on a movie review dataset.

You can find the workflow at `examples/workflows/rnn_classification.json`.
You can also use GRU workflow at `examples/workflows/gru_classification.json` or LSTM workflow at `examples/workflows/lstm_classification.json`.

![Workflow](images/rnn_classification.png)

The flow consists of the following groups of nodes:
* Loading data
* Setting up the model, loss, optimizer, and scheduler
* Training

## Loading Data

Specify a Hugging Face text dataset using **Ptf Hf Glove Dataset** to load the text data, and encode each word using GloVe embeddings.  
Then use **Pt Data Loader** to batch the data and feed it into the training node.

## Setting Up Optimizer and Scheduler

This workflow uses **Pto Adam** as the optimizer and **Pto Lr Scheduler Reduce On Plateau** as the scheduler, but you can experiment with other optimizers and schedulers included in this extension.

## Loss

Use **Ptn BCE with Logits Loss**, which includes a built-in sigmoid activation, allowing you to pass raw logits directly from the model.

## Model

Use an RNN model with a linear head (**Ptn RNN Linear**).  
Set the input dimension to match the GloVe embedding size (e.g., `100` in this case).

## Training

Use **Pt Train RNN Model** to configure training parameters.  
One important parameter is `use_valid_token_mean`:
* When `True`, it averages outputs of all non-zero tokens.
* When `False`, it uses only the last token's output, which may be weak if there's heavy padding.

If your loss does not decrease, check this flag first.  
You can also enable `classification_metrics` to print validation accuracy to the console.

## Expected Accuracy

Using this workflow, you should get around **83% validation accuracy** on the Hugging Face version of the IMDB dataset, which is a solid result for an RNN.
Both GRU & LSTM workflows achieve approximately **87% validation accuracy**.
