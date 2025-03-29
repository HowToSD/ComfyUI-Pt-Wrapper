# Training Transformer for Text Classification

You can use **ComfyUI-Pt-Wrapper** to train a Transformer for text classification, such as sentiment analysis on a movie review dataset.

You can find the workflow at `examples/workflows/embedding_transformer_classification.json`.

The flow consists of the following groups of nodes:
* Loading data
* Setting up the model, loss, optimizer, and scheduler
* Training

## Loading Data

This phase consists of two steps:
* Specify the SentencePiece encoder for converting text to token IDs
* Specify a Hugging Face text dataset to load

## Specify SentencePiece encoder for converting text to token IDs
Use Sp Load Model and Sp Encode to load the SentencePiece model and specify the encode function for text to token ID convertion.

## Specify a Hugging Face text dataset to load
Specify **Ptv Hf Dataset With Token Encode** to load the text data, and encode each word into token IDs using the Sp Encode function above.

Then use **Pt Data Loader** to batch the data and feed it into the training node.

## Setting Up Optimizer and Scheduler

This workflow uses **Pto AdamW** as the optimizer and **Pto Lr Scheduler Cosine Annealing** as the scheduler. You can also experiment with other optimizers and schedulers included in this extension.

## Loss

Use **Ptn BCE with Logits Loss**, which includes a built-in sigmoid activation, allowing you to pass raw logits directly from the model.

## Model

Use an Transformer model with a linear head (**Ptn Embedding Transformer Linear**). This model contains the embedding layer at the bottom.  Note that this embedding layer is not pretrained, so we are training this layer from scratch in this workflow.

## Training

Specify **Pt Train Transformer Classification Model** and configure training parameters.  
Set the vocabulary size to **32000** to match the SentencePiece tokenization model.

## Expected Accuracy

Using this workflow, you should achieve around 83% validation accuracy on the Hugging Face version of the IMDB dataset. While this may seem low to some readers, it's important to note that the model is trained entirely from scratch, with no pretrained layers (aside from tokenization to IDs). Given that, this accuracy is within the expected range.
