# Dog cat classification model training tutorial
This tutorial covers how to use **ComfyUI Data Analysis** to train an image classification model with your own image dataset.  
We'll use a **dog and cat image dataset v2** created specifically for this tutorial, which is available publicly on [my Hugging Face repo](https://huggingface.co/datasets/HowToSD/dog_and_cat).  

Once you complete the tutorial, you can experiment with your own images.

**Disclaimer:** The model architecture is a traditional convolutional neural network (CNN) and is **not state-of-the-art**. This choice keeps the model small, enabling faster training and easier explanation.

---

## Overview of Steps
1. Install ComfyUI-Data-Analysis extension
2. Downloading the dog and cat dataset from the Hugging Face repo
3. Edit the training workflow  
4. Run the training workflow  
5. Edit the evaluation workflow
6. Run the evaluation workflow
7. Use the trained model to classify the images generated in Stable Diffusion
---
## 1. Install ComfyUI-Data-Analysis extension.
If you haven't installed ComfyUI-Data-Analysis extension, please do so first.

## 2. Download the Dog & Cat Dataset  
Make sure that you have installed "git lfs".

After that, visit [my Hugging Face repo](https://huggingface.co/datasets/HowToSD/dog_and_cat) and download the dataset.

To download, you can:
```
git clone https://huggingface.co/datasets/HowToSD/dog_and_cat
```

The downloaded dataset follows this directory structure:  

```
dog_and_cat
  train
    cats
    dogs

  val
    cats
    dogs
```


Take note of the **full paths** to the `train` and `val` directories, as you'll need them in the next step.

---

## 3. Editing the Training Workflow

Drag and drop the below file into ComfyUI:  
```
examples/workflow/dog_cat_classification_ds2_train_model_v1.json
```
Update the following in relevant nodes:
* train dataset path
* val dataset path
* model save path
---

## 4. Running the Training Workflow  

Click **Queue** to start training.  

Monitor the console for errors.

If you encounter an error:  
- Ensure dataset paths are correct.  
- Reduce **batch size** if your GPU runs out of memory (OOM error).  
---

## 5. Editing the Evaluation Workflow
Now, let's verify if the model was trained correctly. In this step, we'll feed the validation data into the trained model and compute metrics to assess its performance.

Drag and drop the below file into ComfyUI:  
```
examples/workflow/dog_cat_classification_ds2_eval_model_v1.json
```

into **ComfyUI**.  

Update the **val dataset path** and **model path**.
---

## 6. Running the Evaluation Workflow  

Click **Queue**. The evaluation process runs much faster than training.  

Check the **Accuracy** node—it should display a value above **0.9**, confirming successful model training.

Congratulations! You’ve successfully trained a machine learning model from scratch without writing a single line of code.

Now, you can use your own images to train the model in the same way.

## 7. Use the Trained Model to Classify Images Generated in Stable Diffusion

Now, let's use the trained model to classify images generated in Stable Diffusion.  
Drop the workflow file: `examples/workflows/dog_cat_classification_ds2_prediction_after_generation_v1.json`.

This workflow uses the **JuggernautXL X** model. If you have it, I recommend using this model. Otherwise, use at least a high-quality photorealistic **SDXL** model.

Click **Queue**, and the workflow will generate an image, pass it to the trained classification model, and classify it as either a dog or a cat. If you generate multiple times, you may notice occasional misclassifications.

### Classification Results
From my test using this workflow, I obtained the following results:

|  | Dog | Cat |
|---|---|---|
| **Correct** | 19 | 30 |
| **Wrong** | 6 | 0 |

I observed a tendency for some dogs to be classified as cats. Possible reasons include:

- Generated images from Juggernaut differ from the training dataset.
- The dataset might be too small.
- The model is not state-of-the-art.

The training dataset was created using **FLUX.1 \[schnell\]**, and the model does not classify images it hasn't seen during training. Although 2000 images in the training dataset might seem like a lot, training from scratch can require a significantly larger dataset for the model to generalize distinguishing features between dogs and cats.  

For this workflow, I chose a traditional convolutional network to speed up training, but this also limits the model's capability. A larger model could improve performance, but it would also require a larger dataset, making it a double-edged sword.  

That said, the model should classify photorealistic images correctly most of the time, as long as they are not too different from the images in the training set. You can also experiment with training a model for different types of images.

