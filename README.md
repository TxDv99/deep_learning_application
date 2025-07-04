----Deeep learning application README Tommaso Ducci---------

## Credits

The ideas behind the `MyResNet` class were discussed together with colleagues Francesco Correnti and Vincenzo Civale.  
The structure and design choices of the `Trainer` class, especially the initial checks, are inspired by Karpathy’s blog post and are based in large part on this repository: [karpathy/minGPT](https://github.com/karpathy/minGPT/blob/master/mingpt/trainer.py).  
The parameters used for the Hugging Face fine-tuning task were taken from this official tutorial video: [Hugging Face Transformers Course](https://www.youtube.com/watch?v=u--UVvH-LIQ&t=1019s).  
Finally, I used ChatGPT to clarify ideas and speed up development, especially since I was new to PyTorch, but there is no code here that I did not fully understand and rewrite myself.


# Deep Learning Utils

This repository provides two main classes: **`MyResNet`** and **`Trainer`**, located inside the `deep_learning_utils` folder.

To make them available, run:

pip install -e .


from inside the folder.

---

## MyResNet

**MyResNet** offers a flexible API to build ResNet-like architectures.

### Main features

- The `__init__` method takes:
  - A tuple list describing the layers (e.g., `("Conv2d", ...)`, `"Linear", ...`).
  - An optional `skip_dict`: a dictionary with `{from_idx: to_idx}` to define skip connections.
  - The input shape, which is needed to perform reprojections when skip connections change dimensions.
- Handles internal adapters when transitioning from a convolutional block to a linear layer.
- Provides helper methods:
  - `test()` — runs evaluation on a dataset.
  - `get_submodel()` — extracts a submodel up to a certain layer.
  - `to()` — inherited from `nn.Module` but adds a `device` attribute to the class for convenience.

---

## Trainer

The **Trainer** class provides a simple training pipeline, inspired by Karpathy-style minimal training loops.

### `__init__` arguments

- `config`: a dictionary with `device`, `num_workers`, and `seed`.
- `model`: the model to train.
- `lr`: learning rate.
- Optional: `optimizer` and `loss_fn` (defaults to Adam and CrossEntropyLoss).

### `train()` arguments

- `data_split`: e.g., `[0.4, 0.6]` for train/validation split.
- `batch_size`, `num_epochs`.
- `early_stopping`: number of validation checks without improvement to stop training.
- `val_check`: frequency of validation checks.
- `checks`: if `True`, the trainer runs initial checks to:
  - Ensure the logits distribution is balanced.
  - Verify that the model can overfit on a small dataset (prints warnings if not).
- `use_wandb` and `project_name`: for experiment logging.
- `augmentation`: tuple with the fraction of the training set to augment and a `torchvision.transforms` object.
- `fgsm`: enables on-the-fly FGSM adversarial training:
  - A single-element tuple means untargeted attack (`(budget,)`).
  - A two-element tuple means targeted attack (`(budget, target_label)`).

---

## FGSM Function

The `FGSM` function generates adversarial examples.

**Arguments:**

- `x`: a single input tensor to perturb.
- `model`: the model to attack.
- `budget`: the perturbation budget.
- `y_true`: true label for untargeted attack.
- `y_target`: target label for targeted attack.
- `loss_fun`: optional, defaults to `CrossEntropyLoss`.



## OOD Pipeline

The `OOD_pipeline` function provides a simple framework to evaluate how well a model can detect Out-Of-Distribution (OOD) samples versus In-Distribution (ID) samples.

### Main features

- Computes model outputs for both ID and OOD datasets.
- Stores:
  - True labels (`y_gt`), predictions (`y_pred`), and confidence scores for ID.
  - Logits for ID and OOD data.
- Supports **temperature scaling**:
  - Finds the best temperature to calibrate the model.
  - Optionally plots calibration curves for different temperatures.
- Plots:
  - Confusion matrix for ID predictions.
  - Histograms of the score distributions for ID and OOD samples (before and after temperature scaling if enabled).
  - ROC curve for ID scores.
  - Precision-Recall curve for OOD detection, using a binary label (1 = ID, 0 = OOD).
- Uses `sklearn.metrics` for all curve computations and visualizations.
- Returns the best temperature and ECE score when temperature scaling is used.

### Key arguments

- `model`: the model to evaluate.
- `device`: e.g., `'cuda'`.
- `ID_ds`: dataset of in-distribution samples.
- `OOD_ds`: dataset of out-of-distribution samples.
- `batch_size`, `num_workers`, `seed`: dataloader and reproducibility settings.
- `T_scaling`: list of temperatures to test for calibration.
- `score_function`: e.g., `'max_softmax'` for confidence.
- `plot`: whether to plot results.
- `plot_performances`: whether to print accuracy and plot confusion matrix.



