## Segmentation to detect defects in mechanical parts

### Description:
This repository contains the code to train and evaluate a segmentation model for identifying defects in mechanical parts.

### Repository structure:
- data - Contains the training and test data.
- model_chkpts - Stores saved model checkpoints.
- notebooks - Jupyter notebooks to visualize data and augmentations.
- results - Sample results from the trained model, including synthetic test data
- brown_bracket_test - Contains scripts, checkpoints, and results for the bracket_brown dataset variant.


### How to Run:
1. **Install Dependencies** by running `pip install -r requirements.txt`
2. **Download the dataset** 'bracket_white.zip' from the [link](https://vutbr-my.sharepoint.com/personal/xjezek16_vutbr_cz/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxjezek16%5Fvutbr%5Fcz%2FDocuments%2FMPDD&ga=1)
3. **Visualise Data**
   
    Open and run the code blocks in `notebooks/visualize_data.ipynb` to unzip the dataset and visualise images, masks and   augmentations
4. **Generate Augmented Dataset**

     Run `python generate_dataset.py` to generate a dataset with augmentations applied to the original images and masks.
5. **Train the Model**

     Run `python train.py` to train the UNet++ model. Training for ~30 epochs produced good results.
6. **Checkpoints**

     After training, model checkpoints will be saved to the `model_chkpts/` directory.
8. **Evaluate the Model**

   Test the saved model using `python evaluate.py`. This will evaluate the model on the validation data from train_test_split and synthetic images generated using DALL-e.
10. **View Results**

       The output of the model - masks and overlaid image will be saved in `results/` directory.
