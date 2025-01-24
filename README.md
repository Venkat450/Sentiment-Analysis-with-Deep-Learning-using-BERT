# Sentiment Analysis with Deep Learning using BERT

## Overview
This project implements sentiment analysis using the pre-trained BERT (Bidirectional Encoder Representations from Transformers) model. The model is fine-tuned on the SMILE Twitter Emotion dataset to classify tweets into various sentiment categories. The project demonstrates data preprocessing, model training, evaluation, and performance metrics for sentiment classification.

## Dataset
- **Dataset**: [SMILE Twitter Emotion Dataset](https://doi.org/10.6084/m9.figshare.3187909.v2)
- **Description**: A collection of tweets annotated with emotion categories. After preprocessing, tweets are categorized into multiple classes for sentiment analysis.
- **Preprocessing Steps**:
  - Removed rows with multiple or invalid annotations.
  - Encoded sentiment labels into numerical values for classification.

## Requirements
- **Python Libraries**:
  - `torch`
  - `transformers`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `tqdm`
- Install dependencies using:
  ```bash
  pip install torch transformers scikit-learn pandas numpy tqdm
## Project Workflow

### Data Preparation
- Loaded the SMILE Twitter dataset and cleaned invalid entries.
- Split the data into training and validation sets using stratified sampling.

### Data Encoding and Tokenization
- Utilized the `BertTokenizer` from the Hugging Face Transformers library.
- Encoded tweets into token IDs with attention masks for BERT input.

### Model Architecture
- Used `BertForSequenceClassification` with:
  - Pre-trained `bert-base-uncased` weights.
  - Output layer customized for the number of sentiment categories.

### Training
- **Optimizer**: `AdamW` with a learning rate of 1e-5.
- **Scheduler**: Linear learning rate decay using `get_linear_schedule_with_warmup`.
- **Batch Size**: 4 for training, 32 for validation.
- **Loss Function**: Cross-entropy loss.
- **Epochs**: 10 with early stopping and gradient clipping.

### Evaluation
- **Metrics**:
  - **F1 Score**: Weighted F1 score for class-level performance.
  - **Accuracy per Class**: Detailed accuracy for each sentiment class.
- Validation performed after each epoch to monitor performance.

## Results

### Key Metrics
- **Weighted F1 Score**: Achieved consistent improvement across epochs.
- **Class-Level Accuracy**: Evaluated using the `accuracy_per_class` function.

### Performance Trends
- The training process highlighted the importance of proper hyperparameter tuning to achieve robust generalization.

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd <repository-folder>
2. Ensure the dataset is in the `Data/` directory as `smile-annotations-final.csv`.
3. Execute the training script or Jupyter Notebook to fine-tune BERT and evaluate the model.

## Discussion

### Strengths
- Leveraged BERT's pre-trained language representations for high-quality sentiment classification.
- Stratified split ensured balanced representation of classes.

### Challenges
- Computational resource requirements for training on GPU.
- Achieving consistent validation performance due to class imbalance.

### Future Improvements
- Experiment with larger batch sizes and longer sequence lengths.
- Fine-tune on additional emotion datasets for broader generalization.
- Explore advanced techniques such as ensemble learning or knowledge distillation.
