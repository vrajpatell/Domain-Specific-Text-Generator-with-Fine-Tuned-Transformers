
# Domain-Specific Text Generator with Fine-Tuned Transformers

This project fine-tunes a pre-trained GPT-2 model on a domain-specific text dataset to generate text in a niche area. It demonstrates transfer learning, data preprocessing, model fine-tuning, and building an interactive demo using Gradio.

---

## Project Structure

```plaintext
domain_text_generator/
├── README.md
├── requirements.txt
├── data/
│   └── domain_dataset.txt
├── scripts/
│   └── fine_tune.py
└── app/
    └── app.py
```

- **`data/domain_dataset.txt`**: Contains the domain-specific text data. Each line represents a training example.
- **`scripts/fine_tune.py`**: Script to fine-tune GPT-2 using Hugging Face Transformers on the provided dataset.
- **`app/app.py`**: Gradio-based interactive demo to generate text using the fine-tuned model.
- **`requirements.txt`**: List of required Python libraries.

---

## Setup Instructions

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/domain_text_generator.git
    cd domain_text_generator
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare your dataset:**

   Replace or update `data/domain_dataset.txt` with your domain-specific text data.  
   Each line should be a separate text sample.

---

## Fine-Tuning the Model

Run the fine-tuning script:

```bash
python scripts/fine_tune.py --data_file data/domain_dataset.txt --output_dir fine_tuned_model --epochs 3 --block_size 128
```

### Arguments

- **`--data_file`**: Path to your text dataset.
- **`--output_dir`**: Directory where the fine-tuned model will be saved.
- **`--epochs`**: Number of training epochs.
- **`--block_size`**: Block size used for text chunking.

---

## Model Evaluation

After training, the script will evaluate the model and print the perplexity metric.

---

## Running the Interactive Demo

Launch the Gradio app:

```bash
python app/app.py
```

This will start a local server and open a web interface where you can input a text prompt and generate text.

---

## Project Dependencies

- Python 3.7+
- Transformers
- Datasets
- Torch
- Gradio

---

## Fine-Tuning Details

The fine-tuning process uses the Hugging Face Trainer API with the following steps:

1. **Data Loading**: The dataset is loaded from a text file and split into training and validation sets.  
2. **Tokenization**: The GPT-2 tokenizer tokenizes the text data.  
3. **Text Chunking**: Tokenized texts are grouped into blocks of a specified size.  
4. **Model Training**: The pre-trained GPT-2 model is fine-tuned on the processed dataset.  
5. **Evaluation**: Model performance is evaluated using perplexity.

---

## License

This project is open source under the [MIT License](https://opensource.org/licenses/MIT).

---

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Gradio](https://github.com/gradio-app/gradio)
```
