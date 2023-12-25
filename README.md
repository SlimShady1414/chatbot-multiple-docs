
# ChatBot for Multiple Files

This project utilizes Streamlit, Hugging Face Transformers (LaMini-T5-738M, LLAMA), LangChain (CSV Loader, PDFMinerLoader, Chroma), Sentence Transformer (all-MiniLM-L6-v2), FAISS, MiniLM, and Tempfile to create interactive chatbots for processing and analyzing PDF and CSV data.



## Installation(ChatBot_PDF)

1. Clone the repository and change to project directory.

2. Install the dependencies.

```bash
pip install -r requirements.txt
```

3. Create two folders named "db" and "docs" in the project directory.

4. Upload the PDFs you want to interact with into the "docs" folder.

5. Clone the [LaMini-T5-738M](https://huggingface.co/MBZUAI/LaMini-T5-738M) model into the project directory.
```bash
git lfs install
git clone https://huggingface.co/MBZUAI/LaMini-T5-738M
```

6. Run the ingest.py file.
```bash
python ingest.py
```

7. Run the chatbot_app.py file.
```bash
streamlit run chatbot_app.py
```

## Installation(ChatBot_CSV)
1. Change directory to chatcsv.
```bash
cd chatcsv
```
2. Install the dependencies
```bash
pip install -r requirements.txt
```
3. Download the [Llama 2 7B Chat - GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML) model and store it in the chatcsv folder.


## Screenshots

[![File in the last row](https://i.postimg.cc/FsF55BmC/Screenshot-2023-12-25-135127.png)](https://postimg.cc/z3QPTpyT)




4. Run the chatcsv.py file.
```bash
streamlit run chatcsv.py
```

## Future Enhancements
Make a single chatbot that can integrate with multiple different types of files(eg: pdf, csv, images etc).
