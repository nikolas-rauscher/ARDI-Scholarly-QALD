# Verbalizer
The following files were taken from another project and adapted from: 

[https://github.com/helemanc/gryffindor/tree/main/src/prompt-generator](https://github.com/helemanc/gryffindor/tree/main/src/prompt-generator)

- `__init__.py`
- `generatePrompt.py`
- `verbalisation_module.py`

To install the verbalizer, follow these steps:

1. Download the `output.tar.gz` file from the following [link](https://drive.google.com/file/d/1OW2MkEffc6j-EqiWciMPVN0l56X3bNnS/view?usp=drive_link).

2. Place the downloaded file in the `graph2text` folder:

    ```bash
    cd src/models/verbalizer/graph2text
    ```

3. Unpack the file in the `graph2text` folder:

    ```bash
    tar -xvf output.tar.gz
    ```

4. Run the following commands to combine and extract the model files:

    ```bash
    cd outputs/t5-base_13881/best_tfmr
    cat pytorch_model.bin.tar.gz.parta* > pytorch_model.bin.tar.gz
    tar -xzvf pytorch_model.bin.tar.gz
    ```