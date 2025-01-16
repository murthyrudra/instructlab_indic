# InstructLab Extension for Indic Languages


## Setting up InstructLab
The first step in setting up InstructLab is cloning the InstructLab repository and installing the required packages.

Let us first clone the instructLab repo.
```bash
git clone --recurse-submodules git@github.com:murthyrudra/instructlab_indic.git
```

We will now go to instructlab folder and set few environment variables
```bash
cd instructlab_indic
export XDG_DATA_HOME="$(pwd)/output"
export XDG_STATE_HOME="$(pwd)/output"
export XDG_CONFIG_HOME="$(pwd)/output"
```

## Installing InstructLab

We will now have to install Instructlab from source.

For Mac systems,
```bash
pip install -e .\[mps\]
```

For Linux systems,
```bash
pip install -e .
```

Next we need to install SDG, Schema
```bash
cd sdg
pip install -e .
cd ../schema
pip install -e .
cd ..
```

This will install all the required packages. 

If in case there is any issue with llama-cpp please run the following command
```bash
CMAKE_ARGS="-DGGML_CUDA=on -DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_FMA=off" FORCE_CMAKE=1 pip install --force-reinstall --no-cache-dir "llama-cpp-python[server]==0.2.90"
```

## Initializing InstructLab
Run the following command to initialize instructlab

```bash
ilab config init
```

When it asks for path to the taxonomy, pls provide the absolute path to the `taxonomy` folder.

The next step is downloading the teacher model
```bash
ilab model download
```

## Running Data Generation
This Readme assumes the user is familiar with adding new knowledge or skills to the taxonomy repo. Once done, the following command needs to be run to generate synthetic data

```bash
ilab data generate --taxonomy-path <path to newly added yaml file in taxonomy> --sdg-scale-factor 2 --quiet --output-dir <path to output folder> --model <path to the downloaded teacher model> --chunk-word-count 200 --model-family phi
```

The synthetic data will be generated and present in the `output` folder.