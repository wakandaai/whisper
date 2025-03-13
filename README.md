# Whisper for IWSLT low resource

## Results

### BigC S2TT
| BLEU    |chrF     | Model | Language Token | Beam Size | Temperature |
|---------|---------|-------|--------------- |-----------|-------------|
|  2.1206 |  11.6449| large | sw             | 0         | 0.2         |
|  2.2547 |  13.5129| large | sw             | 0         | 0.2         |
|  22.6308 |  46.8872| tiny-ft | sw             | 5         | 0.2         |
|  27.5882 |  52.3598 | small-ft | sw             | 5         | 0.2         |


### BembaSpeech ASR
| WER     | Model | Language Token | Beam Size | Temperature |
|---------|-------|--------------- |-----------|-------------|
| 1.3467  | turbo | none           | 0         | 0.0         |
| 1.1781  | turbo | sw             | 0         | 0.0         |
| 1.0453  | large | sw             | 0         | 0.0         |
| 1.0444  | large | sw             | 0         | 0.2         |

### BigC ASR (finetuned)
| WER    | Model | Language Token | Beam Size | Temperature | Epochs |
|---------|-------|--------------- |-----------|-------------|-------------|
| 0.5404  | tiny-ft | sw             | 0         | 0.2         |1|
| 0.4946 | tiny-ft | sw             | 5         | 0.2         |1|
| 0.4244 | tiny-ft | sw             | 5         | 0.2         |5|
| 0.4133 | small-ft | sw             | 5         | 0.2         |2|


## Setup

### Environment
```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
# answer yes to terms and to automatically setting up Miniconda
# reopen terminal
conda deactivate
conda create -n whisper python=3.10
conda activate whisper
```
### Repo

    pip install git+https://github.com/Alexgichamba/whisper.git 

To update the package to the latest version of this repository, please run:

    pip install --upgrade --no-deps --force-reinstall git+https://github.com/Alexgichamba/whisper.git

It also requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

You may need [`rust`](http://rust-lang.org) installed as well, in case [tiktoken](https://github.com/openai/tiktoken) does not provide a pre-built wheel for your platform. If you see installation errors during the `pip install` command above, please follow the [Getting started page](https://www.rust-lang.org/learn/get-started) to install Rust development environment. Additionally, you may need to configure the `PATH` environment variable, e.g. `export PATH="$HOME/.cargo/bin:$PATH"`. If the installation fails with `No module named 'setuptools_rust'`, you need to install `setuptools_rust`, e.g. by running:

```bash
pip install setuptools-rust
```
## Train and Eval guide
First, download the dataset, for example BembaSpeech or BigC from their publically available sources.

Then do dataprep for example here: this is for bigC with translation as the task.
```shell
python3 iwslt/utils/dataprep_bigc.py --base_path ~/corpora/bigc/data/bem/ --output_dir corpora/ --task translate
```
You can then make a training config file or use an existing one to call the trainer.
```shell
python3 whisper/trainer.py --config iwslt/train_conf/whisper_bigc_s2tt_ft.yaml
```
Lastly, evaluate your model
```shell
python3 iwslt/evaluate/run_eval.py --model results/whisper_bigc_s2tt_ft/checkpoint_5.pt --temperature 0.2 --batch_size 1 --beam_size 5
```

## License

Whisper's code and model weights are released under the MIT License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for further details.
