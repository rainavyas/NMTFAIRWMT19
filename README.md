# Objective

Use [Facebook's Fair model](https://huggingface.co/facebook/wmt19-de-en) submitted to the WMT19 translation task.
Evaluation is performed on the WMT19 test set using the [sacreBLEU tool](https://github.com/mjpost/sacreBLEU).

# Requirements

python3.6 or above

## Install with PyPI
`pip install torch transformers sacrebleu`

# Quick Start

Download the dataset using the CLI:

```console
mkdir -p $DATA_DIR
sacrebleu -t wmt19 -l $PAIR --echo src > $DATA_DIR/val.source
sacrebleu -t wmt19 -l $PAIR --echo ref > $DATA_DIR/val.target
```
