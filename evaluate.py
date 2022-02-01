'''
Evaluate predictions from model

Expect files at input:
.target
.prediction
'''

import sys
import os
import argparse
from sacrebleu.metrics import BLEU, CHRF, TER

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('TARGET', type=str, help='Specify data .target file')
    commandLineParser.add_argument('PRED', type=str, help='Specify data .prediction file')
    args = commandLineParser.parse_args()

    # Save the command run
    text = ' '.join(sys.argv)+'\n'
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/evaluate.cmd', 'a') as f:
        f.write(text)
    print(text)

    # Load files
    with open(args.TARGET, 'r') as f:
        targets = f.readlines()
    with open(args.PRED, 'r') as f:
        predictions = f.readlines()
    
    assert len(targets)==len(predictions), "Mismatch in target and prediction file items"

    # BLEU score
    bleu = BLEU()
    result =  bleu.corpus_score(predictions, targets)
    print(f'BLEU:\t{result}')
    
    # CHRF score
    chrf = CHRF()
    result =  chrf.corpus_score(predictions, targets)
    print(f'CHRF:\t{result}')

    # TER score
    ter = TER()
    result =  ter.corpus_score(predictions, targets)
    print(f'TER:\t{result}')
