'''
Apply NMT model at inference time
Generates a .prediction file
'''

import sys
import os
import argparse
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('OUT', type=str, help='Specify output .prediction file')
    commandLineParser.add_argument('SOURCE', type=str, help='Specify data .source file')
    commandLineParser.add_argument('--translation', type=str, default='en-de', help="Specify translation")
    commandLineParser.add_argument('--num_beams', type=int, default=15, help="Specify number of decoding beams")
    args = commandLineParser.parse_args()

    # Save the command run
    text = ' '.join(sys.argv)+'\n'
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/predict.cmd', 'a') as f:
        f.write(text)
    print(text)

    # Load model
    mname = f'facebook/wmt19-{args.translation}'
    tokenizer = FSMTTokenizer.from_pretrained(mname)
    model = FSMTForConditionalGeneration.from_pretrained(mname)

    # Load source data
    with open(args.SOURCE, 'r') as f:
        sentences = f.readlines()

    # Prediction
    predictions = []
    for i, source in enumerate(sentences):
        print(f'Decoding {i}/{len(sentences)}')
        input_ids = tokenizer.encode(source, return_tensors="pt")
        outputs = model.generate(
            input_ids = input_ids,
            num_beams = args.num_beams,
            do_sample = False,
            max_length = 256,
            length_penalty = 1.0,
            early_stopping = True,
            use_cache = True,
            num_return_sequences = 1)
        prediction = tokenizer.decode(outputs.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(prediction)
        predictions.append(prediction)
    
    # Save to file
    predictions = [predictions[0]] + ['\n'+pred for pred in predictions[1:]]
    with open(args.OUT, 'w') as f:
        f.writelines(predictions)
