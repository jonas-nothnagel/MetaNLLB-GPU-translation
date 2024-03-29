import pandas as pd
import transformers
import json
import os
import time
import torch


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'running on device: {device}')

"""define functions"""
def translate_xcsr(tgt_lang, english_data, path = './data/raw/', print_update = True):

  '''
  Not super refined yet...

  Translates the X-CSR dataset.

  Returns a dictionary.

  Requires pipeline from the transformer library.

  Expects a set up NLLB model, e.g. facebook/nllb-200-distilled-600M or facebook/nllb-200-3.3B from Huggingface

  Expects a loaded the english dev.jsonl X-CSR dataset in english_dict

  Path default is set to run in Google Colab
  '''

  #Set up pipelines


  print('Setting up ' + tgt_lang + ' translator...')
  if tgt_lang == 'swh':
    # Swahili translator
    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang='eng_Latn', tgt_lang='swh_Latn', max_length = 200, device=device)
  elif tgt_lang == 'kik':
    # Kikuyu translator
    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang='eng_Latn', tgt_lang='kik_Latn', max_length = 200, device=device)
  elif tgt_lang == 'luo':
    # Luo translator
    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang='eng_Latn', tgt_lang='luo_Latn', max_length = 200, device=device)
  elif tgt_lang == 'hin':
    # Hindi translator
    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang='eng_Latn', tgt_lang='hin_Deva', max_length = 200, device=device)
  elif tgt_lang == 'bho':
    #Bhojpuri translator
    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang='eng_Latn', tgt_lang='bho_Deva', max_length = 200, device=device)
  else:
    print('Translator has not set up, please add it to the function. And improve the function...')
    return

  #Set up new dataset

  translated_data = []

  #translate

  counter = 1

  for english_dict in english_data:
    start_time = time.time()

    translated_dict = {}

    for i in english_dict.keys():
        translated_dict[i] = None

    translated_dict['id'] = english_dict['id']
    translated_dict['lang'] = tgt_lang

    #question is a dictionary with stem: text, choices: list of dictionaries
    question_translated = {}

    stem = english_dict['question']['stem']

    stem_translated = translator(stem)

    question_translated['stem'] = stem_translated[0]['translation_text']

    #translating the choices
    choices = english_dict['question']['choices']

    choices_translated = []

    for choice in choices:
      translated_choice = {}

      translated_choice['label'] = choice['label']

      text = choice['text']

      text_translated = translator(text)

      translated_choice['text'] = text_translated[0]['translation_text']

      choices_translated.append(translated_choice)

    question_translated['choices'] = choices_translated

    translated_dict['question'] = question_translated

    translated_dict['answerKey'] = english_dict['answerKey']

    translated_data.append(translated_dict)

    end_time = round(time.time() - start_time)

    if print_update:
      print('Successfully translated question number ' + str(counter) + ' from en to ' + tgt_lang + '. Time needed: ' + str(end_time) + ' seconds')

    counter += 1

  return translated_data

def save_translated_data(translated_data, tgt_lang, path = './data/processed/'):
  # Serializing json
  json_object = json.dumps(translated_data, indent=4)

  # Writing to results_sw.json
  with open(path + tgt_lang + '_dev.json', "w") as outfile:
      outfile.write(json_object)

  print('Saved data for ' + tgt_lang + '.')

if __name__ == '__main__':
  
    # load data
    with open('./data/raw/dev.jsonl') as f:
        en = [json.loads(line) for line in f] # work around needed as dataset is in jsonl format

    print('Data loaded with length ' + str(len(en)))

    # load model
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

    #model = 'facebook/nllb-200-distilled-600M'
    model = 'facebook/nllb-200-3.3B' # from https://huggingface.co/facebook/nllb-200-3.3B

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model)

    print("Model loaded!")

    # define target translation languages
    tgt_langs = ['kik', 'bho', 'luo']

    # translate data
    for tgt_lang in tgt_langs:
       translated_data = translate_xcsr(tgt_lang, en, path = './data/raw/', print_update = True)
       save_translated_data(translated_data, tgt_lang, path = './data/processed/')


    print("done!")