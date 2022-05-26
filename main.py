import json
import os
import argparse
from typing import Dict,  List, Optional
from collections import Counter
import fasttext
from pycountry import languages
from apify_client import ApifyClient

def get_language_id_model():
    """load fastText model 

    Returns:
        obj: language detection model
    """
    PRETRAINED_MODEL_PATH = "lid.176.ftz"
    model = fasttext.load_model(PRETRAINED_MODEL_PATH)
    fasttext.FastText.eprint = lambda x: None
    return model


def get_language_of_text(text:str, min_len:Optional[int] = 0)->str:
    """returns the most likely language of a sentence, removing sentence shorter than min_len

    Args:
        text (str): a sentence
        min_len (int, optional): skip sentences below min length threshold. language detection might not work well on ultra-short sentence with 1-2 words, but if having menu items with only 1 word you might want to set this to 0. Defaults to 0.

    Returns:
        str: detected language
    """

    model = get_language_id_model()
    sentences_raw = text.split("\n")
    sentences = [i for i in sentences_raw if len(i.split(" ")) > min_len]
    predictions = model.predict(sentences)

    return max(
        list(Counter([x[0].replace("__label__", "") for x in predictions[0]]).items()),
        key=lambda x: x[1],
    )[0]


def get_language_mix(txt_lst:List[str], min_len:Optional[int] = 0, show_language_names:Optional[bool] = False) -> Dict[str, int]:
    """returns breakdown of language for each sentence

    Args:
        txt_lst (list): list of sentences. this is the text portion from the html, could contain menu bar text or body text, excluding the html tags etc
        min_len (int, optional): skip sentences below min length threshold. Defaults to 0 to include all sentences.
        show_language_names(bool, optional): whether to show language name (e.g. English) instead of iso2 code (e.g. EN) Defaults to false
    Returns:
        dict: a dictionary with language name as key and number of sentences in that language as value
    """
    lang_lst = []
    n_skip = 0
    for t in txt_lst:
        try:
            lang_detected = get_language_of_text(t, min_len=min_len)
            lang_lst.append(lang_detected)
        except:
            n_skip += 1
            pass
    if n_skip > 0:
        print(f"skipped {n_skip} sentences")

    lang_code_counter = Counter(lang_lst).items()
    
    if show_language_names is True:
        lang_name_counter = {
            languages.get(alpha_2=k).name: v for (k, v) in lang_code_counter
        }
        return lang_name_counter
    else:
        return dict(lang_code_counter)


if __name__ == "__main__":
    # Initialize the main ApifyClient instance
    client = ApifyClient(os.environ['APIFY_TOKEN'], api_url=os.environ['APIFY_API_BASE_URL'])

    # Get the resource subclient for working with the default key-value store of the actor
    default_kv_store_client = client.key_value_store(os.environ['APIFY_DEFAULT_KEY_VALUE_STORE_ID'])

    # Get the value of the actor input and print it
    actor_input = default_kv_store_client.get_record(os.environ['APIFY_INPUT_KEY'])['value']
    print('Actor input:')
    print(json.dumps(actor_input, indent=2))
    print(actor_input)

    print(get_language_mix(actor_input['texts'], min_len=0, show_language_names=True))
