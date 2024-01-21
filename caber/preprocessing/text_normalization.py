from unidecode import unidecode
import nltk
import re

clean_html_tag = re.compile('<.*?>')


###############################################################################
# Text Normalization
class TextNormalization:
    """
    The Text Normalization is useful text preprocessing
    -----------------------
    Parameters:
    data
    -----------------------
    Methods:
    lower_text:
    remove_html_tags:
    translate_unicode_to_ascii:
    """
    def __init__(self,
                 data,
                 column):
        self.data = data
        self.column = column
    
    def lower_text(self):
        self.data.loc[:,self.column] = self.data[self.column].str.lower()

    def remove_html_tags(self):
        self.data.loc[:,self.column] = [re.sub(clean_html_tag, '', text) for text in self.data[self.column]]
    
    def translate_unicode_to_ascii(self):
        self.data.loc[:,self.column] = [unidecode(text) for text in self.data[self.column]]
    
    