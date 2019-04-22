# aspect
## Description 
cross-domain aspect extraction

http://www.zmonster.me/2016/06/08/use-stanford-nlp-package-in-nltk.html


from nltk.internals import find_jars_within_path
from nltk.tag import StanfordPOSTagger
from nltk import word_tokenize

# Alternatively to setting the CLASSPATH add the jar and model via their path:
jar = '/Users/nischi/PycharmProjects/stanford-postagger-full-2015-12-09/stanford-postagger.jar'
model = '/Users/nischi/PycharmProjects/stanford-postagger-full-2015-12-09/models/english-left3words-distsim.tagger'

pos_tagger = StanfordPOSTagger(model, jar)

# Add other jars from Stanford directory
stanford_dir = pos_tagger._stanford_jar.rpartition('/')[0]
stanford_jars = find_jars_within_path(stanford_dir)
pos_tagger._stanford_jar = ':'.join(stanford_jars)

text = pos_tagger.tag(word_tokenize("What's the airspeed of an unladen swallow ?"))
print(text)
