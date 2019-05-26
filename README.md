
# Title Generation using Recurrent Neural Networks
I never know what I should title most things I have written. I hope that by using a corpus of titles, recurrent neural networks (RNNs) can write my titles for me.

I thought a fitting title to generate would be something within Machine Learning, so I used [Publish or Perish](https://harzing.com/resources/publish-or-perish) to fetch any title from Google Scholar associated with *Machine Learning*. It retrieved 950 titles, which you can view [here](https://gist.github.com/AngusTheMack/defadcbc503e2d625720661e9893ff0a). 

If you want to use this to  generate your own titles (or any text whatsoever), just change the `url` to download the data you want to use, or change the `save_location` to where your data is stored.

## Titles Generated
During the time playing around with the implementations below I was able to gain some very cool sounding titles:
  * Function Classification Using Machine Learning Techniques
  * Bayesian Approximation of Effective Machine Learning
  * Data Classification With Machine Learning
  * Computer Multi-agent Boltzmann Machine Learning
  * Machine Learning Approaches for Visual Classification
  * New Machine Learning for Astrophysics
  * Neural Machine Learning for Medical Imaging
  * Deep Similarity Learning Filters

You can view some more of the outputs here:
  - [Char Level Generated Titles](#char-level-generated-titles)
  - [Word Level Generated Titles](#word-level-generated-titles)
  - [LSTM Generated Titles](#lstm-generated-titles)
  
## Implementations
I wanted to compare results between somewhat vanilla RNN implementations and a Long Short Term Memory (LSTM) model. To that end I used a character level RNN, word level RNN and a LSTM. This was done mainly to try and better understand the underlying concepts in RNNs, and what differentiates them from LSTMs.

I used [Andrej Karpathy's blog](https://karpathy.github.io/) post [The Unreasonable Effectiveness of Recurrent Neural Networks ](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) as my starting point - and utilised his amazing [112 line char level RNN](https://gist.github.com/karpathy/d4dee566867f8291f086) implemented in vanilla python.

After that I used [Denny Britz's](https://github.com/dennybritz) [word level RNN](https://github.com/dennybritz/rnn-tutorial-rnnlm/blob/master/RNNLM.ipynb) from his series of [blogsposts](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/) on the topic.

Finally, I used [Shivam Bansal's](https://www.kaggle.com/shivamb) [Beginners Guide to Text Generation using LSTMs](https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms/notebook) for the LSTM implementation.


# Results
When trying to analyse each different method I used the number of titles that made sense from start to finish and whether a title contained a sub-string that made sense. I named these two metrics **Coherent Titles** and **Coherent Sub-strings**.

I then generated 15 titles with each model and calculated the following results:

|Model|Coherent Titles|Coherent Sub-strings|
|------|-----|-----|
|Char RNN|6.67%|6.67%|
|Word RNN|40%|53%|
|LSTM|60%|100%|

Its apparent that the LSTM outperforms the RNNs, but that was to be expected. I think the word level RNN is actually quite good, and the char level one can definitely be improved upon but wasn't entirely terrible. Also, the dataset is quite small. With a large corpus I think the results would likely improve. 

However, a more formalised method for comparing the models is definitely necessary for further research.

# Going Forward
I think a possible method of comparing the different models would be to use a language model that can indicate whether a sentence makes sense to some degree. Then that could be used on the generated titles in order to extrapolate some more meaningful and reproducible results. I was advised by my lecturer that a possible method of doing this was to use something like [Google Ngram](https://books.google.com/ngrams/info), and check whether a title or a substring of a title has been previously used to a certain degree. If it has, then it likely makes some sense.

The parameters for the different implementations can also be experimented with to gain better results.

I was also advised that an interesting area of research would be to generate a title for your paper (or writings) based on the abstract (or some subsection of your writings).  This would very likely lead to titles that are more related to the actual content.

This was a very fun and interesting experience, and was inspired by the following:
* [Harry Potter and the Portrait of what Looked like a Large Pile of Ash by Botnik ](https://botnik.org/content/harry-potter.html)
* [King James Programming](https://kingjamesprogramming.tumblr.com/)
* [Alice in Elsinore](https://www.eblong.com/zarf/markov/alice_in_elsinore.txt) from [Fun with Markov Chains](https://www.eblong.com/zarf/markov/)
* [Stack Exchange Simulator](https://se-simulator.lw1.at/)
* [Pun Generation with Suprise](https://github.com/hhexiy/pungen)

# Running The Code
If you run this via [Google Colab](https://colab.research.google.com/) it should work from the get go. If you want to run it on your local
try to install the necessary requirements via conda **Note:** I haven't had a chance to check the conda installation - so sadly there may be one or two issues regarding missing packages.
```
conda create --name titles --file requirements.txt
```
Accept any prompt that pops up, activate the environment `conda activate titles` and then add the necessary kernel for the jupyter notebook:
```
python -m ipykernel install --user --name=titles
```

# Code
Here is the code and associated output from the [RNNs.ipynb](RNNs.ipynb) notebook. If you want to look at a specific implementation:
- [Char Level RNN](#char-level-rnn)
- [Word Level RNN](#word-level-rnn)
- [LSTM](#lstm)
```python
import numpy as np
import matplotlib.pyplot as plt
import string
import urllib.request
import pickle
%matplotlib inline
```


```python
def download_data(url, save_location):
    """
    Download data to be used as corpus
    """
    print('Beginning file download...')  
    urllib.request.urlretrieve(url,save_location)
    print("Downloaded file, saving to:",save_location)

def load_data(save_location):
    """
    Load data from Textfile
    """
    file = open(save_location,"r")
    data = file.read()
    return data

def avg_char_per_title(data):
    """
    Calculate the average number of chars in a title for the sequence length
    """
    lines = data.split("\n")
    line_lengths = np.zeros(len(lines))
    for i,line in enumerate(lines):
        line_lengths[i] = len(line)
    return np.average(line_lengths)
        

def save_object(obj, filename):
    """
    Save an object - used to save models
    """
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, -1)
```


```python
# Change the URL to whatever text you want to train with
url = "https://gist.githubusercontent.com/AngusTheMack/defadcbc503e2d625720661e9893ff0a/raw/bb978a5ef025ff104009ab8139da4a0b7367992f/Titles.txt"

# Save Location will be used to load the data in
save_location = "Titles.txt" # either the name of the file downloaded with the URL above, or the location of your own file to load in
```


```python
# Downloads the data, and loads it in
download_data(url,save_location)
data = load_data(save_location)
```

    Beginning file download...
    Downloaded file, saving to: Titles.txt



```python
# Print first 100 characters of the data
print(data[:100])
```

    Scikit-learn: Machine learning in Python
    Pattern recognition and machine learning
    Gaussian processes



```python
def clean_text(data):
    """
    Removes non essential characters in corpus of text
    """
    data = "".join(v for v in data if v not in string.punctuation).lower()
    data = data.encode("utf8").decode("ascii",'ignore')
    return data
```


```python
# You don't need to clean, but it can make things simpler
cleaned = clean_text(data)
print(cleaned[:100])
```

    scikitlearn machine learning in python
    pattern recognition and machine learning
    gaussian processes i



```python
def unique_chars(data):
    """
    Get all unique Characters in the Dataset
    """
    return list(set(data))
```


```python
# Some info about the data
chars = unique_chars(cleaned)
data_size, input_size = len(cleaned), len(chars)
print("Data has %d characters, %d of them are unique" % (data_size, input_size))
```

    Data has 64663 characters, 36 of them are unique



```python
def tokenize_chars(chars):
    """
    Create dictionaries to make it easy to convert from tokens to chars
    """
    char_to_idx = {ch:i for i,ch in enumerate(chars)}
    idx_to_char = {i:ch for i,ch in enumerate(chars)}
    return char_to_idx, idx_to_char
```


```python
# Create dictionaries, and display example using 11 chars
char_to_idx, idx_to_char = tokenize_chars(chars)
first_title = cleaned[:11]
print("{0:<2}|{1:<2}".format('Character', 'Index'))
print("________________")
for i in range(len(first_title)):
    char_index = char_to_idx[first_title[i]]
    print("{0:<9}|{a:d}".format(idx_to_char[char_index], a=char_to_idx[first_title[i]]))
```

    Character|Index
    ________________
    s        |12
    c        |28
    i        |29
    k        |19
    i        |29
    t        |7
    l        |2
    e        |18
    a        |9
    r        |10
    n        |6


# Char Level RNN
Created by Andrej Karpathy, available here:  [here](https://gist.github.com/karpathy/d4dee566867f8291f086)


```python
"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License

Ever so slightly modified to be used with the above code
"""
data = cleaned
chars = unique_chars(cleaned)
data_size, vocab_size = len(cleaned), len(chars)

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_idx[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_idx[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(idx_to_char[ix] for ix in sample_ix)
    print('----\n %s \n----' % (txt, ))

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss)) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 
```

    ----
     wzaobr1wn1v0i1s4dfxsy nzzhacv6wnmx3h3al4rdeuuau5s36la1gv 
    wpoucjnqgg1zwc1qy7ug 6gmn3jr0 7mxr07 ls3
    bzrjcrjd0g12g nrb00fzrnxeyfturlbrrmxuiq5vccy 61om
    auqwdaepa5g1atew
    o6kqtu0pyu72ikrlqdqu5vripn a
    7e6q1 
    ----
    iter 0, loss: 89.587970
    ----
     ieetaihi mkn r pareutm shierulunetanec aem nept  ren lsigne7 
    er metenetahhveceyem l gayepgrehatn2enaa ieiop pnnti lslecelom
    tjnet mew dtiestmeaoi 4ensl lcidiereeon
    i rhi h rnipi  sctiebh  lng senniem 
    ----
    iter 100, loss: 90.192833
    ----
     oa
    fghic7tn0tmtatenabcna wiaeprlcsaalsrs fele ea1hyah r inearfro tnrd rand
    d hiagee ntefghrtooieetnpnaea s fearp
    aeaeeuesenptkafysii trwiu nea t liama  renaupibsa 
    rgmpopealcy ynaaaurd
    aeor5e  llirryf 
    ----
    iter 200, loss: 88.911862
    ----
    This goes on for a few thousand iterations...
    ----
     ne learning
    in machine learcing stal learning news seate bleturkizic machine learning of procyndaring the proxadeoric
    new lorplivisces
    metrods
    aruntitis planitive mo learning frorkatiog approsied sta  
    ----
    iter 111700, loss: 33.573679
    ----
     anjiong algorpal intracper ins statrectiven muttl elalal and binenussifich ovistins using provie fobiod attramizantily shegeclanal hy machine learning balser clodillar intato chods
    destin intetrewomus 
    ----
    iter 111800, loss: 33.693775
    ----
     machine learning mathine gupuinimg hezimizitan peult learning
    doterth detach
    weng to andrime
    the many pactem
    machine learning asqforc
    sumast
    cotition and using dethe state algormicitatfor rang chopilo 
    ----
    iter 111900, loss: 33.613680

I stopped the above compilation after 1119000 iterations as it takes quite a while to generate meaninful text - and sometimes it doesn't seem to converge at all. Here is the output from an implementation I had running for a  day or two that got down to about 16 for its loss.

## Char Level Generated Titles
```
Oxprensur Machine Learning Based Comparison Imagepredalyic Problem A Machine Learning Shidenticing With Stomement Machine
Genetional Translingl Data O
Tby Of Panadigunoous Of Machine Learning Approach
Machine Learning Approaches And Ancerxards
Applications
Ortamenopforcher Image Of And Comparison Hen
Bytesca For Dete
Semapt Recognition
Neural Ontropicaty Stvediction
Thance Resules Of Machinelearning Based And Machine Learning
Ma
Rward Algorithms
Thek Support Vector Machine Learning Toces
Survey
Subperai Scalistose Machine Learning
Classer Ald Optimization
Spatsimentar Scanisys
Twarites In Machine Learning For Algorithms
Realtime S Forildetion For Support Vector Machine Learning Techniques For The Laond Machine Learning For S
Syppbys
Mumporaty Researchon Using Of Temporing
Entruasian Designs Spevied Alghid Machine Learning
Clesit A Dizen Interaninergopers
Machine Learning
D
Operpne Mencal Work2Bated Athito Mativing Thootimic Optoraty For Machine Learning Methodent Methods In Dete Detection Of The Ancherch Of Contratecompu
Hacingar Proborion
Machine Learning In Metric Learning Transif Trassing An Learning
Machine Learning Audomement Machine Learning Of Machine Learning T
Ttymane Learning Coneftrand An Application For Mmfes On Undersec Auport Text Machine Learning A Machine Learning With Stalsaby Data Misuse Contronimic
Rsenticing Machineleseratigg
Machinelearning Of Vector
Machine Learning
Hungersing On Machine Learning And Activity
Approach To Trugbal Machine Learni
Rcemative Learning
Machine Learning And Compilianc User Introppshibution Of Brain Berial Distoneer Machine Learning
Discovery Descnessow Of Ant Seqmen
Oventicing Using Recognstimessing Practical Frainetation
Mesticabily For Parxam Experimaphitist Besk Coxican
Machine Learning Bos Automated Machine Le
Fxamentle Image Of Machine Learning Gave Trapean Schemass Of Machine Learning Of Methods Inty On Combinion Gane Technical Deabficimation Classaletrati
Esintiafforcemental Nerkase Deterabe Optimization Agversitoraling
A For Decision Techniques And Optimization For Usey In Machine Learning Corsed Machi
Onedential Machine Learning
Detection
Drepoutivelearning Machine Learning
Computtess Design Re6Aition To By Intempregressir Tomation
Suportiva Contere
Raph Incrotelaxics Ylame Tring Code
Anemoriomative Reperimity In Paraller
Munt Langouupmi Plediction Of Machine Learning
Predicting Prowibley Increman
Ecosting Machine Learning
Predict Learning And Smanced
Machine Learning
Data With Machine Learning Toateraby Ougcing Word Feature Ussifbees
Jachi Elar
Dations
Analysis Of Liagn Twictite Classification
Patferetistic Prospe Identificies Clamngenoun
Progmaris
Machine Learning For Anpreaching Methoduntac
Ocion Ad Applisition Reclasy Envinids
Quantsys A Otsum Mazining A Machine Learning
Machine Learning
Machine Learning
Extraction
Machine Learning Appro
Iches Using Machine Learning Pprssmase To Machine Learning Approach To Filteral Progrom Om Feremble Identifica Optiman Enviroptimization Of The Use In
```

As you can see, they are generally quite nonsensical. Although, the simple RNN does latch onto a few words that it has learned based on character sequences alone which is really cool! It has basically learned a tiny and very focused bit of the english language simply by trying to guess the next character.

# Word Level RNN
The second implementation uses [Denny Brit'z Word level model](https://github.com/dennybritz/rnn-tutorial-rnnlm)


```python
import csv
import itertools
import operator
import nltk
import sys
from datetime import datetime
```


```python
# Chops the stream of titles into an array of titles based on new line characters
titles = cleaned.split("\n")
titles[0]
```




    'scikitlearn machine learning in python'




```python
unknown_token = "UNKNOWN_TOKEN"
title_start_token = "SENTENCE_START"
title_end_token = "SENTENCE_END"
```


```python
# Add the start and end token to the title
titles = ["%s %s %s" % (title_start_token, x, title_end_token) for x in titles]
```


```python
# Ensure that nltk has the punkt package
nltk.download('punkt')
```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.
    True




```python
tokenized_titles = [nltk.word_tokenize(t) for t in titles]
```


```python
word_freq = nltk.FreqDist(itertools.chain(*tokenized_titles))
print("Found %d unique words tokens." % len(word_freq.items()))
```

    Found 1841 unique words tokens.



```python
vocabulary_size=2000#len(word_freq.items())
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
```


```python
print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))
```

    Using vocabulary size 2000.
    The least frequent word in our vocabulary is 'ethical' and appeared 1 times.



```python
# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_titles):
    tokenized_titles[i] = [w if w in word_to_index else unknown_token for w in sent]
```


```python
print("\nExample sentence: '%s'" % titles[0])
print("\nExample sentence after Pre-processing: '%s'" % tokenized_titles[0])
```

    
    Example sentence: 'SENTENCE_START scikitlearn machine learning in python SENTENCE_END'
    
    Example sentence after Pre-processing: '['SENTENCE_START', 'scikitlearn', 'machine', 'learning', 'in', 'python', 'SENTENCE_END']'



```python
# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_titles])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_titles])
```


```python
# Print training data example
x_example, y_example = X_train[17], y_train[17]
print("x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example))
print("\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example))
```

    x:
    SENTENCE_START supervised machine learning a review of classification techniques
    [0, 66, 3, 2, 7, 49, 6, 16, 18]
    
    y:
    supervised machine learning a review of classification techniques SENTENCE_END
    [66, 3, 2, 7, 49, 6, 16, 18, 1]



```python
def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)
```


```python
class RNNNumpy:
    
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
```


```python
def forward_propagation(self, x):
    # The total number of time steps
    T = len(x)
    # During forward propagation we save all hidden states in s because need them later.
    # We add one additional element for the initial hidden, which we set to 0
    s = np.zeros((T + 1, self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim)
    # The outputs at each time step. Again, we save them for later.
    o = np.zeros((T, self.word_dim))
    # For each time step...
    for t in np.arange(T):
        # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
        s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
        o[t] = softmax(self.V.dot(s[t]))
    return [o, s]

RNNNumpy.forward_propagation = forward_propagation

def predict(self, x):
    # Perform forward propagation and return index of the highest score
    o, s = self.forward_propagation(x)
    return np.argmax(o, axis=1)

RNNNumpy.predict = predict
```


```python
np.random.seed(10)
model = RNNNumpy(vocabulary_size)
o, s = model.forward_propagation(X_train[10])
print(o.shape)
print(o)
```

    (5, 2000)
    [[0.00050094 0.00049585 0.00050577 ... 0.00050363 0.00049082 0.00049915]
     [0.00050011 0.00050381 0.00050253 ... 0.00050514 0.00050839 0.0005072 ]
     [0.00050025 0.00049864 0.00049696 ... 0.00049498 0.00049688 0.00050403]
     [0.00050167 0.00050213 0.00049959 ... 0.00049484 0.00050239 0.00050337]
     [0.00050468 0.00049741 0.00050422 ... 0.00050882 0.00050223 0.00051026]]



```python
predictions = model.predict(X_train[10])
print(predictions.shape)
print(predictions)
```

    (5,)
    [1755  202    3 1314 1300]



```python
def calculate_total_loss(self, x, y):
    L = 0
    # For each sentence...
    for i in np.arange(len(y)):
        o, s = self.forward_propagation(x[i])
        # We only care about our prediction of the "correct" words
        correct_word_predictions = o[np.arange(len(y[i])), y[i]]
        # Add to the loss based on how off we were
        L += -1 * np.sum(np.log(correct_word_predictions))
    return L

def calculate_loss(self, x, y):
    # Divide the total loss by the number of training examples
    N = np.sum((len(y_i) for y_i in y))
    return self.calculate_total_loss(x,y)/N

RNNNumpy.calculate_total_loss = calculate_total_loss
RNNNumpy.calculate_loss = calculate_loss
```


```python
# Limit to 1000 examples to save time
print("Expected Loss for random predictions: %f" % np.log(vocabulary_size))
print("Actual loss: %f" % model.calculate_loss(X_train[:1000], y_train[:1000]))
```

    Expected Loss for random predictions: 7.600902


    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:14: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      


    Actual loss: 7.601434



```python
def bptt(self, x, y):
    T = len(y)
    # Perform forward propagation
    o, s = self.forward_propagation(x)
    # We accumulate the gradients in these variables
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.
    # For each output backwards...
    for t in np.arange(T)[::-1]:
        dLdV += np.outer(delta_o[t], s[t].T)
        # Initial delta calculation
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
        # Backpropagation through time (for at most self.bptt_truncate steps)
        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            dLdW += np.outer(delta_t, s[bptt_step-1])              
            dLdU[:,x[bptt_step]] += delta_t
            # Update delta for next step
            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
    return [dLdU, dLdV, dLdW]

RNNNumpy.bptt = bptt
```


```python
def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
    # Calculate the gradients using backpropagation. We want to checker if these are correct.
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to check.
    model_parameters = ['U', 'V', 'W']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter = operator.attrgetter(pname)(self)
        print("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            gradplus = model.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            gradminus = model.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            # Reset parameter to original value
            parameter[ix] = original_value
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                print("+h Loss: %f" % gradplus)
                print("-h Loss: %f" % gradminus)
                print("Estimated_gradient: %f" % estimated_gradient)
                print("Backpropagation gradient: %f" % backprop_gradient)
                print("Relative Error: %f" % relative_error)
                return 
            it.iternext()
        print("Gradient check for parameter %s passed." % (pname))

RNNNumpy.gradient_check = gradient_check

# To avoid performing millions of expensive calculations we use a smaller vocabulary size for checking.
grad_check_vocab_size = 100
np.random.seed(10)
word_model = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
word_model.gradient_check([0,1,2,3], [1,2,3,4])
```

    Performing gradient check for parameter U with size 1000.
    Gradient Check ERROR: parameter=U ix=(0, 0)
    +h Loss: 30.432536
    -h Loss: 30.432536
    Estimated_gradient: 0.000000
    Backpropagation gradient: -0.177072
    Relative Error: 1.000000



```python
# Performs one step of SGD.
def numpy_sdg_step(self, x, y, learning_rate):
    # Calculate the gradients
    dLdU, dLdV, dLdW = self.bptt(x, y)
    # Change parameters according to gradients and learning rate
    self.U -= learning_rate * dLdU
    self.V -= learning_rate * dLdV
    self.W -= learning_rate * dLdW

RNNNumpy.sgd_step = numpy_sdg_step
```


```python
# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
```


```python
np.random.seed(10)
word_model = RNNNumpy(vocabulary_size)
%timeit model.sgd_step(X_train[10], y_train[10], 0.005)
```

    100 loops, best of 3: 8.06 ms per loop



```python
np.random.seed(10)
model = RNNNumpy(vocabulary_size)
losses = train_with_sgd(model, X_train[:1000], y_train[:1000], nepoch=100, evaluate_loss_after=1)
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:14: DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
      


    2019-05-25 15:41:43: Loss after num_examples_seen=0 epoch=0: 7.601434
    2019-05-25 15:42:00: Loss after num_examples_seen=950 epoch=1: 5.100385
    2019-05-25 15:42:17: Loss after num_examples_seen=1900 epoch=2: 4.883128
    2019-05-25 15:42:34: Loss after num_examples_seen=2850 epoch=3: 4.780716
    2019-05-25 15:42:51: Loss after num_examples_seen=3800 epoch=4: 4.696207
    2019-05-25 15:43:09: Loss after num_examples_seen=4750 epoch=5: 4.619038
    2019-05-25 15:43:26: Loss after num_examples_seen=5700 epoch=6: 4.552703
    2019-05-25 15:43:43: Loss after num_examples_seen=6650 epoch=7: 4.491730
    2019-05-25 15:44:00: Loss after num_examples_seen=7600 epoch=8: 4.434393
    2019-05-25 15:44:18: Loss after num_examples_seen=8550 epoch=9: 4.380702
    2019-05-25 15:44:35: Loss after num_examples_seen=9500 epoch=10: 4.329521
    2019-05-25 15:44:53: Loss after num_examples_seen=10450 epoch=11: 4.280264
    2019-05-25 15:45:10: Loss after num_examples_seen=11400 epoch=12: 4.233772
    2019-05-25 15:45:28: Loss after num_examples_seen=12350 epoch=13: 4.191475
    2019-05-25 15:45:45: Loss after num_examples_seen=13300 epoch=14: 4.154720
    2019-05-25 15:46:02: Loss after num_examples_seen=14250 epoch=15: 4.105941
    2019-05-25 15:46:20: Loss after num_examples_seen=15200 epoch=16: 4.067841
    2019-05-25 15:46:37: Loss after num_examples_seen=16150 epoch=17: 4.036423
    2019-05-25 15:46:54: Loss after num_examples_seen=17100 epoch=18: 3.982951
    2019-05-25 15:47:12: Loss after num_examples_seen=18050 epoch=19: 3.947387
    2019-05-25 15:47:29: Loss after num_examples_seen=19000 epoch=20: 3.915304
    2019-05-25 15:47:46: Loss after num_examples_seen=19950 epoch=21: 3.888233
    2019-05-25 15:48:03: Loss after num_examples_seen=20900 epoch=22: 3.858850
    2019-05-25 15:48:21: Loss after num_examples_seen=21850 epoch=23: 3.827078
    2019-05-25 15:48:38: Loss after num_examples_seen=22800 epoch=24: 3.806026
    2019-05-25 15:48:55: Loss after num_examples_seen=23750 epoch=25: 3.791999
    2019-05-25 15:49:12: Loss after num_examples_seen=24700 epoch=26: 3.776599
    2019-05-25 15:49:29: Loss after num_examples_seen=25650 epoch=27: 3.760362
    2019-05-25 15:49:47: Loss after num_examples_seen=26600 epoch=28: 3.751018
    2019-05-25 15:50:04: Loss after num_examples_seen=27550 epoch=29: 3.757270
    Setting learning rate to 0.002500
    2019-05-25 15:50:22: Loss after num_examples_seen=28500 epoch=30: 3.640106
    2019-05-25 15:50:39: Loss after num_examples_seen=29450 epoch=31: 3.625389
    2019-05-25 15:50:57: Loss after num_examples_seen=30400 epoch=32: 3.616406
    2019-05-25 15:51:14: Loss after num_examples_seen=31350 epoch=33: 3.613986
    2019-05-25 15:51:31: Loss after num_examples_seen=32300 epoch=34: 3.610226
    2019-05-25 15:51:48: Loss after num_examples_seen=33250 epoch=35: 3.601274
    2019-05-25 15:52:06: Loss after num_examples_seen=34200 epoch=36: 3.594444
    2019-05-25 15:52:23: Loss after num_examples_seen=35150 epoch=37: 3.587431
    2019-05-25 15:52:40: Loss after num_examples_seen=36100 epoch=38: 3.595749
    Setting learning rate to 0.001250
    2019-05-25 15:52:57: Loss after num_examples_seen=37050 epoch=39: 3.534395
    2019-05-25 15:53:14: Loss after num_examples_seen=38000 epoch=40: 3.523400
    2019-05-25 15:53:32: Loss after num_examples_seen=38950 epoch=41: 3.510381
    2019-05-25 15:53:49: Loss after num_examples_seen=39900 epoch=42: 3.500720
    2019-05-25 15:54:06: Loss after num_examples_seen=40850 epoch=43: 3.495431
    2019-05-25 15:54:23: Loss after num_examples_seen=41800 epoch=44: 3.490556
    2019-05-25 15:54:41: Loss after num_examples_seen=42750 epoch=45: 3.488818
    2019-05-25 15:54:58: Loss after num_examples_seen=43700 epoch=46: 3.485845
    2019-05-25 15:55:15: Loss after num_examples_seen=44650 epoch=47: 3.483638
    2019-05-25 15:55:33: Loss after num_examples_seen=45600 epoch=48: 3.481622
    2019-05-25 15:55:51: Loss after num_examples_seen=46550 epoch=49: 3.478715
    2019-05-25 15:56:08: Loss after num_examples_seen=47500 epoch=50: 3.477929
    2019-05-25 15:56:26: Loss after num_examples_seen=48450 epoch=51: 3.479510
    Setting learning rate to 0.000625
    2019-05-25 15:56:43: Loss after num_examples_seen=49400 epoch=52: 3.429259
    2019-05-25 15:57:00: Loss after num_examples_seen=50350 epoch=53: 3.424342
    2019-05-25 15:57:17: Loss after num_examples_seen=51300 epoch=54: 3.418742
    2019-05-25 15:57:35: Loss after num_examples_seen=52250 epoch=55: 3.414435
    2019-05-25 15:57:52: Loss after num_examples_seen=53200 epoch=56: 3.410370
    2019-05-25 15:58:09: Loss after num_examples_seen=54150 epoch=57: 3.406576
    2019-05-25 15:58:27: Loss after num_examples_seen=55100 epoch=58: 3.403000
    2019-05-25 15:58:44: Loss after num_examples_seen=56050 epoch=59: 3.400307
    2019-05-25 15:59:01: Loss after num_examples_seen=57000 epoch=60: 3.398605
    2019-05-25 15:59:19: Loss after num_examples_seen=57950 epoch=61: 3.397894
    2019-05-25 15:59:36: Loss after num_examples_seen=58900 epoch=62: 3.398161
    Setting learning rate to 0.000313
    2019-05-25 15:59:53: Loss after num_examples_seen=59850 epoch=63: 3.350161
    2019-05-25 16:00:11: Loss after num_examples_seen=60800 epoch=64: 3.343019
    2019-05-25 16:00:29: Loss after num_examples_seen=61750 epoch=65: 3.338262
    2019-05-25 16:00:46: Loss after num_examples_seen=62700 epoch=66: 3.335535
    2019-05-25 16:01:04: Loss after num_examples_seen=63650 epoch=67: 3.333591
    2019-05-25 16:01:22: Loss after num_examples_seen=64600 epoch=68: 3.331748
    2019-05-25 16:01:39: Loss after num_examples_seen=65550 epoch=69: 3.329749
    2019-05-25 16:01:56: Loss after num_examples_seen=66500 epoch=70: 3.327510
    2019-05-25 16:02:14: Loss after num_examples_seen=67450 epoch=71: 3.324991
    2019-05-25 16:02:31: Loss after num_examples_seen=68400 epoch=72: 3.322275
    2019-05-25 16:02:48: Loss after num_examples_seen=69350 epoch=73: 3.319588
    2019-05-25 16:03:06: Loss after num_examples_seen=70300 epoch=74: 3.317079
    2019-05-25 16:03:23: Loss after num_examples_seen=71250 epoch=75: 3.314758
    2019-05-25 16:03:40: Loss after num_examples_seen=72200 epoch=76: 3.312568
    2019-05-25 16:03:58: Loss after num_examples_seen=73150 epoch=77: 3.310483
    2019-05-25 16:04:15: Loss after num_examples_seen=74100 epoch=78: 3.308518
    2019-05-25 16:04:32: Loss after num_examples_seen=75050 epoch=79: 3.306699
    2019-05-25 16:04:49: Loss after num_examples_seen=76000 epoch=80: 3.305048
    2019-05-25 16:05:07: Loss after num_examples_seen=76950 epoch=81: 3.303578
    2019-05-25 16:05:24: Loss after num_examples_seen=77900 epoch=82: 3.302305
    2019-05-25 16:05:42: Loss after num_examples_seen=78850 epoch=83: 3.301242
    2019-05-25 16:05:59: Loss after num_examples_seen=79800 epoch=84: 3.300367
    2019-05-25 16:06:17: Loss after num_examples_seen=80750 epoch=85: 3.299641
    2019-05-25 16:06:34: Loss after num_examples_seen=81700 epoch=86: 3.299042
    2019-05-25 16:06:52: Loss after num_examples_seen=82650 epoch=87: 3.298558
    2019-05-25 16:07:09: Loss after num_examples_seen=83600 epoch=88: 3.298173
    2019-05-25 16:07:26: Loss after num_examples_seen=84550 epoch=89: 3.297873
    2019-05-25 16:07:43: Loss after num_examples_seen=85500 epoch=90: 3.297648
    2019-05-25 16:08:01: Loss after num_examples_seen=86450 epoch=91: 3.297494
    2019-05-25 16:08:18: Loss after num_examples_seen=87400 epoch=92: 3.297409
    2019-05-25 16:08:35: Loss after num_examples_seen=88350 epoch=93: 3.297387
    2019-05-25 16:08:52: Loss after num_examples_seen=89300 epoch=94: 3.297422
    Setting learning rate to 0.000156
    2019-05-25 16:09:10: Loss after num_examples_seen=90250 epoch=95: 3.253089
    2019-05-25 16:09:27: Loss after num_examples_seen=91200 epoch=96: 3.247362
    2019-05-25 16:09:44: Loss after num_examples_seen=92150 epoch=97: 3.243674
    2019-05-25 16:10:01: Loss after num_examples_seen=93100 epoch=98: 3.241192
    2019-05-25 16:10:18: Loss after num_examples_seen=94050 epoch=99: 3.239189



```python
def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        #print(next_word_probs[0][-1])
        #print(max(next_word_probs[0][-1]))
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[0][-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

num_sentences = 15
senten_min_length = 5

for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(model)
    print(" ".join(sent).title())
```

## Word Level Generated Titles

    Networked Inference And Quantitative Classification Using Ubiquitous And Machine Learning Algorithm
    Machine Learning Based Perspectives Applications
    Support Vector Machine For Library Machine Authentication In Data
    Ptsd Bayesian Resolution An Human Adaptation Using Machine Learning
    Data Mining Machine Learning On Deep Classification
    A Study Of Practice And Thinks In Classification Of Supervised Learning And Sites Using Machine Learning
    Interactive Machine Learning In Twitter
    Machine Learning Techniques For Reasoning Interpolation
    A Machine Learning Approach To Spark Page Of Permission Carbon Space And Application
    Machine Learning In Prediction Radios Svmlight Adhoc Umls
    Detecting Ensemble Methods And Machine Learning Techniques
    The Practical System In Evaluating Machine Learning
    Contentbased Coadaptive For Patternmatch Breakage Ngram Sense Using Machine Learning
    Network Exploration Quantum Science Classification Tool Selection And Interactive Machine Learning And Language Support
    Identification Genomescale Course From A Transversals Smallfootprint Machine Learning Approaches Sparse Server


# LSTM
 This is the [Beginners guide to text generation with LSTM](https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms) implementation


```python
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 


from tensorflow import set_random_seed
from numpy.random import seed
set_random_seed(2)
seed(1)

import os 

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
```


```python
corpus = cleaned.split("\n")
print(corpus[:10])
```

    ['scikitlearn machine learning in python', 'pattern recognition and machine learning', 'gaussian processes in machine learning', 'machine learning in automated text categorization', 'machine learning', 'thumbs up sentiment classification using machine learning techniques', 'ensemble methods in machine learning', 'c4 5 programs for machine learning', 'uci machine learning repository', 'data mining practical machine learning tools and techniques']



```python
tokenizer = Tokenizer()

def get_sequence_of_tokens(corpus):
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words

inp_sequences, total_words = get_sequence_of_tokens(corpus)
print(total_words)
inp_sequences[:10]
```

    1840
    [[161, 2],
     [161, 2, 1],
     [161, 2, 1, 7],
     [161, 2, 1, 7, 137],
     [162, 42],
     [162, 42, 4],
     [162, 42, 4, 2],
     [162, 42, 4, 2, 1],
     [138, 163],
     [138, 163, 7]]




```python
def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)
print(max_sequence_len)
```

    21



```python
def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 10, input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    
    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

lstm_model = create_model(max_sequence_len, total_words)
lstm_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, 20, 10)            18400     
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 100)               44400     
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 100)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 1840)              185840    
    =================================================================
    Total params: 248,640
    Trainable params: 248,640
    Non-trainable params: 0
    _________________________________________________________________



```python
lstm_model.fit(predictors, label, epochs=100, verbose=5)
```

    Epoch 1/100
    Epoch 2/100
    Epoch 3/100
    Epoch 4/100
    Epoch 5/100
    Epoch 6/100
    Epoch 7/100
    Epoch 8/100
    Epoch 9/100
    Epoch 10/100
    Epoch 11/100
    Epoch 12/100
    Epoch 13/100
    Epoch 14/100
    Epoch 15/100
    Epoch 16/100
    Epoch 17/100
    Epoch 18/100
    Epoch 19/100
    Epoch 20/100
    Epoch 21/100
    Epoch 22/100
    Epoch 23/100
    Epoch 24/100
    Epoch 25/100
    Epoch 26/100
    Epoch 27/100
    Epoch 28/100
    Epoch 29/100
    Epoch 30/100
    Epoch 31/100
    Epoch 32/100
    Epoch 33/100
    Epoch 34/100
    Epoch 35/100
    Epoch 36/100
    Epoch 37/100
    Epoch 38/100
    Epoch 39/100
    Epoch 40/100
    Epoch 41/100
    Epoch 42/100
    Epoch 43/100
    Epoch 44/100
    Epoch 45/100
    Epoch 46/100
    Epoch 47/100
    Epoch 48/100
    Epoch 49/100
    Epoch 50/100
    Epoch 51/100
    Epoch 52/100
    Epoch 53/100
    Epoch 54/100
    Epoch 55/100
    Epoch 56/100
    Epoch 57/100
    Epoch 58/100
    Epoch 59/100
    Epoch 60/100
    Epoch 61/100
    Epoch 62/100
    Epoch 63/100
    Epoch 64/100
    Epoch 65/100
    Epoch 66/100
    Epoch 67/100
    Epoch 68/100
    Epoch 69/100
    Epoch 70/100
    Epoch 71/100
    Epoch 72/100
    Epoch 73/100
    Epoch 74/100
    Epoch 75/100
    Epoch 76/100
    Epoch 77/100
    Epoch 78/100
    Epoch 79/100
    Epoch 80/100
    Epoch 81/100
    Epoch 82/100
    Epoch 83/100
    Epoch 84/100
    Epoch 85/100
    Epoch 86/100
    Epoch 87/100
    Epoch 88/100
    Epoch 89/100
    Epoch 90/100
    Epoch 91/100
    Epoch 92/100
    Epoch 93/100
    Epoch 94/100
    Epoch 95/100
    Epoch 96/100
    Epoch 97/100
    Epoch 98/100
    Epoch 99/100
    Epoch 100/100
    <keras.callbacks.History at 0x7fcb67351630>




```python
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()
```


```python
print (generate_text("", 5, lstm_model, max_sequence_len))
print (generate_text("euclidean", 4, lstm_model, max_sequence_len))
print (generate_text("generative", 5, lstm_model, max_sequence_len))
print (generate_text("ground breaking", 5, lstm_model, max_sequence_len))
print (generate_text("new", 4, lstm_model, max_sequence_len))
print (generate_text("understanding", 5, lstm_model, max_sequence_len))
print (generate_text("long short term memory", 6, lstm_model, max_sequence_len))
print (generate_text("LSTM", 6, lstm_model, max_sequence_len))
print (generate_text("a", 5, lstm_model, max_sequence_len))
print (generate_text("anomaly", 5, lstm_model, max_sequence_len))
print (generate_text("data", 7, lstm_model, max_sequence_len))
print (generate_text("designing", 7, lstm_model, max_sequence_len))
print (generate_text("reinforcement", 7, lstm_model, max_sequence_len))
```
## LSTM Generated Titles

     Inference Algorithms For Machine Learning
    Euclidean Inference Algorithms For Machine
    Generative Optimization And Machine Learning In
    Ground Breaking Inference Algorithms For Machine Learning
    New Forests A Unified Framework
    Understanding Machine Learning For Sledgehammer Microarray
    Long Short Term Memory Classifications With Machine Learning Classifiers For
    Lstm Network Benchmarking Machine Learning Algorithms For
    A Machine Learning Approach To Coreference
    Anomaly Detection Using Machine Learning Techniques
    Data Mining And Machine Learning In Cybersecurity Biologytowards
    Designing Inference Algorithms For Machine Learning Techniques For
    Reinforcement Learning Chapelle O Et Al Eds 2006
