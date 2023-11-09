#!/usr/bin/env python
# coding: utf-8

# Предсказание следующих 10 MCC-кодов на основании историй транзакций каждого клиента с помощью HMM

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


df_train = pd.read_csv('df_train.csv', sep=';')
df_test = pd.read_csv('df_test.csv', sep=';')
df_train['Data'] = df_train.Data.apply(lambda s: list(map(str, s.split(','))))
df_train['Target'] = df_train.Target.apply(lambda s: list(map(str, s.split(','))))
df_test['Data'] = df_test.Data.apply(lambda s: list(map(str, s.split(','))))


# In[ ]:


limit_of_predictor_len = 7
limit_of_prediction_len = 10


# In[ ]:


# лексикон по всем клиентам

lexicon_general = {}

for i, row in df_train.iterrows():
    codeline = row['Data']
    for predictor_len in range(1, min(len(codeline), limit_of_predictor_len) + 1):
        for i in range(len(codeline) - predictor_len):
            predictor = tuple(codeline[i:i+predictor_len])
            prediction = codeline[i+predictor_len:i+predictor_len+limit_of_prediction_len]
            if predictor not in lexicon_general:
                places = {i+1: {prediction[i]: 1} if i < len(prediction) else {} for i in range(limit_of_prediction_len)}
                lexicon_general.update({predictor: places})
            else:
                for i in range(len(prediction)):
                    lexicon_general[predictor][i+1][prediction[i]] = lexicon_general[predictor][i+1].get(prediction[i], 0) + 1

for key1, val1 in lexicon_general.items():
    for key2, val2 in val1.items():
        val2 = {key3: val3/sum(val2.values()) for key3, val3 in val2.items()}
        val1[key2] = val2


# In[ ]:


# лексикон для каждого клиента

def get_lexicon(sequence):
    
    lexicon = {}

    for predictor_len in range(1, min(len(sequence), limit_of_predictor_len) + 1):
        for i in range(len(sequence) - predictor_len):
            predictor = tuple(sequence[i:i+predictor_len])
            prediction = sequence[i+predictor_len:i+predictor_len+limit_of_prediction_len]
            if predictor not in lexicon:
                places = {i+1: {prediction[i]: 1} if i < len(prediction) else {} for i in range(limit_of_prediction_len)}
                lexicon.update({predictor: places})
            else:
                for i in range(len(prediction)):
                    lexicon[predictor][i+1][prediction[i]] = lexicon[predictor][i+1].get(prediction[i], 0) + 1

    for key1, val1 in lexicon.items():
        for key2, val2 in val1.items():
            N = sum(val2.values())
            val2 = {key3: val3/N for key3, val3 in val2.items()}
            val1[key2] = val2

    return lexicon


# In[ ]:


# частота MCC-кода для каждого клиента

def get_frequencies(sequence):
    
    frequencies = {}
    
    for code in sequence:
        frequencies[code] = frequencies.get(code, 0) + 1
    N = sum(frequencies.values())
    frequencies = {code: number/N for code, number in frequencies.items()}
    
    return frequencies


# In[ ]:


def predict_sequence(sequence, predictor_length, window_length, prediction_length=10):

    lexicon = get_lexicon(sequence)
    frequencies = get_frequencies(sequence)

    count = prediction_length

    while count > 0:

        original_predictor = tuple(sequence[-predictor_length:])
        window = [0] * min(window_length, count)

        for i in range(min(window_length, count)):

            match = False

            predictor = original_predictor
            while predictor and not match:
                if lexicon.get(predictor):
                    if lexicon[predictor][i+1]:
                        options = lexicon[predictor][i+1]
                        window[i] = np.random.choice(list(options.keys()), p=list(options.values()))
                        match = True
                    else:
                        predictor = predictor[1:]
                else:
                    predictor = predictor[1:]

            if not match:
                predictor = original_predictor
                while predictor and not match:
                    if lexicon_general.get(predictor):
                        if lexicon_general[predictor][i+1]:
                            options = lexicon_general[predictor][i+1]
                            window[i] = np.random.choice(list(options.keys()), p=list(options.values()))
                            match = True
                        else:
                            predictor = predictor[1:]
                    else:
                        predictor = predictor[1:]

            if not match:
                window[i] = np.random.choice(list(frequencies.keys()), p=list(frequencies.values()))

            count -= 1

        sequence.extend(window)

    return sequence[-prediction_length:]


# In[ ]:


df_test['Predicted'] = df_test.apply(lambda x: predict_sequence(x['Data'], predictor_length=7, window_length=10), axis=1)


# In[ ]:


df = df_test[['Id', 'Predicted']]
df['Predicted'] = df.Predicted.astype(str).str.replace(',', '')
df['Predicted'] = df['Predicted'].str.replace('\'', '')
df.to_csv('df.csv', index=False)

