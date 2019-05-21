# Surname Generating using PyTorch

#### The surnames dataset, a collection of 10,000 surnames from 18 different nationalities collected by the authors from different name sources on the internet. The top three classes account for more than 60% of the data: 27% are English, 21% are Russian, and 14% are Arabic. The remaining 15 nationalities have decreasing frequency.

#### This repo construct a language model (also un-condition and condition) for generating surname, it also contain data preprocess pipeline for handel raw data.

* surname_dataset.py define classes for prepare data (convert from csv to vector, vocab, create batch, ...)
* utils.py some util functions
* generation_model.py define un-condition language model (generating surname randomly) and condition language model (generating surname with national info)
* conditional_experiment.py train and generate samples using condition language model
* unconditional_experiment.py train and generate samples using un-condition language model

Some result with condition language model:

```text
Vietnamese: 
  - Tuan
  - Lua
  - Pian
  - Leun
  - Tury
  - Anon
  - Vuon
  - Vuong
  - Pha
  - Lan
Chinese: 
   - Shang
   - Lui
   - Lang
   - Lai
   - Zhi
   - Lang
   - Lan
   - Sui
   - Lan
   - Lai
English: 
   - Mcharring
   - Paone
   - Denner
   - Leas
   - Raddin
   - Shinker
   - Towroum
   - Ander
   - Tomes
   - Roder
```

Happy coding!!!!!

