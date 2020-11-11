# config

# training params
vocab_size      = 10000
num_epochs      = 100
embedding_dim   = 100
max_length      = 100
train_portion   = .8
batch_size      = 64

trunc_type      = 'post'
padding_type    = 'post'
oov_tok         = '<OOV>'
optimizer_fn    = 'adam'
loss_fn         = 'sparse_categorical_crossentropy'


# data params
tasks     = ['nullDefToken', 'npathLabel', 'methodName', 'operatorLabel', 'completionLabel']
models    = ['DummNN','LSTM','CNN']

# path to preformatted .csv files for the corresponding tasks
csvlist   = { 'nullDefToken'   : '/datasets/GLUECODE_NTKN_MAIN_joined.csv',
              'npathLabel'     : '/datasets/GLUECODE_NPTH_MAIN_joined.csv', 
              'methodName'     : '/datasets/GLUECODE_NAME_MAIN_joined.csv',
	          'operatorLabel'  : '/datasets/GLUECODE_OPER_MAIN_joined.csv',
              'completionLabel': '/datasets/GLUECODE_COMP_MAIN_joined.csv' }

# path to directory where the training models will be saved 
saved_models_path = '/saved_models' 

