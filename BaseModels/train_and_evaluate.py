from config import *
from imports import *
from keras_models import *

tf.disable_v2_behavior()
warnings.filterwarnings('ignore')
csv.field_size_limit(sys.maxsize)

class CustomSaver(Callback):
    def on_epoch_end(self, epoch, logs={}):
        reports = saved_models_path + '/GlueCode_EvalSuite/models/model_' + exp_name +'/' + reports_dir + '/'
        Popen('mkdir -p {}'.format(reports), shell=True).communicate()
        self.model.save('{}/GlueCode_EvalSuite/models/model_{}/model_{}_EPOCH{}.hd5'.format(saved_models_path, exp_name, exp_name, (epoch+1)))
        global curr_epoch
        curr_epoch = epoch+1

def shuffle(train_articles, train_labels):
    combined = list(zip(train_articles, train_labels))
    random.shuffle(combined)
    train_articles, train_labels = zip(*combined)
    return train_articles, train_labels

def plot_graphs(history, string, savepath):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.savefig(savepath)

def getWinklerDistance(valid_padded, model, savepath, msg):
    epoch_cn = msg.split(" ")[-1]
    eval_csv = savepath + "_eval_epoch_"+ str(epoch_cn) +".csv"
    with open(eval_csv, 'w+') as evalfile:
        csvWriter = csv.writer(evalfile)
        csvWriter.writerow(["Predicted", "Original"])

        sumdist = 0
        avgwink = 0
        for i in range(0, valid_padded.shape[0]):
            item_padded   = valid_padded[i]
            item_padded.reshape((-1,1))
            item_padded   = np.array([item_padded])
            prediction    = model.predict(item_padded)   
            predictionlbl = prediction.argmax(axis=-1)[0]
            
            pred_label = lenc_mapping.inverse_transform([predictionlbl])
            orig_label = lenc_mapping.inverse_transform([valid_labels[i]])
            
            csvWriter.writerow([str(pred_label[0]), str(orig_label[0])])
            distance = jaro_winkler(str(pred_label[0]), str(orig_label[0]))
            sumdist  = sumdist + distance

        try:
            avgwink = (sumdist / valid_padded.shape[0])
        except ZeroDivisionError:
            print('Error: validation list may be empty with 0 rows!')
        return avgwink

def writeEvaluationScores(valid_padded, true_labels, predictions, eval_model, savepath, msg):
    with open(savepath, 'a') as wr:
        wr.write('\n----\n{}\n----\n'.format(msg))

        # if you want to bias your metric towards the most populated metric --- multi-class classification with imbalanced data
        wr.write("Precision (micro):    {}\n".format(precision_score(true_labels, predictions, average='micro')))
        wr.write("Recall (micro):       {}\n".format(recall_score(true_labels, predictions, average='micro')))
        wr.write("F1 score (micro):     {}\n\n".format(f1_score(true_labels, predictions, average='micro')))

        # if you want to bias your metric towards the least populated label
        wr.write("Precision (macro):    {}\n".format(precision_score(true_labels, predictions, average='macro')))
        wr.write("Recall (macro):       {}\n".format(recall_score(true_labels, predictions, average='macro')))
        wr.write("F1 score (macro):     {}\n\n".format(f1_score(true_labels, predictions, average='macro')))

        # if you want to unbias your metric towards any label whatsoever
        wr.write("Precision (weighted): {}\n".format(precision_score(true_labels, predictions, average='weighted')))
        wr.write("Recall (weighted):    {}\n".format(recall_score(true_labels, predictions, average='weighted')))
        wr.write("F1 score (weighted):  {}\n\n".format(f1_score(true_labels, predictions, average='weighted')))

        wr.write("Accuracy:             {}\n".format(accuracy_score(true_labels, predictions, normalize=True, sample_weight=None)))
        wr.write("Jaro-Winkler Sim.:    {}\n".format(getWinklerDistance(valid_padded, eval_model, savepath, msg)))


if __name__ == '__main__':

    for taskname in tasks:
        print('\n----\nRunning models on task: {}\n----\n'.format(taskname))
        lblcolname  = taskname
        txtcolname  = 'methodTokens'
        csvmainfile = csvlist.get(taskname,'Error')

        # data handling
        df = pd.read_csv(csvmainfile, header=0)
        articles = df[txtcolname].tolist()
        labels   = df[lblcolname].tolist() 

        train_size  = int(len(articles) * train_portion)
        train_articles = articles[0:train_size]
        train_labels   = labels[0:train_size]
        train_articles, train_labels = shuffle(train_articles, train_labels)

        valid_articles = articles[train_size:]
        valid_labels   = labels[train_size:]

        tokenizer = Tokenizer(split=' ', num_words=vocab_size, oov_token=oov_tok)
        tokenizer.fit_on_texts(articles)
        train_padded = pad_sequences(tokenizer.texts_to_sequences(train_articles), maxlen=max_length, padding=padding_type, truncating=trunc_type)
        valid_padded = pad_sequences(tokenizer.texts_to_sequences(valid_articles), maxlen=max_length, padding=padding_type, truncating=trunc_type)

        le = LabelEncoder()
        lenc_mapping = le.fit(labels)
        train_labels = le.transform(train_labels)
        valid_labels = le.transform(valid_labels)

        # model training & evaluation
        for modelname in models:

            #######################################################################    
            exp_name    = csvmainfile.split('/')[-1].split('.')[0]  + '_'+ taskname +'_'+ modelname
            experiment  = Experiment(api_key="<removed for privacy>", project_name=exp_name, workspace="<removed to ensure double-blind review standards>", auto_param_logging=True)

            timestamp   = time.localtime()
            reports_dir = 'reports-{}.{}.{}-{}.{}.{}'.format(timestamp.tm_hour, timestamp.tm_min, timestamp.tm_sec, timestamp.tm_mday, timestamp.tm_mon, timestamp.tm_year)
            #######################################################################

            # model training
            n_epochs = num_epochs
            n_labels = len(le.classes_) +1
            md_saver = CustomSaver()
            wt_class = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
            curr_epoch = 0

            model = create_model(modelname, n_labels)
            model.compile(loss=loss_fn, optimizer=optimizer_fn, metrics=['accuracy'])

            try:
                history = model.fit(train_padded, train_labels, callbacks=[md_saver], epochs=n_epochs, validation_split=0.2, class_weight=wt_class, batch_size=batch_size, verbose=1)
            except KeyboardInterrupt:
                print('\n\nEarly Stopping encountered ...\nStarting Evaluation now ...\n')


            # model evaluation
            eval_epoch = input('Enter the epoch number to load weights from: ')
            eval_model = create_model(modelname, n_labels)
            eval_model.load_weights("{}/GlueCode_EvalSuite/models/model_{}/model_{}_EPOCH{}.hd5".format(saved_models_path, exp_name, exp_name, eval_epoch))

            pred_orig_model = model.predict_classes(valid_padded, verbose=1)
            pred_eval_model = eval_model.predict_classes(valid_padded, verbose=1)
            true_labels     = valid_labels

            savepath = '{}/GlueCode_EvalSuite/models/model_{}/{}/results_{}_{}.txt'.format(saved_models_path, exp_name, reports_dir, taskname, modelname)
            writeEvaluationScores(valid_padded, true_labels, pred_orig_model, model, savepath, "Results for last-updated _MODEL >>> Epoch: {}".format(curr_epoch))
            writeEvaluationScores(valid_padded, true_labels, pred_eval_model, eval_model, savepath, "Results for user-selected _MODEL >>> Epoch: {}".format(eval_epoch))