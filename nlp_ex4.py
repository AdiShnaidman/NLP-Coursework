

###################################################
# Exercise 4 - Natural Language Processing 67658  #
###################################################

import numpy as np
import matplotlib.pyplot as plt
import torch

# subset of categories that we will use
category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                 }

def get_data(categories=None, portion=1.):
    """
    Get data for given categories and portion
    :param portion: portion of the data to use
    :return:
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    # train
    train_len = int(portion*len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test


# Q1
def linear_classification(portion=1.):
    """
    Perform linear classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    tf = TfidfVectorizer(stop_words='english', max_features=1000)
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    tfidf_x_train,tfidf_x_test = tf.fit_transform(x_train),tf.transform(x_test)
    preds = LogisticRegression().fit(tfidf_x_train, y_train).predict(tfidf_x_test)
    return accuracy_score(y_test,preds)


# Q2
def transformer_classification(portion=1.):
    """
    Transformer fine-tuning.
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    import torch

    class Dataset(torch.utils.data.Dataset):
        """
        Dataset object
        """
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    from datasets import load_metric
    metric = load_metric("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    from transformers import Trainer, TrainingArguments
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base', cache_dir=None)
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base',
                                                               cache_dir=None,
                                                               num_labels=len(category_dict),
                                                               problem_type="single_label_classification")

    model.to(get_available_device())
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    training_args = TrainingArguments(
        output_dir="Q2",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
    )
    train_data = Dataset(tokenizer(x_train,padding="longest",truncation=True),y_train)
    test_data = Dataset(tokenizer(x_test,padding="longest",truncation=True),y_test)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer.evaluate()  # calc accuracy


    # Add your code here
    # see https://huggingface.co/docs/transformers/v4.25.1/en/quicktour#trainer-a-pytorch-optimized-training-loop
    # Use the DataSet object defined above. No need for a DataCollator

def get_available_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Q3
def zeroshot_classification(portion=1.):
    """
    Perform zero-shot classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from transformers import pipeline
    from sklearn.metrics import accuracy_score
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    clf = pipeline("zero-shot-classification", model='cross-encoder/nli-MiniLM2-L6-H768'
                   ,device=get_available_device())
    candidate_labels = list(category_dict.values())

    preds = clf(x_test, candidate_labels=candidate_labels)
    preds_labels = [candidate_labels.index(p["labels"][np.argmax(p["scores"])]) for p in preds]
    return accuracy_score(y_test,preds_labels)


    # Add your code here
    # see https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline

def plot_accuracies(data_points):
    portion,acc = zip(*data_points)
    plt.scatter(portion, acc, marker='x')  # Use 'o' for circle markers
    for i, txt in enumerate(acc):
        plt.annotate(f"{txt:.3f}", (portion[i], acc[i]), textcoords="offset points", xytext=(0,5), ha='center')

    plt.title("Accuracies per Portion")
    plt.xlabel("Portion")
    plt.ylabel("Accuracy")
    plt.show()


if __name__ == "__main__":
    portions = [0.1, 0.5, 1.]
    # Q1
    res = []
    print("Logistic regression results:")
    for p in portions:
        print(f"Portion: {p}")
        acc = linear_classification(p)
        res.append((p,acc))
        print(acc)
    plot_accuracies(res)

    # Q2
    res = []
    print("\nFinetuning results:")
    for p in portions:
        print(f"Portion: {p}")
        acc = transformer_classification(portion=p)
        res.append((p,acc["eval_accuracy"]))
        print(acc)
    plot_accuracies(res)

    # Q3
    print("\nZero-shot result:")
    print(zeroshot_classification())