import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
import string
import seaborn as sns
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from tkinter import ttk
import tkinter as tk

class ToxicCommentAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.num_rows = 0
        self.model = None
        self.vectorizer = None

    def load_and_clean_data(self):
        print('Loading and cleaning data...')
        self.data = pd.read_csv(self.data_path)
        alphanumeric = lambda x: re.sub(r'\w*\d\w*', ' ', x)
        punc_lower = lambda x: re.sub(f"[{re.escape(string.punctuation)}]", ' ', x.lower())
        remove_n = lambda x: re.sub("\n", " ", x)
        remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]', r' ', x)
        self.data['comment_text'] = (self.data['comment_text']
                                     .map(alphanumeric)
                                     .map(punc_lower)
                                     .map(remove_n)
                                     .map(remove_non_ascii))
        self.num_rows = len(self.data)
        print(f"Number of rows in data: {self.num_rows}")

    def plot_toxic_comment_distribution(self):
        print('Plotting toxic comment distribution...')
        tox_per = self.data['toxic'].sum() / self.num_rows * 100
        sevTox_per = self.data['severe_toxic'].sum() / self.num_rows * 100
        obs_per = self.data['obscene'].sum() / self.num_rows * 100
        thr_per = self.data['threat'].sum() / self.num_rows * 100
        ins_per = self.data['insult'].sum() / self.num_rows * 100
        hate_per = self.data['identity_hate'].sum() / self.num_rows * 100

        ind = np.arange(6)
        plt.barh(ind, [tox_per, obs_per, ins_per, sevTox_per, hate_per, thr_per])
        plt.xlabel('Percentage (%)', size=20)
        plt.title('% of Toxic Comments', size=22)
        plt.xticks(np.arange(0, 30, 5), size=20)
        plt.yticks(ind, ['Toxic', 'Obscene', 'Insult', 'Severe Toxic', 'Identity Hate', 'Threat'], size=15)
        plt.show()

    def generate_wordcloud(self, label):
        print(f"Generating wordcloud for {label}...")
        subset = self.data[self.data[label] == 1]
        text = subset.comment_text.values
        wc = WordCloud(background_color="black", max_words=4000).generate(" ".join(text))
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.title(f"Words Frequenced in {label}", fontsize=20)
        plt.imshow(wc.recolor(colormap='gist_earth', random_state=244), alpha=0.98)
        plt.show()

    def prepare_data(self, label, ratio=0.2):
        print(f"Preparing data for {label}...")
        data_positive = self.data[self.data[label] == 1]
        data_negative = self.data[self.data[label] == 0]
        sample_size = min(len(data_positive), int(len(data_negative) * ratio))
        balanced_data = pd.concat([data_positive.iloc[:sample_size], data_negative.iloc[:sample_size * 4]], axis=0)
        return balanced_data

    def train_and_save_logistic_regression(self, data, label, model_filename):
        print(f"Training logistic regression for {label}...")
        X = data['comment_text']
        y = data[label]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
        X_vect = self.vectorizer.fit_transform(X)
        self.model = LogisticRegression()
        self.model.fit(X_vect, y)
        print(f"Model trained for {label}. Saving to {model_filename}...")
        with open(model_filename, 'wb') as f:
            pickle.dump({'model': self.model, 'vectorizer': self.vectorizer}, f)

    def load_model(self, model_filename):
        print(f"Loading model from {model_filename}...")
        with open(model_filename, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
        print("Model loaded successfully.")

    def predict_comment_toxicity(self, comment):
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model and vectorizer are not loaded. Train or load a model first.")
        comment_vectorized = self.vectorizer.transform([comment])
        prediction = self.model.predict(comment_vectorized)[0]
        result = "Toxic" if prediction == 1 else "Not Toxic"
        #print(f"Comment: {comment}\nPrediction: {result}")
        return result


def main():
    analyzer = ToxicCommentAnalyzer('train.csv')
    analyzer.load_and_clean_data()

    # Plot distributions and wordclouds
    analyzer.plot_toxic_comment_distribution()
    analyzer.generate_wordcloud('toxic')
    analyzer.generate_wordcloud('identity_hate')

    # Prepare data and train the logistic regression model
    toxic_data = analyzer.prepare_data('toxic')
    model_filename = 'logistic_toxicity_model.pkl'
    analyzer.train_and_save_logistic_regression(toxic_data, 'toxic', model_filename)

    # Predict a sample comment
    analyzer.load_model(model_filename)
    window = tk.Tk()
    window.configure(bg='#443751')
    window.title('Toxic Comment Analyzer')
    window.geometry('500x400')
    foont=('Arial',20)
    fonnt=('Arial',12)
    lbl=tk.Label(window,text='Toxic Comment Analyzer',font=foont,fg='#F0E4E4',bg='#443751').pack()
    lbl=tk.Label(window,text='Enter your comment below',font=fonnt,fg='#F0E4E4',bg='#443751').pack()

    foont=('Arial',30)
    ent=tk.Entry(window,)
    ent.pack()
    def get_entry():
        r=ent.get()
        c=analyzer.predict_comment_toxicity(r)
        lbl=tk.Label(window,text='The given comment is:'+c,font=fonnt,fg='#F0E4E4',bg='#443751',)
        lbl.place(x=13-,y=170)
            
    
    but=tk.Button(window,text='Enter',command=get_entry)
    but.pack()
    
    window.mainloop()
    

    ###################################################################
    
    


if __name__ == "__main__":
    main()
