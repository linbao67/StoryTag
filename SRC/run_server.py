#!/usr/bin/env python3

import fasttext
import pandas as pd
import FastText

import numpy as np
from flask import Flask, render_template, request



app = Flask(__name__)



@app.route('/', methods=('GET', 'POST'))
def predict():
    prompt = ""
    texts = []
    df_category = pd.read_csv("C:/Users/lbao009/Documents/category.csv")
    print(df_category)
    if request.method == 'POST':
        prompt = request.form['prompt'].strip()

        classifier = fasttext.load_model("model_userstory.bin")


        if prompt:
            label = classifier.predict(prompt)
            Label_Id = int(label[0][0][9:])
            df_result = df_category[df_category['LabelID'] == Label_Id]
            result = df_result['Capability'].iloc[0]+'/'+df_result['Sub-Capability'].iloc[0]

        else:
            result = ""
        texts.append(result)


    return render_template("page.html", prompt=prompt, texts=texts)



if __name__ == "__main__":
    print(("* Loading model and Flask starting server..."
           "please wait until server has fully started"))
    # Run app
    app.run(host="localhost", port=5000)