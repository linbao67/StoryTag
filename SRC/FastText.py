import fasttext
import pandas as pd
import os

def get_tagsystem():
    df_tagsystem = pd.read_csv("../Files/tag_system.csv")
    df_tagsystem['Tag'] = df_tagsystem.applymap(lambda x: '__label__' + str(x))['TagID']
    return df_tagsystem



def training_model():

    model = fasttext.train_supervised(input="../Files/training_data.txt", lr=0.5, epoch=25,
                                      wordNgrams=2, bucket=200000, dim=50, loss='ova')
    model.save_model("../Model/model_userstory.bin")
    #result = model.test("C:/Users/lbao009/Documents/testing_data.txt")


if __name__ == "__main__":
    model_file = '../Model/model_userstory.bin'
    if not os.path.exists(model_file):
        training_model()
    df_tags = get_tagsystem()

    df_predict = pd.read_csv("../Files/testing_data.csv")
    texts = df_predict["Story"].str.strip().str.replace('\n', '').tolist()

    classifier = fasttext.load_model(model_file)

    labels = classifier.predict(texts)
    df_predict['Tag'] = pd.Series(labels[0]).apply(lambda x : x[0])


    df_result = pd.merge(df_predict,df_tags, on='Tag',how='left')
    df_result.to_csv('../Files/result.csv',columns=['RecordID','Summary','Story','Acceptance Criteria', 'Capability', 'Sub-Capability','Epic','Tag'],index=False)


