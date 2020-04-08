# Tosin Dairo
# /Module for Binary Classification


from classification import ClassificationModel
import pandas as pd
import logging
import sklearn
# import fire


eval_df = pd.read_csv('/Volumes/Loopdisk/Bi_Transformer/data/dev.csv', sep='\t')
eval_df = eval_df[['0', '1']]
eval_df['0'] = eval_df['0'].astype(str)


model = ClassificationModel("bert", "outputs", use_cuda=False)
result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)


def classify(inport):
    predictions, raw_outputs = model.predict([inport])
    print("For {} accuracy, classification belongs to".format(result['acc']*10))
    if predictions == 1:
        outport = "\nClass: Eligibility Criteria"
    else:
        outport = "\nClass: Others"
    return outport

    
if __name__ == "__main__":
    inport = input('')
    outport = classify(inport)
    print(outport)
