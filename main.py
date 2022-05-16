from model import Model

mode
while True:
    d.apply_labels()
    d.encode_labels()

    m = model.predictive_model(d.df_filtered)
    m.evaluate()
    d.predict_unlabeled(m)
    d.get_labels(n=100)

