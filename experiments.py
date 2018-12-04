def get_experiment(exp_name):
  if exp_name == "trancos":
    dataset_name="trancos"
    model_name="ResFCN"
    metric_name = "MAE"

  if exp_name == "shanghai":
     dataset_name="shanghai"
     model_name="FCN8"
     metric_name = "MAE"

  if exp_name == "pascal":
    dataset_name="pascal"
    model_name="ResFCN"
    metric_name = "mRMSE"


  print("Model: {} - Dataset: {} - Metric: {}".format(model_name, dataset_name,metric_name))
  return dataset_name, model_name, metric_name
