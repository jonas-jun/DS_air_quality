from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from sklearn.metrics import classification_report
import pandas as pd

def custom_metric(matrix):
    bad = matrix.loc['truth_bad'][:-1]
    worst = matrix.loc['truth_worst'][:-1]
    t_bad = sum(bad)
    t_worst = sum(worst)
    right_bad = bad['pr_bad']+bad['pr_worst']
    right_worst = worst['pr_worst']+worst['pr_bad']
    return round(((right_bad + right_worst) / (t_bad + t_worst)), 4)

def get_scores(y_test, predicted):
    cf = pd.DataFrame(confusion_matrix(y_test, predicted))
    cf.index = ['truth_good', 'truth_moderate', 'truth_bad', 'truth_worst']
    cf.columns = ['pr_good', 'pr_moderate', 'pr_bad', 'pr_worst']
    cf['truth_total'] = cf['pr_good']+cf['pr_moderate']+cf['pr_bad']+cf['pr_worst']
    recall_bad = custom_metric(cf)
    acc = round(accuracy_score(y_test, predicted), 4)
    f1 = round(f1_score(y_test, predicted, average='macro'), 4)
    
    print(' >> recall_bad: {:.02f}%'.format(recall_bad*100))
    print(' >> total acc.: {:.02f}%'.format(acc*100))
    print(' >> total F1: {:.02f}'.format(f1*100))
    return recall_bad, acc, f1, cf


# for insert mode