import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn import metrics
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import ShuffleSplit, cross_val_score, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
from itables import init_notebook_mode
init_notebook_mode(all_interactive=True)
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import auc, make_scorer, f1_score
import pickle
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
import logging
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)

class Test:

    x_train, x_test, y_train, y_test = None, None, None, None
    model = None
    model_path = None
    loaded_model = None
    classes = ['Below 10%', 'Above 10%']
    params = {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 4, 'bootstrap': True, 'max_features': 'log2', 'max_leaf_nodes': 20, 'min_impurity_decrease': 0.0004209089091486704, 'class_weight': 'balanced'}

    def __init__(self):
        self.model_path = 'model.sav'
        pass

    def load_dataset(self):
        self.x_train = pd.read_csv("x_train.csv",header=None)
        self.x_test = pd.read_csv("x_test.csv",header=None)
        self.y_train = pd.read_csv("y_train.csv",header=None)
        self.y_test = pd.read_csv("y_test.csv",header=None)
        return self

    def train_model(self):
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(self.x_train,self.y_train)
        print("INFO: Model trained successfully")
        return self

    def load_model(self):
        self.loaded_model = pickle.load(open(self.model_path, 'rb'))
        print("INFO: model loaded successfully")
        return self
    
    @staticmethod
    def model_metrics(y_true, y_pred):
        print("--------------------------------")
        print("\nModel Evaluation:")
        print("accurancy: ",metrics.accuracy_score(y_true,y_pred))
        print("precision: ",metrics.precision_score(y_true,y_pred,average='macro'))
        print("recall: ",metrics.recall_score(y_true,y_pred,average='micro'))
        print("f-measure: ",metrics.f1_score(y_true,y_pred,average='weighted'))
        print("--------------------------------")

    @staticmethod
    def plot_conf_matrix(classes, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.xticks(np.arange(len(classes)) + 0.5, classes)
        plt.yticks(np.arange(len(classes)) + 0.5, classes, rotation=0)
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_probab, y_true):
        probabilities = y_probab
        # Compute ROC curve and ROC AUC for each class
        n_classes = probabilities.shape[1]
        roc_auc = []
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true, probabilities[:, i], pos_label=i)
            roc_auc.append(auc(fpr, tpr))
        # Compute average ROC AUC score
        average_roc_auc = sum(roc_auc) / n_classes
        # Print ROC AUC score for each class
        for i, auc_score in enumerate(roc_auc):
            print(f"Class {i} ROC AUC: {auc_score}")
        # Print average ROC AUC score
        print("Average ROC AUC:", average_roc_auc)
        # Plot ROC curve for each class
        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true, probabilities[:, i], pos_label=i)
            plt.plot(fpr, tpr, label=f'Class {i} (ROC AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Each Class')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

    
    @staticmethod
    def plot_precission_recal_curve(y_probab,y_pred,y_true):
        # predict probabilities
        lr_probs = y_probab
        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]
        # predict class values
        yhat = y_pred
        precision, recall, _ = precision_recall_curve(y_true, lr_probs)
        # plot the precision-recall curves
        no_skill = len(y_true[y_true==1]) / len(y_true)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.plot(recall, precision, marker='.', label='Model')
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()
    
    # below code is pulled from sklearn documentation
    @staticmethod
    def plot_learning_curve(
        estimator,
        X,
        y,
        ylim=None,
        n_jobs=None,
        scoring=None,
        train_sizes=np.linspace(0.1, 1.0, 5),
    ):
        _, axes = plt.subplots(3, 1, figsize=(10, 15))
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        title = 'learning curve'

        """
        Generate 3 plots: the test and training learning curve, the training
        samples vs fit times curve, the fit times vs score curve.

        Parameters
        ----------
        estimator : estimator instance
            An estimator instance implementing `fit` and `predict` methods which
            will be cloned for each validation.

        title : str
            Title for the chart.

        X : array-like of shape (n_samples, n_features)
            Training vector, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        y : array-like of shape (n_samples) or (n_samples, n_features)
            Target relative to ``X`` for classification or regression;
            None for unsupervised learning.

        axes : array-like of shape (3,), default=None
            Axes to use for plotting the curves.

        ylim : tuple of shape (2,), default=None
            Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

            - None, to use the default 5-fold cross-validation,
            - integer, to specify the number of folds.
            - :term:`CV splitter`,
            - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : int or None, default=None
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        scoring : str or callable, default=None
            A str (see model evaluation documentation) or
            a scorer callable object / function with signature
            ``scorer(estimator, X, y)``.

        train_sizes : array-like of shape (n_ticks,)
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the ``dtype`` is float, it is regarded
            as a fraction of the maximum size of the training set (that is
            determined by the selected validation method), i.e. it has to be within
            (0, 1]. Otherwise it is interpreted as absolute sizes of the training
            sets. Note that for classification the number of samples usually have
            to be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            estimator,
            X,
            y,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=train_sizes,
            return_times=True,
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        axes[0].fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        axes[0].plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )
        axes[0].plot(
            train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
        )
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, "o-")
        axes[1].fill_between(
            train_sizes,
            fit_times_mean - fit_times_std,
            fit_times_mean + fit_times_std,
            alpha=0.1,
        )
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        fit_time_argsort = fit_times_mean.argsort()
        fit_time_sorted = fit_times_mean[fit_time_argsort]
        test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
        test_scores_std_sorted = test_scores_std[fit_time_argsort]
        axes[2].grid()
        axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
        axes[2].fill_between(
            fit_time_sorted,
            test_scores_mean_sorted - test_scores_std_sorted,
            test_scores_mean_sorted + test_scores_std_sorted,
            alpha=0.1,
        )
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        plt.show()
    
    # model evaluation function (calls all evaluation methods)
    def test_model(self,load=False):
        if load:
            model = self.loaded_model
        else:
            model = self.model
        result = cross_val_score(model, self.x_train, self.y_train, cv=6, scoring='accuracy')
        print("Cross validatation Accuracy: ", np.mean(result))
        y_pred = model.predict(self.x_test)
        print(classification_report(self.y_test, y_pred,target_names=self.classes))
        y_pred_probab = model.predict_proba(self.x_test)
        self.model_metrics(self.y_test,y_pred)
        self.plot_conf_matrix(self.classes,self.y_test,y_pred)
        self.plot_roc_curve(y_pred_probab,self.y_test)
        self.plot_learning_curve(model,self.x_train,self.y_train,ylim = (0.0,1.1),n_jobs=4,scoring="accuracy")
        self.plot_precission_recal_curve(y_pred_probab,y_pred,self.y_test)
        return self
    
    def loaded_model_pipeline(self):
        self.load_dataset().load_model().test_model(load=True)

    def new_model_pipeline(self):
        self.load_dataset().train_model().test_model()
