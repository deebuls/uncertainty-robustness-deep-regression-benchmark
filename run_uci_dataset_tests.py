import argparse
import numpy as np
import pandas as pd
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from scipy import stats
import re

import edl
import data_loader
import trainers
import models
from models.toy.h_params import h_params


parser = argparse.ArgumentParser()
parser.add_argument("--num-trials", default=20, type=int,
                    help="Number of trials to repreat training for \
                    statistically significant results.")
parser.add_argument("--num-epochs", default=40, type=int)
parser.add_argument('--datasets', nargs='+', default=["yacht"],
                    choices=['boston', 'concrete', 'energy-efficiency',
                            'kin8nm', 'naval', 'power-plant', 'protein',
                            'wine', 'yacht'])
parser.add_argument("--load-pkl", action='store_true',
                    help="Load predictions for a cached pickle file or \
                        recompute from scratch by feeding the data through \
                        trained models")
args = parser.parse_args()

output_dir = "figs/uci"
"""" ================================================"""
#training_schemes = [trainers.Likelihood, trainers.Likelihood, trainers.Evidential, trainers.Ensemble]
#method_names = ["Gaussian", "Laplace", "Evidential", "Ensemble"]
training_schemes = [trainers.Likelihood]
method_names = ["Gaussian"]
datasets = args.datasets
num_trials = args.num_trials
num_epochs = args.num_epochs
dev = "/cpu:0" # for small datasets/models cpu is faster than gpu
"""" ================================================"""
def predict(method_name, model, x):
    if method_name == "Dropout":
        preds = tf.stack([model(x, training=True) for _ in range(n_samples)], axis=0) #forward pass
        mu, var = tf.nn.moments(preds, axes=0)
        return mu, tf.sqrt(var)

    elif method_name == "Evidential":
        outputs = model(x, training=False)
        mu, v, alpha, beta = tf.split(outputs, 4, axis=-1)
        sigma = tf.sqrt(beta/(v*(alpha + 1e-6 - 1)))
        print ("Evidential predict shape : ", mu.shape, sigma.shape)
        return mu, sigma

    elif method_name == "Ensemble":
        preds = tf.stack([f(x) for f in model], axis=0)
        y, sigmas = tf.split(preds, 2, axis=-1)
        mu = tf.reduce_mean(y, axis=0)
        sigma = tf.math.reduce_std(sigmas, axis=0)
        var = tf.reduce_mean(sigmas**2 + tf.square(y), axis=0) - tf.square(mu)
        #preds = tf.stack([f(x) for f in model], axis=0)
        print ("Ensemble preds shape ", preds.shape)
        #mu, var = tf.nn.moments(preds, 0)
        print ("Ensemble predict shape : ", mu.shape, var.shape)
        return mu, tf.sqrt(var)

    elif method_name == "Gaussian":
        outputs = model(x, training=False)
        mu,  var = tf.split(outputs, 2, axis=-1)
        return mu, tf.sqrt(var)

    elif method_name == "Laplace":
        outputs = model(x, training=False)
        mu,  b = tf.split(outputs, 2, axis=-1)
        print ("Laplace predict shape : ", mu.shape, b.shape)
        return mu, b
        
    

    else:
        raise ValueError("Unknown model")


def get_prediction_summary(dataset, method_name, model, x_batch, y_batch):
    #First collect predictions
    mu_batch, sigma_batch = predict(method_name, model, x_batch)
    #mu_batch = np.clip(mu_batch, 0, 1)
    mu_batch = np.squeeze(mu_batch.numpy())
    sigma_batch = np.squeeze(sigma_batch.numpy())
    y_batch = np.squeeze(y_batch)

    print (" Prediction summary : ", method_name, mu_batch.shape, sigma_batch.shape, y_batch.shape)
    ### Save the predictions to some dataframes for later analysis
    summary = [{"Dataset": dataset, "Method": method_name,"Target": y, "Mu": mu, "Sigma": sigma}
        for y,mu,sigma in zip(y_batch, mu_batch, sigma_batch)]
    return summary



"""" ================================================"""
def compute_predictions():
    RMSE = np.zeros((len(datasets), len(training_schemes), num_trials))
    NLL = np.zeros((len(datasets), len(training_schemes), num_trials))
    df_pred_uci = pd.DataFrame(columns=["Dataset", "Method", "Target", "Mu", "Sigma"] )
    for di, dataset in enumerate(datasets):
        for ti, trainer_obj in enumerate(training_schemes):
            for n in range(num_trials):
                (x_train, y_train), (x_test, y_test), y_scale = data_loader.load_dataset(dataset, return_as_tensor=False)
                batch_size = h_params[dataset]["batch_size"]
                num_iterations = num_epochs * x_train.shape[0]//batch_size
                print ("Num of iterations :", num_iterations)
                done = False
                while not done:
                    with tf.device(dev):
                        model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
                        model, opts = model_generator.create(input_shape=x_train.shape[1:])
                        if method_names[ti] == "Laplace": #training scheme is likelihood; as its 2nd in list
                            print ("Trainienr lalpace likelihood")
                            trainer = trainer_obj(model, opts, "laplace", dataset, learning_rate=h_params[dataset]["learning_rate"])
                        elif method_names[ti] == "Gaussian":
                            print ("Trainienr Gaussian likelihood")
                            trainer = trainer_obj(model, opts, "gaussian", dataset, learning_rate=h_params[dataset]["learning_rate"])
                        else:
                            trainer = trainer_obj(model, opts, dataset, learning_rate=h_params[dataset]["learning_rate"])
                        model, rmse, nll = trainer.train(x_train, y_train, x_test, y_test, y_scale, iters=num_iterations, batch_size=batch_size, verbose=True)
                       
                        #Compute on validation data and save predictions
                        summary_to_add = get_prediction_summary(
                             dataset, method_names[ti], model, x_test, y_test)
                        df_pred_uci = df_pred_uci.append(summary_to_add, ignore_index=True)
    
                        del model
                        tf.keras.backend.clear_session()
                        done = False if np.isinf(nll) or np.isnan(nll) else True
                print("saving {} {}".format(rmse, nll))
                RMSE[di, ti, n] = rmse
                NLL[di, ti, n] = nll
    
    RESULTS = np.hstack((RMSE, NLL))
    mu = RESULTS.mean(axis=-1)
    error = np.std(RESULTS, axis=-1)
    
    print("==========================")
    print("[{}]: {} pm {}".format(dataset, mu, error))
    print("==========================")
    
    print("TRAINERS: {}\nDATASETS: {}".format([trainer.__name__ for trainer in training_schemes], datasets))
    print("MEAN: \n{}".format(mu))
    print("ERROR: \n{}".format(error))

    return df_pred_uci

def gen_paper_plots(df_pred_uci):
    cm = 1/2.54  # centimeters in inches
    sns.set()
    sns.set_style("white")
    sns.set_style("ticks")
    sns.despine()
    sns.set_context("paper")
    sns.color_palette("tab10")

    print ("Generating point plots") 
    print (df_pred_uci.head())
    dataset_names = df_pred_uci["Dataset"].unique()
    new_dataset_names = [label.replace('-', '-\n') for label in dataset_names]


    #================================================
    print (f"Generating Interval Score")
    df_pred_uci["lower"] = df_pred_uci["Mu"] - 2*df_pred_uci["Sigma"]
    df_pred_uci["lower"].mask(df_pred_uci["Method"]=="Laplace", df_pred_uci["Mu"] - 3*df_pred_uci["Sigma"], inplace=True)
    df_pred_uci["upper"] = df_pred_uci["Mu"] + 2*df_pred_uci["Sigma"]
    df_pred_uci["upper"].mask(df_pred_uci["Method"]=="Laplace", df_pred_uci["Mu"] + 3*df_pred_uci["Sigma"], inplace=True)
    
    df_pred_uci["Interval Score"] = df_pred_uci["upper"] - df_pred_uci["lower"] \
     + (2/0.95)*(df_pred_uci["lower"]-df_pred_uci["Target"])*(df_pred_uci["Target"]<df_pred_uci["lower"]) \
     + (2/0.95)*(df_pred_uci["Target"] - df_pred_uci["upper"])*(df_pred_uci["Target"]>df_pred_uci["upper"])
    
    fig = plt.figure(figsize=(14.2*cm, 14.2*cm/2.0))
    gs = fig.add_gridspec(1, 9)
    for i,dataset_name in enumerate(dataset_names):
        ax = fig.add_subplot(gs[0,i])
         
        g = sns.pointplot(x="Method", y="Interval Score", hue="Method", 
                          markers=["o", "x", "*", "D"],
                          linestyles=["-","--","-.",":"],
                           data=df_pred_uci[df_pred_uci.Dataset == dataset_name], legend=False)
        #g.set(yscale="log")
        g.get_legend().remove()
        g.set_xlabel(new_dataset_names[i], fontsize='xx-small')
        g.axes.get_xaxis().set_ticks([])
        g.axes.get_yaxis().set_ticks([])
        if i != 0:
            g.axes.get_yaxis().get_label().set_visible(False)
            g.axes.get_yaxis().set_ticks([])
    handles, labels = g.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.55, 0.92), loc='lower center', ncol=5, fancybox=True, shadow=True)
    plt.savefig(os.path.join(output_dir, f"Interval_score_point_uci.pdf"), bbox_inches='tight')
    plt.show()
    plt.clf()

    #RMSE
    df_pred_uci["RMSE"] = (df_pred_uci["Mu"] - df_pred_uci["Target"])**2
    
    fig = plt.figure(figsize=(14.2*cm, 14.2*cm/2.0))
    gs = fig.add_gridspec(1, 9)
    for i,dataset_name in enumerate(dataset_names):
        ax = fig.add_subplot(gs[0,i])
         
        g = sns.pointplot(x="Method", y="RMSE", hue="Method", 
                          markers=["o", "x", "*", "D"],
                          linestyles=["-","--","-.",":"],
                           data=df_pred_uci[df_pred_uci.Dataset == dataset_name], legend=False)
        g.get_legend().remove()
        g.set_xlabel(new_dataset_names[i], fontsize='xx-small')
        g.axes.get_xaxis().set_ticks([])
        g.axes.get_yaxis().set_ticks([])
        if i != 0:
            g.axes.get_yaxis().get_label().set_visible(False)
        
    handles, labels = g.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.55, 0.92), loc='lower center', ncol=5, fancybox=True, shadow=True)
    plt.savefig(os.path.join(output_dir, "RMSE_point_uci.pdf"), bbox_inches='tight')
    plt.show()
    
def simplified_paper_plot(df_pred_uci):
    cm = 1/2.54  # centimeters in inches
    sns.set()
    sns.set_style("white")
    sns.set_style("ticks")
    sns.despine()
    sns.set_context("paper")
    sns.color_palette("tab10")

    print ("Generating point plots") 
    print (df_pred_uci.head())
    dataset_names = df_pred_uci["Dataset"].unique()
    new_dataset_names = [label.replace('-', '-\n') for label in dataset_names]
    print (new_dataset_names, dataset_names)


    #================================================
    print (f"Generating Interval Score")
    df_pred_uci["lower"] = df_pred_uci["Mu"] - 2*df_pred_uci["Sigma"]
    df_pred_uci["lower"].mask(df_pred_uci["Method"]=="Laplace", df_pred_uci["Mu"] - 3*df_pred_uci["Sigma"], inplace=True)
    df_pred_uci["upper"] = df_pred_uci["Mu"] + 2*df_pred_uci["Sigma"]
    df_pred_uci["upper"].mask(df_pred_uci["Method"]=="Laplace", df_pred_uci["Mu"] + 3*df_pred_uci["Sigma"], inplace=True)
    
    df_pred_uci["Interval Score"] = df_pred_uci["upper"] - df_pred_uci["lower"] \
     + (2/0.95)*(df_pred_uci["lower"]-df_pred_uci["Target"])*(df_pred_uci["Target"]<df_pred_uci["lower"]) \
     + (2/0.95)*(df_pred_uci["Target"] - df_pred_uci["upper"])*(df_pred_uci["Target"]>df_pred_uci["upper"])
    
    fig = plt.figure(figsize=(14.2*cm, 14.2*cm/2.0))
    gs = fig.add_gridspec(2, 1)
    ax = fig.add_subplot(gs[0,0])
    #RMSE
    df_pred_uci["RMSE"] = (df_pred_uci["Mu"] - df_pred_uci["Target"])**2
    
    g = sns.pointplot(x="Dataset", y="RMSE", hue="Method", 
                      markers=["*","D","o", "x" ],
                      linestyles=["-","--","-.",":"],
                      dodge=0.45,
                      join=False, ci='sd',palette=["C2", "C3", "C0", "C1"],
                       data=df_pred_uci, legend=False)
    g.set_xticklabels([])
    g.get_legend().remove()
    # Improve the legend 
    handles, labels = g.get_legend_handles_labels()
    print (labels)
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.85), loc='lower center', ncol=4, fancybox=True, shadow=True)

    ax = fig.add_subplot(gs[1,0])
    g = sns.pointplot(x="Dataset", y="Interval Score", hue="Method", 
                      markers=["*","D","o", "x" ],
                      linestyles=["-","--","-.",":"],
                      dodge=0.45,
                      join=False, ci='sd',palette=["C2", "C3", "C0", "C1"],
                       data=df_pred_uci, legend=False)
    g.set(yscale="log")
    g.get_legend().remove()
    g.set_xticklabels(new_dataset_names, fontsize='x-small')
    plt.savefig(os.path.join(output_dir, "RMSE_IS_Comb_point_uci.pdf"))
    plt.show()

def gen_plots(df_pred_uci):
    
    print (f"Generating Entropy plot")
    print (df_pred_uci.head())
    print (df_pred_uci[df_pred_uci['Method']=="Ensemble"].head())
    df_pred_uci["Sigma"] = pd.to_numeric(df_pred_uci.Sigma, errors='coerce')
    df_pred_uci["Mu"] = pd.to_numeric(df_pred_uci.Mu, errors='coerce')
    df_pred_uci["Target"] = pd.to_numeric(df_pred_uci.Target, errors='coerce')

    # No use of platting Entropy so disabling
    #df_pred_uci["Entropy"] = 0.5*np.log( 2 * np.pi * np.exp(1.) * (df_pred_uci["Sigma"])**2 )
    #df_pred_uci["Entropy"].mask(df_pred_uci["Method"]=="Laplace", np.log(2*df_pred_uci["Sigma"]*np.exp(1.)), inplace=True) #  entropy for laplace distirbution
    ### Plot PDF for each Dataset 
    #g = sns.FacetGrid(df_pred_uci, col="Dataset", hue="Method")
    #g.map(sns.distplot, "Entropy").add_legend()
    #plt.savefig(os.path.join(".", f"entropy_pdf_per_method.pdf"))
    #plt.show()
    
    #sns.catplot(x="Dataset", y="Entropy", hue="Method", data=df_pred_uci, kind="box", whis=0.5, showfliers=False)
    #plt.savefig(os.path.join(output_dir, f"entropy_box_uci.pdf"))
    #plt.show()
    
    print ("Plot error distribution")
    df_pred_uci["RMSE"] = (df_pred_uci["Mu"] - df_pred_uci["Target"])**2
    g = sns.catplot(x="Dataset", y="RMSE", hue="Method", data=df_pred_uci, kind="box", whis=0.5, showfliers=False, aspect=4.0)
    plt.savefig(os.path.join(output_dir, f"RMSE_box_uci.pdf"))
    plt.show()
    
    g = sns.catplot(x="Method", y="RMSE", col="Dataset", data=df_pred_uci, kind="box", whis=0.5, showfliers=False, aspect=0.3)
    plt.savefig(os.path.join(output_dir, f"RMSE_box_uci_panel.pdf"))
    plt.show()

    print (f"Generating Interval Score")
    df_pred_uci["lower"] = df_pred_uci["Mu"] - 2*df_pred_uci["Sigma"]
    df_pred_uci["lower"].mask(df_pred_uci["Method"]=="Laplace", df_pred_uci["Mu"] - 3*df_pred_uci["Sigma"], inplace=True)
    df_pred_uci["upper"] = df_pred_uci["Mu"] + 2*df_pred_uci["Sigma"]
    df_pred_uci["upper"].mask(df_pred_uci["Method"]=="Laplace", df_pred_uci["Mu"] + 3*df_pred_uci["Sigma"], inplace=True)
    
    df_pred_uci["Interval Score"] = df_pred_uci["upper"] - df_pred_uci["lower"] \
     + (2/0.95)*(df_pred_uci["lower"]-df_pred_uci["Target"])*(df_pred_uci["Target"]<df_pred_uci["lower"]) \
     + (2/0.95)*(df_pred_uci["Target"] - df_pred_uci["upper"])*(df_pred_uci["Target"]>df_pred_uci["upper"])
    
    g = sns.catplot(x="Dataset", y="Interval Score", hue="Method", data=df_pred_uci, kind="box", whis=0.5, showfliers=False, aspect=4.0)
    g.set(yscale="log")
    plt.savefig(os.path.join(output_dir, f"Interval_score_box_uci.pdf"))
    plt.show()
    
    g = sns.catplot(x="Method", y="Interval Score", col="Dataset", data=df_pred_uci, kind="box", whis=0.5, showfliers=False, aspect=0.7)
    g.set(yscale="log")
    plt.savefig(os.path.join(output_dir, f"Interval_score_box_uci_panel.pdf"))
    plt.show()

if args.load_pkl:
    print("Loading!")
    df_pred_uci = pd.read_pickle("cached_uci_results.pkl")
else:
    df_pred_uci = compute_predictions()
    df_pred_uci.to_pickle("cached_uci_results.pkl")

"""=============================="""
Path(output_dir).mkdir(parents=True, exist_ok=True)
#gen_plots(df_pred_uci)
#gen_paper_plots(df_pred_uci)
simplified_paper_plot(df_pred_uci)
