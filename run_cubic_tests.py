import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import tensorflow_probability as tfp
from pathlib import Path
import os
import time

import edl
import data_loader
import trainers
import models


parser = argparse.ArgumentParser()

parser.add_argument("--load-pkl", action='store_true',
                    help="Load predictions for a cached pickle file or \
                        recompute from scratch by feeding the data through \
                        trained models")
args = parser.parse_args()

'''========================================================='''
seed = 1234
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

save_fig_dir = "./figs/toy"
batch_size = 128
iterations = 8000
show = True
'''
noise_changing = False
train_bounds = [[-4, 4]]
x_train = np.concatenate([np.linspace(xmin, xmax, 1000) for (xmin, xmax) in train_bounds]).reshape(-1,1)
y_train, sigma_train = data_loader.generate_cubic(x_train, noise=True)
x_train_robustness = np.concatenate([np.linspace(xmin, xmax, 50) for (xmin, xmax) in train_bounds]).reshape(-1,1)
y_train_robustness = np.random.uniform(-150, 150, 50).reshape(-1,1)
x_train = np.concatenate((x_train,x_train_robustness))
y_train = np.concatenate((y_train, y_train_robustness))

test_bounds = [[-7,+7]]
x_test = np.concatenate([np.linspace(xmin, xmax, 1000) for (xmin, xmax) in test_bounds]).reshape(-1,1)
y_test, sigma_test = data_loader.generate_cubic(x_test, noise=False)

print ("Before x_train y_train ", x_train.shape, y_train.shape)
'''

x_train, y_train, sigma_train = data_loader.synthetic_sine_heteroscedastic(n_points=1000)
x_train = x_train.reshape(-1,1); y_train=y_train.reshape(-1,1)
x_test, y_test, sigma_test = data_loader.synthetic_sine_heteroscedastic(n_points=1000, noise=False)
x_test = x_test.reshape(-1,1); y_test=y_test.reshape(-1,1)
print ("After x_train y_train ", x_train.shape, y_train.shape)
x_train_robustness = np.linspace(0, 15, 50).reshape(-1,1)
y_train_robustness = np.random.uniform(-10, 10, 50).reshape(-1,1)
#x_train = np.concatenate((x_train,x_train_robustness))
#y_train = np.concatenate((y_train, y_train_robustness))

### Plotting helper functions ###
def plot_scatter():
    plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0)
    plt.scatter(x_train_robustness, y_train_robustness, s=1.2, c='r', marker='x',zorder=0)
    plt.plot(x_test, y_test, 'r--', zorder=2)
    plt.gca().set_xlim((min(x_test), max(x_test)))
    #plt.gca().set_ylim(min(y_train),max(y_train))
    plt.gca().set_ylim(-8.0,8.0)
    plt.title("Toy dataset Sine wave with Heteroscedastic uncertainty")
    plt.savefig(save_fig_dir+"/data.pdf", transparent=True)
    if show:
        plt.show(block=False)
        time.sleep(1.0)
    plt.clf()


def plot_scatter_with_var(mu, var, path, n_stds=3):
    plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0)
    plt.scatter(x_train_robustness, y_train_robustness, s=1.2, c='r', marker='x',zorder=0)
    for k in np.linspace(0, n_stds, 4):
        plt.fill_between(x_test[:,0], (mu-k*var)[:,0], (mu+k*var)[:,0], alpha=0.1, edgecolor=None, facecolor='#00aeef', linewidth=0, antialiased=True, zorder=1)

    plt.plot(x_test, y_test, 'r--', zorder=2)
    plt.plot(x_test, mu, color='#007cab', zorder=3)
    plt.gca().set_xlim((min(x_test), max(x_test)))
    #plt.gca().set_ylim(min(y_train),max(y_train))
    plt.gca().set_ylim(-8.0,8.0)
    plt.axis("off")
    #plt.title(path)
    plt.savefig(path, transparent=True, dpi=80)
    if show:
        plt.show(block=False)
        time.sleep(1.0)
    plt.clf()

def plot_ng(model, x_test, save="ng", ext=".pdf"):
    x_test_input = tf.convert_to_tensor(x_test, tf.float32)
    outputs = model(x_test_input)
    mu, v, alpha, beta = tf.split(outputs, 4, axis=1)

    epistemic = np.sqrt(beta/(v*(alpha-1)))
    epistemic = np.minimum(epistemic, 1e3) # clip the unc for vis
    aleatoric = np.sqrt(beta/(alpha - 1))
    aleatoric = np.minimum(aleatoric, 1e3) # clip the unc for vis
    #plot_scatter_with_var(mu, epistemic, path=save+ext, n_stds=3)
    plot_scatter_with_var(mu, aleatoric, path=save+experiment_name+ext, n_stds=3)
    return mu, aleatoric


def plot_ensemble(models, x_test, save="ensemble", ext=".pdf"):
    x_test_input = tf.convert_to_tensor(x_test, tf.float32)
    preds = tf.stack([model(x_test_input, training=False) for model in models], axis=0) #forward pass
    mus, sigmas = tf.split(preds, 2, axis=-1)

    mean_mu = tf.reduce_mean(mus, axis=0)
    epistemic = tf.math.reduce_std(mus, axis=0) + tf.reduce_mean(sigmas, axis=0)
    aleatoric = tf.reduce_mean(sigmas, axis=0)
    #plot_scatter_with_var(mean_mu, epistemic, path=save+experiment_name+ext, n_stds=3)
    plot_scatter_with_var(mean_mu, aleatoric, path=save+experiment_name+ext, n_stds=3)
    return mean_mu, aleatoric

def plot_dropout(model, x_test, save="dropout", ext=".pdf"):
    x_test_input = tf.convert_to_tensor(x_test, tf.float32)
    preds = tf.stack([model(x_test_input, training=True) for _ in range(15)], axis=0) #forward pass
    mus, logvar = tf.split(preds, 2, axis=-1)
    var = tf.exp(logvar)

    mean_mu = tf.reduce_mean(mus, axis=0)
    epistemic = tf.math.reduce_std(mus, axis=0) + tf.reduce_mean(var**0.5, axis=0)
    aleatoric = tf.reduce_mean(var**0.5, axis=0)
    #plot_scatter_with_var(mean_mu, epistemic, path=save+experiment_name+ext, n_stds=3)
    plot_scatter_with_var(mean_mu, aleatoric, path=save+experiment_name+ext, n_stds=3)
    return mean_mu, aleatoric


def plot_gaussian(model, x_test, save="gaussian", ext=".pdf"):
    x_test_input = tf.convert_to_tensor(x_test, tf.float32)
    preds = model(x_test_input, training=False) #forward pass
    mu, var = tf.split(preds, 2, axis=-1)
    plot_scatter_with_var(mu, var**0.5, path=save+experiment_name+ext, n_stds=3)
    return mu, var**0.5

def plot_laplace(model, x_test, save="laplace", ext=".pdf"):
    x_test_input = tf.convert_to_tensor(x_test, tf.float32)
    preds = model(x_test_input, training=False) #forward pass
    mu, b = tf.split(preds, 2, axis=-1)
    plot_scatter_with_var(mu, 1.5*b, path=save+experiment_name+ext, n_stds=3) #1normal_sigma = 1.5lapalce b
    return mu, b
'''=========================================================================
def plot_laplace_likelihood_ensemble(models, x_test, save="laplace_ensemble", ext=".pdf"):
    x_test_input = tf.convert_to_tensor(x_test, tf.float32)
    preds = tf.stack([model(x_test_input, training=False) for model in models], axis=0) #forward pass
    mus, sigmas = tf.split(preds, 2, axis=-1)

    mean_mu = tf.reduce_mean(mus, axis=0)
    epistemic = tf.math.reduce_std(mus, axis=0) + 2*tf.square(tf.reduce_mean(sigmas, axis=0))
    plot_scatter_with_var(mean_mu, epistemic, path=save+ext, n_stds=3)

def plot_bbbp(model, x_test, save="bbbp", ext=".pdf"):
    x_test_input = tf.convert_to_tensor(x_test, tf.float32)
    preds = tf.stack([model(x_test_input, training=True) for _ in range(15)], axis=0) #forward pass

    mean_mu = tf.reduce_mean(preds, axis=0)
    epistemic = tf.math.reduce_std(preds, axis=0)
    plot_scatter_with_var(mean_mu, epistemic, path=save+ext, n_stds=3)
    return mean_mu, epistemic
========================================================================= '''
#### Different toy configurations to train and plot
def evidence_reg_2_layers_50_neurons():
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=50, num_layers=2)
    trainer = trainer_obj(model, opts, dataset="toy", learning_rate=5e-3, lam=1e-2, maxi_rate=0.)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    plot_ng(model, os.path.join(save_fig_dir,"evidence_reg_2_layer_50_neurons"))

def evidence_reg_2_layers_100_neurons():
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=100, num_layers=2)
    trainer = trainer_obj(model, opts, dataset="toy", learning_rate=5e-3, lam=1e-2, maxi_rate=0.)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    plot_ng(model, os.path.join(save_fig_dir,"evidence_reg_2_layers_100_neurons"))

def evidence_reg_4_layers_50_neurons():
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=50, num_layers=4)
    trainer = trainer_obj(model, opts, dataset="toy", learning_rate=5e-3, lam=1e-2, maxi_rate=0.)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    plot_ng(model, os.path.join(save_fig_dir,"evidence_reg_4_layers_50_neurons"))

def evidence_reg_4_layers_100_neurons( x_train, y_train, x_test, lam=1e-2,):
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=100, num_layers=4)
    trainer = trainer_obj(model, opts, dataset="toy", learning_rate=5e-3, lam=lam, maxi_rate=0., save_files=False)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    return plot_ng(model, x_test, os.path.join(save_fig_dir,"evidence_reg_4_layers_100_neurons"))

def evidence_noreg_4_layers_50_neurons():
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=50, num_layers=4)
    trainer = trainer_obj(model, opts, dataset="toy", learning_rate=5e-3, lam=0., maxi_rate=0.)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    plot_ng(model, os.path.join(save_fig_dir,"evidence_noreg_4_layers_50_neurons"))

def evidence_noreg_4_layers_100_neurons():
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=100, num_layers=4)
    trainer = trainer_obj(model, opts, dataset="toy", learning_rate=5e-3, lam=0., maxi_rate=0.)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    plot_ng(model, os.path.join(save_fig_dir,"evidence_noreg_4_layers_100_neurons"))

def ensemble_4_layers_100_neurons(x_train, y_train, x_test):
    trainer_obj = trainers.Ensemble
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=100, num_layers=4)
    trainer = trainer_obj(model, opts, dataset="toy", learning_rate=5e-3, save_files=False)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    return plot_ensemble(model, x_test, os.path.join(save_fig_dir,"ensemble_4_layers_100_neurons"))

def laplace_ensemble_4_layers_100_neurons(): #This function will not work because trainer chagnes
    trainer_obj = trainers.Likelihood
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=100, num_layers=4)
    trainer = trainer_obj(model, opts, "laplace", dataset="toy", learning_rate=5e-3)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    plot_laplace_likelihood_ensemble(model, os.path.join(save_fig_dir,"likelihood_4_layers_100_neurons"))
    
def laplace_4_layers_100_neurons(x_train, y_train, x_test): 
    trainer_obj = trainers.Likelihood
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=100, num_layers=4)
    trainer = trainer_obj(model, opts, "laplace", dataset="toy", learning_rate=5e-3 , save_files=False)
    print (x_train.shape)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    return plot_laplace(model, x_test, os.path.join(save_fig_dir,"laplace_4_layers_100_neurons"))

def gaussian_4_layers_100_neurons(x_train, y_train, x_test): #This function will not work because trainer chagnes
    trainer_obj = trainers.Likelihood
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=100, num_layers=4)
    trainer = trainer_obj(model, opts, "gaussian", dataset="toy",learning_rate=5e-3, save_files=False)
    print (x_train.shape)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    return plot_gaussian(model, x_test, os.path.join(save_fig_dir,"gaussian_4_layers_100_neurons"))
'''    
def gaussian_4_layers_100_neurons():
    trainer_obj = trainers.Gaussian
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=100, num_layers=4)
    trainer = trainer_obj(model, opts, learning_rate=5e-3)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    plot_gaussian(model, os.path.join(save_fig_dir,"gaussian_4_layers_100_neurons"))
'''    

def dropout_4_layers_100_neurons(x_train, y_train, x_test):
    trainer_obj = trainers.Dropout
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=100, num_layers=4, sigma=True)
    trainer = trainer_obj(model, opts, dataset="toy", learning_rate=5e-3, save_files=False)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    return plot_dropout(model, x_test, os.path.join(save_fig_dir,"dropout_4_layers_100_neurons"))

def bbbp_4_layers_100_neurons(x_train, y_train, x_test):
    trainer_obj = trainers.BBBP
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=100, num_layers=4)
    trainer = trainer_obj(model, opts, dataset="toy", learning_rate=1e-3)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    plot_bbbp(model, x_test, os.path.join(save_fig_dir,"bbbp_4_layers_100_neurons"))

def calculate_rmse_interval_score(y_test, mu, aleatoric, loss_type="gaussian"):
    mu = np.squeeze(mu.numpy())
    aleatoric = np.squeeze(aleatoric)
    y_test = np.squeeze(y_test)
    
    rmse = (mu - y_test)**2
    #Calculat IS
    if "gaussian" == loss_type:
        lower = mu - 2*aleatoric
        upper = mu + 2*aleatoric
    elif "laplace" == loss_type:
        lower = mu - 3*aleatoric
        upper = mu + 3*aleatoric

    interval_score = (upper - lower) + (2/0.95)*(lower-y_test)*(y_test<lower) \
                                     + (2/0.95)*(y_test-upper)*(y_test>upper)
    return rmse, interval_score
     
def emperical_breakaway_point():
    #Break away point is the point where the amount of outliers causes the network to give significant bias
    global x_train, y_train, x_train_robustness, y_train_robustness, experiment_name
    function_names = [ensemble_4_layers_100_neurons, 
                      gaussian_4_layers_100_neurons, 
                      evidence_reg_4_layers_100_neurons,
                      laplace_4_layers_100_neurons]
    method_names = ['Ensemble','Gaussian','Evidential','Laplace']

    df_pred = pd.DataFrame(columns=["Method", "noise", "RMSE", "Interval Score", "X", "Y", "Mu", "Sigma", "GT_Sigma"])
    for j in range(3):
        x_train_first, y_train_first, sigma_train = data_loader.synthetic_sine_heteroscedastic(n_points=1000)
        x_train_first = x_train_first.reshape(-1,1); y_train_first=y_train_first.reshape(-1,1)
        x_test, y_test, sigma_test = data_loader.synthetic_sine_heteroscedastic(n_points=1000, noise=False)
        x_test = x_test.reshape(-1,1); y_test=y_test.reshape(-1,1)
        print ("After x_train y_train ", x_train_first.shape, y_train_first.shape)
     
        for f, method_name in zip(function_names, method_names):
            for noise in [0, 100, 300, 500]:
                experiment_name = f"{method_name}_{j}_{noise}"
                x_train_robustness = np.linspace(0, 15, noise).reshape(-1,1)
                y_train_robustness = np.random.uniform(-10, 10, noise).reshape(-1,1)
                x_train = np.concatenate((x_train_first,x_train_robustness))
                y_train = np.concatenate((y_train_first, y_train_robustness))
                mu, sigma = f(x_train, y_train, x_test)
                if method_name == "Laplace":
                    rmse, interval_score = calculate_rmse_interval_score(y_test, mu, sigma, loss_type="laplace")
                else:
                    rmse, interval_score = calculate_rmse_interval_score(y_test, mu, sigma, loss_type="gaussian")

                mu = np.squeeze(mu.numpy())
                sigma = np.squeeze(sigma)
                sigma_train = np.squeeze(sigma_train)
                summary = [{"run":j,"Method":method_name, "noise":noise, "RMSE":r, "Interval Score":i ,
                            "X":x, "Y":y, "Mu":m, "Sigma":s, "GT_Sigma":st} for r,i,x,y,m,s,st in zip(rmse, interval_score, x_test, y_test, mu, sigma, sigma_train)]
                df_pred = df_pred.append(summary, ignore_index=True)

                print (df_pred.head())
                tf.keras.backend.clear_session()

    return df_pred


def plot_breakaway_point(df_pred):
    cm = 1/2.54  # centimeters in inches
    sns.set()
    sns.set_style("white")
    sns.set_style("ticks")
    sns.despine(trim=True)
    sns.set_context("paper")
    sns.color_palette("tab10")

    #convert noise to noise percentage
    print ("Noise unique values ", df_pred['noise'].unique())
    df_pred['noise'] = (df_pred['noise']/1000) #1000 clean labels
    print ("Noise unique values ", df_pred['noise'].unique())
    percentage_noise = df_pred['noise'].unique()
    df_pred = df_pred.rename(columns={"noise":"Noise %"})

    #Removing Dropout
    df_pred = df_pred[df_pred.Method != "Dropout"]
    fig = plt.figure(figsize=(14.2*cm, 14.2*cm/2.0))
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    g = sns.pointplot(x="Noise %", y="RMSE", hue="Method", 
                      markers=["o", "x", "*", "D"],
                      linestyles=["-","--","-.",":"],
                       data=df_pred, legend=False)
    #g.legend(bbox_to_anchor=(0.8, 1.2 ), loc='upper center', ncol=5, fontsize=5, fancybox=True, shadow=True) 
    g.get_legend().remove()
    # manipulate x tick labels to add percentage
    vals = g.get_xticks()
    print ("vals : ", vals)
    print (["% vals {:.0%}".format(x) for x in percentage_noise])
    g.set_xticklabels(['{:.0%}'.format(x) for x in percentage_noise])

    #g.set_aspect(1.5)
    ax = fig.add_subplot(gs[0, 1])
    g = sns.pointplot(x="Noise %", y="Interval Score", hue="Method", 
                      markers=["o", "x", "*", "D"],
                      linestyles=["-","--","-.",":"],
                       data=df_pred, legend=False)
    g.set(yscale="log")
    # manipulate x tick labels to add percentage
    vals = g.get_xticks()
    #g.set_xticklabels(['{:.0%}'.format(x) for x in vals])
    g.set_xticklabels(['{:.0%}'.format(x) for x in percentage_noise])

    #fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)  # create some space below the plots by increasing the bottom-value
    #g.legend(loc='upper center', bbox_to_anchor=(0.1, -0.12), ncol=5)
    g.get_legend().remove()
    handles, labels = g.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.55, 0.92), loc='lower center', ncol=5, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_fig_dir, "noise_RMSE_IS_point_plot.pdf"),bbox_inches='tight')
    plt.show(block=True)
    plt.clf()

    '''
    g = sns.catplot(x="Method", y="RMSE", hue="Noise %", kind="box", 
                    data=df_pred, whis=0.5)
    g.set(yscale="log")
    plt.savefig(os.path.join(save_fig_dir, "noise_RMSE_box_plot.pdf"))
    plt.show(block=False)
    plt.clf()

    g = sns.catplot(x="Method", y="Interval Score", hue="Noise %", kind="box",
                    data=df_pred, whis=0.5)
    g.set(yscale="log")
    plt.savefig(os.path.join(save_fig_dir, "noise_IS_box_plot.pdf"))
    plt.show(block=False)
    plt.clf()
    '''

    ''' An attempt to plot 1 below other
    fig = plt.figure(figsize=(14.2*cm/2, (14.2*cm/3)*2.2))
    gs = fig.add_gridspec(2, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    g = sns.lineplot(x="Noise %", y="RMSE", hue="Method", style="Method", data=df_pred, legend='brief')
    g.legend(bbox_to_anchor=(0.3, 1.2 ), loc='upper center', ncol=2, fontsize=5, fancybox=True, shadow=True) 
    # manipulate x tick labels to add percentage
    vals = g.get_xticks()
    g.set_xticklabels(['{:.0%}'.format(x) for x in vals])
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.axes.get_xaxis().get_label().set_visible(False)

    #g.set_aspect(1.5)
    ax = fig.add_subplot(gs[1, 0], sharex=ax1)
    g = sns.lineplot(x="Noise %", y="Interval Score", hue="Method", style="Method", data=df_pred, legend=False)

    g.set(yscale="log")
    # manipulate x tick labels to add percentage
    vals = g.get_xticks()
    g.set_xticklabels(['{:.0%}'.format(x) for x in vals])
    plt.tight_layout()
    plt.savefig(os.path.join(save_fig_dir, "noise_RMSE_IS_line_plot.pdf"))
    plt.show(block=True)
    plt.clf()

    g = sns.lineplot(x="Noise %", y="Interval Score", hue="Method", style="Method", data=df_pred)
    g.set(yscale="log")
    # manipulate x tick labels to add percentage
    vals = g.get_xticks()
    g.set_xticklabels(['{:.0%}'.format(x) for x in vals])
    g.set_aspect(1.5)
    plt.savefig(os.path.join(save_fig_dir, "noise_IS_line_plot.pdf"))
    plt.show(block=False)
    plt.clf()

    df_pred['Error'] = abs(df_pred['Y'][0] - df_pred['Mu'])
    print(df_pred.head())
    g = sns.catplot(x="Method", y="Error", hue="Noise %", kind="box",
                    data=df_pred, whis=0.5)
    plt.savefig(os.path.join(save_fig_dir, "noise_Error_box_plot.pdf"))
    plt.show(block=False)
    plt.clf()

    g = sns.lineplot(x="Noise %", y="Error", hue="Method", data=df_pred)
    plt.savefig(os.path.join(save_fig_dir, "noise_Error_line_plot.pdf"))
    plt.show(block=False)
    plt.clf()

    df_pred['Sigma Error'] = abs(df_pred['Sigma'] - df_pred['GT_Sigma'])
    g = sns.catplot(x="Method", y="Sigma Error", hue="Noise %", kind="box",
                    data=df_pred, whis=0.5)
    plt.savefig(os.path.join(save_fig_dir, "noise_Sigma_Error_box_plot.pdf"))
    plt.show(block=False)
    plt.clf()
    g = sns.lineplot(x="Noise %", y="Sigma Error", hue="Method", data=df_pred)
    g.set(yscale="log")
    plt.savefig(os.path.join(save_fig_dir, "noise_Sigma_Error_line_plot.pdf"))
    plt.show(block=False)
    plt.clf()
    '''


    
### Main file to run the different methods and compare results ###
if __name__ == "__main__":
    print ("Tf Version :", tf.__version__ )
    print (tf.executing_eagerly() )
    Path(save_fig_dir).mkdir(parents=True, exist_ok=True)

    #evidence_reg_4_layers_100_neurons(lam=1e-3)
    #evidence_reg_4_layers_100_neurons(lam=5e-3)
    #evidence_reg_4_layers_100_neurons(lam=1e-2)
    #evidence_reg_4_layers_100_neurons(lam=5e-2)
    #evidence_noreg_4_layers_100_neurons()

    #ensemble_4_layers_100_neurons()
    #gaussian_4_layers_100_neurons()
    #dropout_4_layers_100_neurons()
    #bbbp_4_layers_100_neurons()
    #laplace_ensemble_4_layers_100_neurons()
    #laplace_4_layers_100_neurons()
    #gaussian_4_layers_100_neurons()

    #plot_scatter()
    if args.load_pkl:
        print("Loading!")
        df_pred = pd.read_pickle(os.path.join(save_fig_dir, "cached_toy_results.pkl")) 
    else:
        df_pred = emperical_breakaway_point()
        df_pred.to_pickle(os.path.join(save_fig_dir, "cached_toy_results.pkl"))
        plot_breakaway_point(df_pred)

    """================Plotting============"""
    plot_breakaway_point(df_pred)
    #print(f"Done! Figures saved to {save_fig_dir}")
