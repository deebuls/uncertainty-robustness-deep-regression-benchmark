import tensorflow as tf
import numpy as np

k=1

def MSE(y, y_, reduce=True):
    ax = list(range(1, len(y.shape)))

    mse = tf.reduce_mean((y-y_)**2, axis=ax)
    return tf.reduce_mean(mse) if reduce else mse

def RMSE(y, y_):
    rmse = tf.sqrt(tf.reduce_mean((y-y_)**2))
    return rmse

def Gaussian_NLL(y, mu, sigma, reduce=True):
    ax = list(range(1, len(y.shape)))

    logprob = -tf.math.log(sigma) - 0.5*tf.math.log(2*np.pi) - 0.5*((y-mu)/sigma)**2
    loss = tf.reduce_mean(-logprob, axis=ax)
    return tf.reduce_mean(loss) if reduce else loss

def Gaussian_NLL_logvar(y, mu, logvar, reduce=True):
    ax = list(range(1, len(y.shape)))

    log_liklihood = 0.5 * (
        -tf.exp(-logvar)*(mu-y)**2 - tf.math.log(2*tf.constant(np.pi, dtype=logvar.dtype)) - logvar
    )
    loss = tf.reduce_mean(-log_liklihood, axis=ax)
    return tf.reduce_mean(loss) if reduce else loss

def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2*beta*(1+v)

    nll = 0.5*tf.math.log(np.pi/v)  \
        - alpha*tf.math.log(twoBlambda)  \
        + (alpha+0.5) * tf.math.log(v*(y-gamma)**2 + twoBlambda)  \
        + tf.math.lgamma(alpha)  \
        - tf.math.lgamma(alpha+0.5)

    return tf.reduce_mean(nll) if reduce else nll

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5*(a1-1)/b1 * (v2*tf.square(mu2-mu1))  \
        + 0.5*v2/v1  \
        - 0.5*tf.math.log(tf.abs(v2)/tf.abs(v1))  \
        - 0.5 + a2*tf.math.log(b1/b2)  \
        - (tf.math.lgamma(a1) - tf.math.lgamma(a2))  \
        + (a1 - a2)*tf.math.digamma(a1)  \
        - (b1 - b2)*a1/b1
    return KL

def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
    # error = tf.stop_gradient(tf.abs(y-gamma))
    error = tf.abs(y-gamma)

    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
        reg = error*kl
    else:
        evi = 2*v+(alpha)
        reg = error*evi

    return tf.reduce_mean(reg) if reduce else reg


def Dirichlet_SOS(y, alpha, t):
    def KL(alpha):
        beta=tf.constant(np.ones((1,alpha.shape[1])),dtype=tf.float32)
        S_alpha = tf.reduce_sum(alpha,axis=1,keepdims=True)
        S_beta = tf.reduce_sum(beta,axis=1,keepdims=True)
        lnB = tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha),axis=1,keepdims=True)
        lnB_uni = tf.reduce_sum(tf.math.lgamma(beta),axis=1,keepdims=True) - tf.math.lgamma(S_beta)
        lnB_uni = tf.reduce_sum(tf.math.lgamma(beta),axis=1,keepdims=True) - tf.math.lgamma(S_beta)

        dg0 = tf.math.digamma(S_alpha)
        dg1 = tf.math.digamma(alpha)

        kl = tf.reduce_sum((alpha - beta)*(dg1-dg0),axis=1,keepdims=True) + lnB + lnB_uni
        return kl

    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    evidence = alpha - 1
    m = alpha / S

    A = tf.reduce_sum((y-m)**2, axis=1, keepdims=True)
    B = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True)

    # annealing_coef = tf.minimum(1.0,tf.cast(global_step/annealing_step,tf.float32))
    alpha_hat = y + (1-y)*alpha
    C = KL(alpha_hat)

    C = tf.reduce_mean(C, axis=1)
    return tf.reduce_mean(A + B + C)

def Sigmoid_CE(y, y_logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_logits)
    return tf.reduce_mean(loss)






"""
==========================
==== OLD trial losses ====
==========================
"""

def NG_MLL_v1(y, mu, v, alpha, beta):
    # computed by manually, seems to work but not fully tested...

    A = tf.sqrt(v/(2*np.pi*(1+v)))
    B = (2*beta + v/(1+v)*(mu-y)**2)
    log_liklihood = (alpha+0.5)*tf.math.log(2.) \
        + alpha*tf.math.log(beta) \
        + tf.math.log(A) \
        + (-alpha-0.5)*tf.math.log(B) \
        + tf.math.lgamma(alpha+0.5) \
        - tf.math.lgamma(alpha)

    loss = -log_liklihood
    return tf.reduce_mean(loss)


def NG_MLL_v2(y, mu, v, alpha, beta):
    # computed with mathematica, also seems to work

    log_liklihood = tf.math.log(v)/2. \
        + alpha * tf.math.log(2*beta*(1+v)) \
        - alpha * tf.math.log(2*beta*(1+v)+v*(mu-y)**2) \
        - 0.5 * tf.math.log(np.pi*(2*beta*(1+v)+v*(mu-y)**2)) \
        - tf.math.lgamma(alpha) \
        + tf.math.lgamma(alpha+0.5)

    loss = -log_liklihood
    return tf.reduce_mean(loss)

def NG_MLL_v3(y, gamma, v, alpha, beta):
    log_liklihood = 0.5 * tf.math.log(2.) \
        + alpha * tf.math.log(2.) \
        + alpha * tf.math.log(beta) \
        + 0.5 * tf.math.log(v) \
        - 0.5 * tf.math.log(2*np.pi * (1+v)) \
        - 0.5 * tf.math.log(2*beta + v/(1+v)*(y-gamma)**2) \
        - alpha * tf.math.log(2*beta + v/(1+v)*(y-gamma)**2) \
        - tf.math.lgamma(alpha) \
        + tf.math.lgamma(alpha+0.5)

    # entropy = alpha - tf.math.log(beta) + tf.math.lgamma(alpha) + (1-alpha)*tf.digamma(alpha)
    # loss = -log_liklihood - 1e-6*alpha/beta
    # loss = -log_liklihood - 1e-1*tf.sqrt(beta/(v*alpha))
    # reg = 0.1*(10*tf.reduce_mean(v) + 1*tf.reduce_mean(alpha) - 4*tf.reduce_mean(beta))
    # reg = tf.reduce_mean(v+0.1*alpha)
    # loss = -log_liklihood + 1e-4*reg

    varX = beta/(v*(alpha-1))
    varT = alpha/(beta**2)
    varTotal = varX #+ 1./varT
    reg = tf.math.log(varTotal / tf.exp(log_liklihood))

    loss = -log_liklihood - 1e-1*reg

    return tf.reduce_mean(loss)
