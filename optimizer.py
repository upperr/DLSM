import tensorflow as tf
from utils import *

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

SMALL = 1e-16

class Optimizer(object):
    def __init__(self, labels, model, epoch, num_nodes, features, pos_weight, norm, weighted_ce, edges_for_loss, norm_feats, pos_weight_feats, node_labels = None, node_labels_mask = None, start_semisup=1.):

        labels_sub = labels
        pos_weight_mod = pos_weight
        epoch = tf.cast(epoch, tf.float32)

        if weighted_ce == 0:
            # Loss not weighted
            norm = 1
            pos_weight = 1

        else:
            pos_weight_mod = pos_weight * FLAGS.bias_weight_1
            
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        preds_sub = model.reconstructions
        #neg_ll = tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight_mod)
        
        # S MC samples:
        self.nll = tf.constant(0.0)
        if(FLAGS.link_prediction):
            for s in range(model.S):
                preds_sub = model.reconstructions_list[s]
                
                if FLAGS.data_type == "count":
                    neg_ll = tf.nn.log_poisson_loss(labels_sub, tf.math.log(preds_sub)) 
                else:
                    neg_ll = self.binary_weighted_crossentropy(preds_sub, labels_sub, pos_weight)
                
                
                neg_ll = neg_ll * edges_for_loss
                neg_ll = norm * tf.reduce_mean(input_tensor=neg_ll)

                self.nll += neg_ll

            self.nll = self.nll / model.S 
        self.check = model.reconstructions_list

        #else:
        #self.nll = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        
        #Regularization Loss
        self.regularization = model.reg_phi + model.get_regualizer_cost(tf.nn.l2_loss)
       
       #self.regularization = tf.reduce_mean([tf.nn.l2_loss(v) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = ".*semisup_weight_vars/weights")])# scope = ".*weights")])
        # X-Reconstruction Loss
        self.x_loss = tf.constant(0.0)
        if(FLAGS.features==1 and FLAGS.reconstruct_x==1):
            #X (features) reconstruction loss
            x_recon = model.x_recon
            #pos_weight = 1
            #0-1 features. weighing required
            self.x_loss = tf.nn.weighted_cross_entropy_with_logits(logits=x_recon, labels=features, pos_weight=pos_weight_feats)
            self.x_loss = tf.reduce_mean(input_tensor=self.x_loss) * norm_feats

        # Classification loss in case of semisupervised training
        self.semisup_loss = tf.constant(0.0)
        self.semisup_acc = tf.constant(0.0)
        if(FLAGS.semisup_train):

            #Only use for finetuning

            preds = model.z
            mask = node_labels_mask

            #preds_softmax = tf.nn.softmax(preds, axis = 1)
            #self.entropy = tf.reduce_mean(tf.reduce_sum(-preds_softmax * tf.log(preds_softmax + SMALL2), axis=1), axis=0)

            self.semisup_acc = masked_accuracy(preds, node_labels,  mask)
            
            loss = tf.nn.softmax_cross_entropy_with_logits(logits = preds, labels = tf.stop_gradient( node_labels))
            mask = tf.cast(mask, dtype = tf.float32)
            mask = mask/tf.reduce_mean(input_tensor=mask)
            self.semisup_loss = tf.reduce_mean(input_tensor=loss * mask)

            #start semisupervised training after a while
            self.semisup_loss = self.semisup_loss * start_semisup


        # KL-divergence loss
        self.kl = 0
        for k in range(model.S):
            #for idx in range(model.num_hidden_layers):
            for idx in range(model.num_decoder_layers):
                
                mean_posterior = model.posterior_theta_param_list[k][idx][0]
                log_std = model.posterior_theta_param_list[k][idx][1]
                #alpha_g = model.prior_theta_list[s][idx][0]
                #beta_g = model.prior_theta_list[s][idx][1]
                #self.kl_term += kl_weibull_gamma(k_w, l_w, alpha_g, beta_g) / num_nodes
                #mean = tf.convert_to_tensor(mean, 'float32')
                #log_std = tf.convert_to_tensor(log_std, 'float32')
                self.kl_z = kl_normal(mean_posterior, log_std) / num_nodes
                
                pi_logit_prior = model.prior_theta_param_list[k][idx][0]
                pi_logit_posterior = model.posterior_theta_param_list[k][idx][2]
                s_logit = model.theta_list[k][idx][1]
                self.kl_s = kl_binconcrete(pi_logit_posterior, pi_logit_prior, s_logit, FLAGS.temp_post, FLAGS.temp_prior) / num_nodes
                #self.kl_s = kl_bernoulli(pi_logit_posterior, pi_logit_prior) / num_nodes
                
                alpha_gam_prior = model.prior_theta_param_list[k][idx][1]
                #alpha_prior = model.prior_theta_param_list[k][idx][1]
                #alpha_prior = model.prior_theta_param_list[k][idx][0]
                #alpha_gam_posterior = model.posterior_theta_param_list[k][idx][3]
                alpha_gam_posterior = model.posterior_theta_param_list[k][idx][3]
                self.kl_alpha_gam = kl_gamma(alpha_gam_posterior, alpha_gam_prior) / num_nodes
                #self.kl_alpha_gam = kl_gamma(alpha_gam_posterior, alpha_prior) / num_nodes
                if FLAGS.directed == 1:
                    alpha_del_prior = model.prior_theta_param_list[k][idx][2]
                    #alpha_del_posterior = model.posterior_theta_param_list[k][idx][4]
                    alpha_del_posterior = model.posterior_theta_param_list[k][idx][4]
                    self.kl_alpha_del = kl_gamma(alpha_del_posterior, alpha_del_prior) / num_nodes
                    #self.kl_alpha_del = kl_gamma(alpha_del_posterior, alpha_prior) / num_nodes
                else:
                    self.kl_alpha_del = 0.
                """
                if FLAGS.directed == 1:
                    beta_a = model.posterior_theta_param_list[k][idx][5]
                    beta_b = model.posterior_theta_param_list[k][idx][6]
                else:
                    beta_a = model.posterior_theta_param_list[k][idx][4]
                    beta_b = model.posterior_theta_param_list[k][idx][5]
                self.kl_v = kl_kumar_beta(beta_a, beta_b, FLAGS.alpha0, log_beta_prior = np.log(1./FLAGS.alpha0)) / num_nodes
                """
                #self.kl += self.kl_z + self.kl_s + self.kl_alpha_gam + self.kl_alpha_del + self.kl_v
                self.kl += self.kl_z + self.kl_s + self.kl_alpha_gam + self.kl_alpha_del
        #Average
        self.kl = self.kl/model.S

        self.wu_beta = epoch/FLAGS.epochs;
        #self.wu_beta = self.wu_beta
        if FLAGS.use_kl_warmup == 0:
            self.wu_beta = 1

        if FLAGS.use_x_warmup == 0:
            self.wu_x = 1
        else:
            self.wu_x = self.wu_beta

        self.wu_semisup_loss = 1. #* epoch/FLAGS.epochs


        self.ae_loss = self.nll + self.regularization * FLAGS.weight_decay
        self.cost = 1. * self.nll + 1. * self.wu_beta * self.kl + self.wu_x * self.x_loss + self.wu_semisup_loss * self.semisup_loss  + FLAGS.weight_decay * self.regularization # + self.entropy * 1.

        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        #gradient clipping
        #self.grad_vars = tf.clip_by_value(self.grads_vars, -5,5)
        self.clipped_grads_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else 0, var)
                for grad, var in self.grads_vars]

        self.opt_op = self.optimizer.apply_gradients(self.clipped_grads_vars)

        
        with tf.compat.v1.variable_scope("inputs"):
            tf.compat.v1.summary.histogram('label', labels)

        with tf.compat.v1.variable_scope("predictions"):
            tf.compat.v1.summary.histogram('outputs', tf.nn.sigmoid(preds_sub))

        with tf.compat.v1.variable_scope("loss"):
            tf.compat.v1.summary.scalar('cost', self.cost)
            tf.compat.v1.summary.scalar('nll', self.nll)
            tf.compat.v1.summary.scalar('kl_divergence', self.kl)
            tf.compat.v1.summary.scalar('regularization', self.regularization)
        if(FLAGS.features==1 and FLAGS.reconstruct_x==1):
            tf.compat.v1.summary.scalar('x_recon_loss', self.x_loss)
       
        # Add histograms for gradients.
        with tf.compat.v1.variable_scope("gradients"):
            for grad, var in self.grads_vars:
                if grad is not None:
                    tf.compat.v1.summary.histogram(var.op.name + '/gradients', grad)
    
        #self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.nn.sigmoid(preds_sub), 0.5), tf.int32),
        #                                   tf.cast(labels_sub, tf.int32))
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


    def binary_weighted_crossentropy(self, preds, labels, pos_weight):
        """
        Expects probabilities preds
        pos weight: scaling factor for 1-labels for use in unbalanced datasets with lots of zeros(?)
        See this thread for more: https://github.com/tensorflow/tensorflow/issues/2462
        """
        SMALL_VAL = 10e-8
        epsilon = tf.constant(SMALL_VAL)
        preds = tf.clip_by_value(preds, epsilon, 1-epsilon)
        #preds = tf.log(preds)
        
        loss = pos_weight * labels * -tf.math.log(preds) + (1 - labels) * -tf.math.log(1 - preds)
        
        neg_ll = loss
        return neg_ll

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(input=preds, axis=1), tf.argmax(input=labels, axis=1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(input_tensor=mask)
    accuracy_all *= mask
 
    return tf.reduce_mean(input_tensor=accuracy_all)
