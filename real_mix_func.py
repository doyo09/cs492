import torch
# import numpy as np
import matplotlib.pyplot as plt

# TSA
def get_tsa_threshold(curr_step, total_steps, start = 0, end = None,schedule = "log-schdule", class_num=265):
    """
    :param curr_step: 
    :param total_steps: 
    :param start: starting step 
    :param end: 
    :param schedule: log or linear or exp 
    :param class_num: 265
    :return: threshold 
    """
    if end is None :
        end = total_steps
    # curr_step/total_steps
    frac_t_T = torch.tensor(curr_step/total_steps, dtype = torch.float32)
    if schedule.startswith("linear"):
        alpha_t = frac_t_T
    elif  schedule.startswith("exp"):
        scale = 5
        alpha_t = torch.exp((frac_t_T-1) * scale)
    elif schedule.startswith("log"):
        scale = 5
        alpha_t = 1- torch.exp(-frac_t_T*scale)
    else :
        raise ValueError("no schedule")
    threshold = alpha_t * (1 - 1/class_num) + 1/class_num

    return threshold

"""
활용법
        if cfg.tsa:
            tsa_thresh = get_tsa_thresh(cfg.tsa, global_step, cfg.total_steps, start=1./logits.shape[-1], end=1)
            larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh   # prob = exp(log_prob), prob > tsa_threshold
            # larger_than_threshold = torch.sum(  F.softmax(pred[:sup_size]) * torch.eye(num_labels)[sup_label_ids]  , dim=-1) > tsa_threshold
            loss_mask = torch.ones_like(label_ids, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
            sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch_device_one())
"""
"""
OODmasking : tf

    def percent_confidence_mask_unsup(self, logits_y, labels_y, loss_l2u):
        # Adapted from google-research/uda/image/main.py

        # This function masks the unsupervised predictions that are below
        # a set confidence threshold. # Note the following will only work
        # using MSE loss and not KL-divergence.

        # Calculate largest predicted probability for each image.
        unsup_prob = tf.nn.softmax(logits_y, axis=-1)
        largest_prob = tf.reduce_max(unsup_prob, axis=-1)

        # Get the indices of the bottom x% of probabilities and mask those out.
        # In other words, get the probability of the image with the x%*#numofsamples
        # lowest probability and use that as the mask.

        # Calculate the current confidence_mask value using the specified schedule:
        sorted_probs = tf.sort(largest_prob, axis=-1, direction='ASCENDING')
        sort_index = tf.math.multiply(tf.to_float(tf.shape(sorted_probs)[0]), FLAGS.percent_mask)
        curr_confidence_mask = tf.slice(sorted_probs, [tf.to_int64(sort_index)], [1])

        # Mask the loss for images that don't contain a predicted
        # probability above the threshold.
        loss_mask = tf.cast(tf.greater(largest_prob, curr_confidence_mask), tf.float32)
        tf.summary.scalar('losses/high_prob_ratio', tf.reduce_mean(loss_mask)) # The ratio of unl images above the thresh
        tf.summary.scalar('losses/percent_confidence_mask', tf.reshape(curr_confidence_mask,[]))
        loss_mask = tf.stop_gradient(loss_mask)
        loss_l2u = loss_l2u * tf.expand_dims(loss_mask, axis=-1)

        # Return the average unsupervised loss.
        avg_unsup_loss = (tf.reduce_sum(loss_l2u) /
                        tf.maximum(tf.reduce_sum(loss_mask) * FLAGS.nclass, 1))
        return avg_unsup_loss
"""



if __name__ == "__main__":

    steps = list(range(100))
    thrh = [get_tsa_threshold(step, len(steps), ) for step in steps]
    plt.plot(thrh)
    plt.show()
