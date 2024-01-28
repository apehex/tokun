import tensorflow as tf

# HISTOGRAMS ##################################################################

def save_model_histograms(model: tf.Module, epoch: int, summary: 'ResourceSummaryWriter') -> None:
    with summary.as_default():
        for __p in model.trainable_variables:
            tf.summary.histogram(__p.name, __p, step=epoch)

# LOSS ########################################################################

def save_loss_plot(data: list, name: str, summary: 'ResourceSummaryWriter', offset: int=0) -> None:
    with summary.as_default():
        for __i, __l in data:
            tf.summary.scalar(name, __l, step=__i + offset)

# GRAD / VALUE ################################################################

def save_ratios_plot(data: list, model: tf.Module, summary: 'ResourceSummaryWriter', offset: int=0) -> None:
    with summary.as_default():
        for __i, __ratios in data:
            for __j, __r in enumerate(__ratios):
                tf.summary.scalar(model.trainable_variables[__j].name + '_log10(gradient/value)', __r, step=__i + offset)
