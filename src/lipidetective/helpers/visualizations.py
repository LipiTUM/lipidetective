import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter


def generate_plots(metrics, trial_id, output_folder, evaluated_params):
    plot_loss_and_accuracy(metrics, trial_id, output_folder, evaluated_params)


def plot_loss_and_accuracy(metric, trial_id, output_path, evaluated_params):
    # Training plot
    output_plot = os.path.join(output_path, f'plot_loss_accuracy_training_trial_{trial_id}.png')

    mean_loss = metric.groupby('epoch').train_loss_step.mean().reset_index()
    accuracy = metric.groupby('epoch').customaccuracy_train_accuracy_step.mean().reset_index()

    figure = plt.figure(figsize=(12, 10))

    ax1 = sns.lineplot(data=mean_loss.train_loss_step, color='orange')
    plt.ylabel('Loss', fontsize=14)
    ax1.set_yscale('log')
    plt.xlabel('Epoch', fontsize=14)
    ax1.legend(['Loss'], loc=(1.15, .92), frameon=False, fontsize=13)

    ax2 = plt.twinx()
    sns.lineplot(data=accuracy.customaccuracy_train_accuracy_step, color='b', ax=ax2)
    plt.ylabel('Accuracy in %', fontsize=14)
    range_nr = list(range(0, 11, 1))
    y_ticks = [x / 10 for x in range_nr]
    ax2.set_yticks(y_ticks)
    y_labels = [str(x) for x in range(0, 101, 10)]
    ax2.set_yticklabels(y_labels)
    ax2.legend(['Accuracy'], loc=(1.15, .95), frameon=False, fontsize=13)

    plt.figtext(0.1, 0, evaluated_params, ha="left", fontsize=10)

    plt.title(f'Loss and Accuracy per Epoch - Trial {trial_id}', fontsize=16)
    plt.tight_layout(w_pad=4)
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close(figure)

    # Validation plot
    output_plot = os.path.join(output_path, f'plot_loss_accuracy_validation_trial_{trial_id}.png')

    loss_accuracy_epoch = metric[['epoch', 'val_loss_epoch', 'customaccuracy_val_accuracy_epoch']]
    loss_accuracy_epoch = loss_accuracy_epoch.rename(columns={'val_loss_epoch': 'loss', 'customaccuracy_val_accuracy_epoch': 'accuracy'})

    loss_accuracy_epoch.index = loss_accuracy_epoch['epoch']
    loss_accuracy_epoch.drop(columns=['epoch'], inplace=True)
    figure = plt.figure(figsize=(12, 10))

    ax1 = sns.lineplot(data=loss_accuracy_epoch.loss, color='orange')
    plt.ylabel('Loss', fontsize=14)
    ax1.set_yscale('log')
    plt.xlabel('Epoch', fontsize=14)
    ax1.legend(['Loss'], loc=(1.15, .92), frameon=False, fontsize=13)

    ax2 = plt.twinx()
    sns.lineplot(data=loss_accuracy_epoch.accuracy, color='b', ax=ax2)
    plt.ylabel('Accuracy in %', fontsize=14)
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(y_labels)
    ax2.legend(['Accuracy'], loc=(1.15, .95), frameon=False, fontsize=13)

    plt.figtext(0.1, 0, evaluated_params, ha="left", fontsize=10)

    plt.title(f'Loss and Accuracy per Epoch - Trial {trial_id}', fontsize=16)
    plt.tight_layout(w_pad=4)
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close(figure)


def plot_loss_and_mae(metric, trial_id, output_path, evaluated_params):
    output_plot_training = os.path.join(output_path, f'plot_loss_mae_training_trial_{trial_id}.png')
    output_plot_validation = os.path.join(output_path, f'plot_loss_mae_validation_trial_{trial_id}.png')

    # Training plot
    mean_loss = metric.groupby('epoch').train_loss_step.mean().reset_index()
    mean_mae_hg = metric.groupby('epoch').train_mae_hg_step.mean().reset_index()
    mean_mae_fa1 = metric.groupby('epoch').train_mae_fa1_step.mean().reset_index()
    mean_mae_fa2 = metric.groupby('epoch').train_mae_fa2_step.mean().reset_index()

    figure = plt.figure(figsize=(12, 10))

    ax1 = sns.lineplot(data=mean_loss.train_loss_step, color='orange')
    plt.ylabel('Loss', fontsize=14)
    ax1.set_yscale('log')
    plt.xlabel('Epoch', fontsize=14)
    ax1.legend(['Loss'], loc=(1.15, .92), frameon=False, fontsize=13)

    ax2 = plt.twinx()
    sns.lineplot(data=mean_mae_hg.train_mae_hg_step, color='darkorchid', ax=ax2)
    sns.lineplot(data=mean_mae_fa1.train_mae_fa1_step, color='cornflowerblue', ax=ax2)
    sns.lineplot(data=mean_mae_fa2.train_mae_fa2_step, color='chocolate', ax=ax2)
    ax2.legend(['mae_hg', '_', 'mae_fa1', '_', 'mae_fa2'], loc=(1.15, .95), frameon=False, fontsize=13)
    plt.ylabel('MAE', fontsize=14)
    ax2.set_yscale('log')

    plt.figtext(0.1, 0, evaluated_params, ha="left", fontsize=10)

    plt.title(f'Loss and MAE per Epoch - Trial {trial_id}', fontsize=16)
    plt.tight_layout(w_pad=4)
    plt.savefig(output_plot_training, dpi=300, bbox_inches='tight')
    plt.close(figure)

    # Validation plot
    mae_metric = metric.copy(deep=True)
    mae_metric.index = mae_metric['epoch']
    mae_metric.drop(columns=['epoch'], inplace=True)

    figure = plt.figure(figsize=(12, 10))

    ax1 = sns.lineplot(data=mae_metric.val_loss_epoch, color='orange')
    plt.ylabel('Loss', fontsize=14)
    ax1.set_yscale('log')
    plt.xlabel('Epoch', fontsize=14)
    ax1.legend(['Loss'], loc=(1.15, .92), frameon=False, fontsize=13)

    ax2 = plt.twinx()
    sns.lineplot(data=mae_metric.val_mae_hg_epoch, color='darkorchid', ax=ax2)
    sns.lineplot(data=mae_metric.val_mae_fa1_epoch, color='cornflowerblue', ax=ax2)
    sns.lineplot(data=mae_metric.val_mae_fa2_epoch, color='chocolate', ax=ax2)
    ax2.legend(['mae_hg', '_', 'mae_fa1', '_', 'mae_fa2'], loc=(1.15, .95), frameon=False, fontsize=13)
    plt.ylabel('MAE', fontsize=14)
    ax2.set_yscale('log')

    plt.figtext(0.1, 0, evaluated_params, ha="left", fontsize=10)

    plt.title(f'Loss and MAE per Epoch - Trial {trial_id}', fontsize=16)
    plt.tight_layout(w_pad=4)
    plt.savefig(output_plot_validation, dpi=300, bbox_inches='tight')
    plt.close(figure)


def plot_loss_and_r2(metric, trial_id, output_path, evaluated_params):
    output_plot_training = os.path.join(output_path, f'plot_loss_r2_training_trial_{trial_id}.png')
    output_plot_validation = os.path.join(output_path, f'plot_loss_r2_validation_trial_{trial_id}.png')

    # Training plot
    mean_loss = metric.groupby('epoch').train_loss_step.mean().reset_index()
    mean_r2 = metric.groupby('epoch').train_r2_step.mean().reset_index()

    figure = plt.figure(figsize=(12, 10))

    ax1 = sns.lineplot(data=mean_loss.train_loss_step, color='orange')
    plt.ylabel('Loss', fontsize=14)
    ax1.set_yscale('log')
    plt.xlabel('Epoch', fontsize=14)
    ax1.legend(['Loss'], loc=(1.15, .92), frameon=False, fontsize=13)

    ax2 = plt.twinx()
    sns.lineplot(data=mean_r2.train_r2_step, color='lightgreen', ax=ax2)
    plt.ylabel('R2', fontsize=14)
    ax2.set_yscale('symlog')
    ax2.yaxis.set_major_formatter(ScalarFormatter())
    ax2.set_ylim(top=3)
    ax2.legend(['R2'], loc=(1.15, .95), frameon=False, fontsize=13)

    plt.figtext(0.1, 0, evaluated_params, ha="left", fontsize=10)

    plt.title(f'Loss and R2 per Epoch - Trial {trial_id}', fontsize=16)
    plt.tight_layout(w_pad=4)
    plt.savefig(output_plot_training, dpi=300, bbox_inches='tight')
    plt.close(figure)

    # Validation plot
    figure = plt.figure(figsize=(12, 10))

    r2_metric = metric.copy(deep=True)
    r2_metric.index = r2_metric['epoch']
    r2_metric.drop(columns=['epoch'], inplace=True)

    ax1 = sns.lineplot(data=r2_metric.val_loss_epoch, color='orange')
    plt.ylabel('Loss', fontsize=14)
    ax1.set_yscale('log')
    plt.xlabel('Epoch', fontsize=14)
    ax1.legend(['Loss'], loc=(1.15, .92), frameon=False, fontsize=13)

    ax2 = plt.twinx()
    sns.lineplot(data=r2_metric.val_r2_epoch, color='lightgreen', ax=ax2)
    plt.ylabel('R2', fontsize=14)
    ax2.set_yscale('symlog')
    ax2.yaxis.set_major_formatter(ScalarFormatter())
    ax2.set_ylim(top=3)
    ax2.legend(['R2'], loc=(1.15, .95), frameon=False, fontsize=13)

    plt.figtext(0.1, 0, evaluated_params, ha="left", fontsize=10)

    plt.title(f'Loss and R2 per Epoch - Trial {trial_id}', fontsize=16)
    plt.tight_layout(w_pad=4)
    plt.savefig(output_plot_validation, dpi=300, bbox_inches='tight')
    plt.close(figure)

