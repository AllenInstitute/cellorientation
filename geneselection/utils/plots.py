import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

plt.switch_backend("agg")

dpi = 100
figx = 6
figy = 4.5


def history(
    simple_logger,
    save_path=None,
    loss_ax1=["recon_loss"],
    loss_ax2=[],
    percentile_max=95,
    percentile_min=0,
    history_len=None,
):

    # Figure out the default color order, and use these for the plots

    if history_len is None:
        history_len = len(simple_logger)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.figure(figsize=(figx, figy), dpi=dpi)

    ax = plt.gca()

    plts = list()
    losses = list()
    i = 0

    for loss_name in loss_ax1:
        # Plot reconstruction loss
        losses += [simple_logger.log[loss_name][-history_len:]]

        plts += ax.plot(
            simple_logger.log["iter"][-history_len:],
            simple_logger.log[loss_name][-history_len:],
            label=loss_name,
            color=colors[i],
        )
        i += 1

    plt.ylabel(loss_name)

    losses = np.hstack(losses)

    ax_max = np.percentile(losses, percentile_max)
    ax_min = np.percentile(losses, percentile_min)

    if ax_max == np.inf:
        y_vals_tmp = np.sort(losses)
        ax_max = y_vals_tmp[y_vals_tmp < np.inf][-1]

    ax.set_ylim([ax_min, ax_max])

    # Print off the reconLoss on it's own scale

    if len(loss_ax2) > 1:
        ax2 = plt.gca().twinx()
        losses = list()

        for loss_name in loss_ax2:
            # Plot reconstruction loss
            losses += [simple_logger.log[loss_name][-history_len:]]

            plts += ax.plot(
                simple_logger.log["iter"][-history_len:],
                simple_logger.log[loss_name][-history_len:],
                label=loss_name,
                color=colors[i],
            )

            i += 1

        losses = np.hstack(losses)

        ax_max = np.percentile(losses, percentile_max)
        ax_min = np.percentile(losses, percentile_min)

        if ax_max == np.inf:
            y_vals_tmp = np.sort(losses)
            ax_max = y_vals_tmp[y_vals_tmp < np.inf][-1]

        ax2.set_ylim([ax_min, ax_max])

        # Get all the labels for the legend from both axes
        labs = [l.get_label() for l in plts]

        # Print legend
        ax.legend(plts, labs)

    plt.ylabel("loss")
    plt.title("History")
    plt.xlabel("iteration")

    # Save
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close()


def PCA_explained_variance(model_pca, save_path, n_dims_cutoff=30):
    # takes in a sklearn pca model object

    if n_dims_cutoff is None:
        n_dims_cutoff = -1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figx, figy))

    dim_var = model_pca.explained_variance_ratio_
    dim_var_cumulative = np.cumsum(dim_var)

    ax1.plot(dim_var[0:n_dims_cutoff], color="k")
    ax1.set_xlabel("dimension #")
    ax1.set_ylabel("dimension variation")
    ax1.set_ylim(0, 1.05)

    ax2.plot(dim_var_cumulative[0:n_dims_cutoff], color="k")
    ax2.set_xlabel("dimension #")
    ax2.set_ylabel("cumulative variation")

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close()


def PCA_dims_multi_y(
    model_pca, save_path_format, X, Y_dict, dims=[0, 1], show_legend=True
):
    princomp = model_pca.transform(X)

    for k in Y_dict:
        Y = Y_dict[k]

        plt.figure()

        scatter(princomp[:, dims[0]], princomp[:, dims[1]], Y)

        ax = plt.gca()
        ax.set_xlabel("pc {}".format(dims[0]))
        ax.set_ylabel("pc {}".format(dims[1]))

        plt.tight_layout()

        if show_legend:
            ax.legend(loc="upper right")

        if save_path_format is not None:
            plt.savefig(save_path_format.format(k), bbox_inches="tight", dpi=dpi)
            plt.close()


def PCA_dims(model_pca, save_path, X, Y=None, dims=[0, 1], top_n_contributers=10):
    # takes in a sklearn pca model object

    princomp = model_pca.transform(X)

    plt.figure()

    scatter(princomp[:, dims[0]], princomp[:, dims[1]], Y)

    ax = plt.gca()
    ax.set_xlabel("pc {}".format(dims[0]))
    ax.set_ylabel("pc {}".format(dims[1]))

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close()


def scatter(x1, x2, Y):

    if Y is None:
        Y = np.ones(x1.shape[0])

    uY = np.unique(Y)
    colors = cm.viridis(np.linspace(0, 1, len(uY)))

    for i, y in enumerate(uY):

        inds = Y == y

        plt.scatter(x1[inds], x2[inds], s=10, color=colors[i], label=y)


# def short_history(
#     simple_logger, save_path, max_history_len=10000, loss_name="recon_loss"
# ):
#     history = int(len(simple_logger.log["epoch"]) / 2)

#     if history > max_history_len:
#         history = max_history_len

#     x = simple_logger.log["iter"][-history:]
#     y = simple_logger.log[loss_name][-history:]

#     epochs = np.floor(np.array(simple_logger.log["epoch"][-history:]))
#     losses = np.array(simple_logger.log[loss_name][-history:])
#     iters = np.array(simple_logger.log["iter"][-history:])
#     uepochs = np.unique(epochs)

#     epoch_losses = np.zeros(len(uepochs))
#     epoch_iters = np.zeros(len(uepochs))
#     i = 0
#     for uepoch in uepochs:
#         inds = np.equal(epochs, uepoch)
#         loss = np.mean(losses[inds])
#         epoch_losses[i] = loss
#         epoch_iters[i] = np.mean(iters[inds])
#         i += 1

#     mval = np.mean(losses)

#     plt.figure(figsize=(figx, figy), dpi=dpi)
#     plt.plot(x, y, label=loss_name)
#     plt.plot(epoch_iters, epoch_losses, color="darkorange", label="epoch avg")
#     plt.plot(
#         [np.min(iters), np.max(iters)],
#         [mval, mval],
#         color="darkorange",
#         linestyle=":",
#         label="window avg",
#     )

#     plt.legend()
#     plt.title("Short history")
#     plt.xlabel("iteration")
#     plt.ylabel("loss")

#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
#     plt.close()


# def embeddings(embedding, save_path):
#     plt.figure(figsize=(figx, figy), dpi=dpi)
#     colors = plt.get_cmap("plasma")(np.linspace(0, 1, embedding.shape[0]))
#     plt.scatter(embedding[:, 0], embedding[:, 1], s=2, color=colors)
#     plt.xlim([-4, 4])
#     plt.ylim([-4, 4])
#     plt.axis("equal")
#     plt.xlabel("z1")
#     plt.ylabel("z2")
#     plt.title("latent space embedding")

#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
#     plt.close()


# def embedding_variation(embedding_paths, figsize=(8, 4), save_path=None):

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
#     colors = cm.viridis(np.linspace(0, 1, len(embedding_paths)))

#     for path, color in zip(embedding_paths, colors):
#         embeddings = pickle.load(open(path, "rb"))

#         var_dims = np.sort(np.var(embeddings, axis=0))[::-1]
#         ax1.plot(var_dims, color=color)
#         ax1.set_xlabel("dimension #")
#         ax1.set_ylabel("dimension variation")
#         ax1.set_ylim(0, 1.05)

#         ax2.plot(np.cumsum(var_dims) / np.sum(var_dims), color=color)
#         ax2.set_xlabel("dimension #")
#         ax2.set_ylabel("cumulative variation")

#     fig.tight_layout()

#     if save_path is not None:
#         plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
#         plt.close()
