import importlib
import json
import sys
import geneselection.utils as utils
import geneselection.utils.plots as plots
from geneselection.datasets.dataset import gsdataset_from_anndata
import numpy as np
import pickle


def run(model_kwargs, dataset_kwargs, save_dir, seed=0, transform_method=None):

    np.random.seed(seed)

    # load the model
    model_module, model_name = model_kwargs["name"].rsplit(".", 1)
    model_module = importlib.import_module(model_module)
    model = getattr(model_module, model_name)(**model_kwargs["kwargs"])

    # load the dataloader
    anndataset_module = importlib.import_module(dataset_kwargs["name"])
    anndata = anndataset_module.load(**dataset_kwargs["kwargs"])

    _, anndata_splits = utils.data.split_anndata(
        anndata, test_size=0.2, random_state=seed
    )

    ds_train = gsdataset_from_anndata(anndata_splits["train"])
    ds_validate = gsdataset_from_anndata(anndata_splits["test"])

    ds_train, ds_validate = utils.data.transform(
        ds_train, ds_validate, transform_method
    )

    X = ds_train.X

    model.fit(X)

    pickle.dump(model, open("{}/model.p".format(save_dir), "wb"), protocol=4)

    plots.PCA_explained_variance(
        model, "{}/pca_explained_variance.png".format(save_dir)
    )

    dims_to_plot = [0, 1]

    day_as_str = ds_train.obs.day.values
    day = [int(d[1:]) for d in day_as_str]

    Y_dict = {}
    Y_dict["day"] = day

    for column in [
        "cell_line",
        "passage",
        "protocol",
        "sample_num",
        "scientist",
        "seq_exp",
    ]:
        Y_dict[column] = ds_train.obs[column].values

    plots.PCA_dims_multi_y(
        model,
        X=X,
        Y_dict=Y_dict,
        save_path_format="{}/pca_dims_by_{}.png".format(save_dir, "{}"),
        dims=dims_to_plot,
    )


if __name__ == "__main__":

    json_input = sys.argv[1]

    with open(json_input, "rb") as f:
        kwargs = json.load(f)

    # run it
    run(**kwargs)
