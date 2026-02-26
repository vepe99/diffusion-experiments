import marimo

__generated_with = "0.19.5"
app = marimo.App()


@app.cell
def _():
    # For multi-objective studies
    import optuna

    from optuna.storages import JournalStorage, JournalFileStorage

    from optuna.visualization import plot_pareto_front, plot_param_importances
    storage = JournalStorage(JournalFileStorage("../data/hyperparameter_tuning/optuna_diffusionmodel.log"))
    study = optuna.load_study(
        # study_name="study_CompositionalDiffusionModel",
        study_name='study_DiffusionModel',
        storage=storage,
    )
    fig = plot_pareto_front(study, target_names=["RMSE", "Calibration Error"])
    fig.show()

    fig = plot_param_importances(study, target=lambda t: t.values[0], target_name="RMSE")
    fig.show()

    fig = plot_param_importances(study, target=lambda t: t.values[1], target_name="Calibration Error")
    fig.show()
    return


if __name__ == "__main__":
    app.run()
