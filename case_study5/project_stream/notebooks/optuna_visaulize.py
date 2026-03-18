# import marimo

# __generated_with = "0.19.5"
# app = marimo.App()


# @app.cell
# def _():
def main():
    # For multi-objective studies
    import matplotlib.pyplot as plt
    import optuna

    from optuna.storages import JournalStorage, JournalFileStorage

    from optuna.visualization import plot_pareto_front, plot_param_importances
    storage = JournalStorage(JournalFileStorage("../data/hyperparameter_tuning/optuna_diffusionmodel_galax.log"))
    study = optuna.load_study(
        # study_name="study_CompositionalDiffusionModel",
        study_name='study_DiffusionModel',
        storage=storage,
    )
    fig = plot_pareto_front(study, target_names=["RMSE", "Calibration Error"])
    # plt.savefig('plots_optuna/pareto_front.png')
    fig.show()

    fig = plot_param_importances(study, target=lambda t: t.values[0], target_name="RMSE")
    # plt.savefig('plots_optuna/param_importances_rmse.png')
    fig.show()

    fig = plot_param_importances(study, target=lambda t: t.values[1], target_name="Calibration Error")
    # plt.savefig('plots_optuna/param_importances_calibration_error.png')
    fig.show()

    return


# @app.cell
# def _():
#     return


# if __name__ == "__main__":
#     app.run()
if __name__ == "__main__":
    main()
