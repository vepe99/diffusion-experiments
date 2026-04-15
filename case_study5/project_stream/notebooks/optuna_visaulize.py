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
    # storage = JournalStorage(JournalFileStorage("../data/hyperparameter_tuning/gala/optuna_diffusionmodel_gala_cutNGC3201.log"))
    # storage = JournalStorage(JournalFileStorage("../data/hyperparameter_tuning/gala/local/optuna_diffusionmodel_galax_local_cutNGC3201.log"))
    # storage = JournalStorage(JournalFileStorage("../data/hyperparameter_tuning/gala/local/optuna_diffusionmodel_gala_local_cutNGC3201_300k.log"))
    # storage = JournalStorage(JournalFileStorage("../data/hyperparameter_tuning/gala/big_heads/optuna_diffusionmodel_gala_cutNGC3201.log"))
    # storage = JournalStorage(JournalFileStorage("../data/hyperparameter_tuning/gala/new_aug/optuna_diffusionmodel_gala_cutNGC3201.log"))
    storage = JournalStorage(JournalFileStorage("../data/hyperparameter_tuning/gala/local/new_aug_jonas/optuna_diffusionmodel_gala_local_cutNGC3201.log"))

    # study_names = optuna.study.get_all_study_names(storage)
    # print("Studies in storage:", study_names)
    study = optuna.load_study(
        # study_name='study_DiffusionModel',
        study_name="study_DiffusionMode_local",
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

    # return
    # storage = JournalStorage(JournalFileStorage("../data/hyperparameter_tuning/new_bf/optuna_diffusionmodel_galax_cutNGC3201.log"))
    
    # # Load both studies
    # study_small = optuna.load_study(study_name='study_DiffusionModel_small', storage=storage)
    # study_full = optuna.load_study(study_name='study_DiffusionModel', storage=storage)
    
    # # Merge trials into a new study
    # merged_study = optuna.create_study(
    #     directions=['minimize', 'minimize'],
    #     study_name='study_DiffusionModel_merged'
    # )
    
    # # Add all trials from both studies
    # for trial in study_small.trials + study_full.trials:
    #     merged_study.add_trial(trial)
    
    # # Visualize merged study
    # fig = plot_pareto_front(merged_study, target_names=["RMSE", "Calibration Error"])
    # fig.show()

    # fig = plot_param_importances(merged_study, target=lambda t: t.values[0], target_name="RMSE")
    # fig.show()

    # fig = plot_param_importances(merged_study, target=lambda t: t.values[1], target_name="Calibration Error")
    # fig.show()

    return


# @app.cell
# def _():
#     return


# if __name__ == "__main__":
#     app.run()
if __name__ == "__main__":
    main()
