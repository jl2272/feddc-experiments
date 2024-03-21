
# Project Details:

All details can be found in the report (FedDC Experiments.pdf)

# Reproducibility

As described in the report:

Care has been taken to ensure the experiments performed are reproducible. Within the source code (project/conf) are 4 .yaml files, each corresponding to an experiment or set of experiments. Running project.main with poetry and these config files will carry out the same experiments described within this report and save the results. The graphs and tables can be produced with the data_analysis notebook contained within the plotting subfolder.

mnist_8_samples.yaml runs three experiments on an IID partition of the MNIST train set assigning 8 samples to 25 clients. It runs these with three different client selection seeds, and on six different strategies. It also runs the same experiments on the non-IID partition of the MNIST train set.

mnist_8_samples_centralised.yaml runs the same experiments on both datasets as described previously, but with centralised training instead. Therefore, the client selection seed is not changed.

mnist_large_and_small.yaml again runs three experiments. This time assigning 8 samples to 20 clients and 1200 samples to 5 clients. Three client selection seeds are used again. This time two extra strategies are tested, those being FedDCW with b=5 and b=10.

Finally, mnist_large_and_small_centralised.yaml runs using the same dataset described within the previous experiment, but with centralised training.

All experiments use the same seed for creating the dataset partition dynamically to ensure we are comparing the performance on the same data each time.